import logging
from collections.abc import Iterator

import torch
from einops import rearrange
from safetensors import safe_open

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import (
    ConditioningItem,
    ConditioningItemAttentionStrengthWrapper,
    VideoConditionByReferenceLatent,
)
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig, VideoEncoder, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    VideoConditioningAction,
    VideoMaskConditioningAction,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings, get_device
from ltx_pipelines.utils.media_io import decode_video_by_frame, encode_video, video_preprocess
from ltx_pipelines.utils.types import ModalitySpec


class ICLoraPipeline:

    def __init__(
        self,
        distilled_checkpoint_path: str,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ):
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(
            distilled_checkpoint_path, gemma_root, self.dtype, self.device, registry=registry
        )
        self.image_conditioner = ImageConditioner(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.stage_1 = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.stage_2 = DiffusionStage(
            distilled_checkpoint_path,
            self.dtype,
            self.device,
            loras=(),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.upsampler = VideoUpsampler(
            distilled_checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(distilled_checkpoint_path, self.dtype, self.device, registry=registry)




        self.reference_downscale_factor = 1
        for lora in loras:
            scale = _read_lora_reference_downscale_factor(lora.path)
            if scale != 1:
                if self.reference_downscale_factor not in (1, scale):
                    raise ValueError(
                        f"Conflicting reference_downscale_factor values in LoRAs: "
                        f"already have {self.reference_downscale_factor}, but {lora.path} "
                        f"specifies {scale}. Cannot combine LoRAs with different reference scales."
                    )
                self.reference_downscale_factor = scale

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        conditioning_attention_mask: torch.Tensor | None = None,
        streaming_prefetch_count: int | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], got {conditioning_attention_strength}"
            )

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.prompt_encoder(
            [prompt],
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
            streaming_prefetch_count=streaming_prefetch_count,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding


        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )


        stage_1_conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                images=images,
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=enc,
                num_frames=num_frames,
                conditioning_attention_strength=conditioning_attention_strength,
                conditioning_attention_mask=conditioning_attention_mask,
            )
        )

        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        video_state, audio_state = self.stage_1(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(
                context=audio_context,
            ),
            streaming_prefetch_count=streaming_prefetch_count,
        )

        if skip_stage_2:

            logging.info("[IC-LoRA] Skipping Stage 2 (--skip-stage-2 enabled)")
            decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent)
            return decoded_video, decoded_audio


        upscaled_video_latent = self.upsampler(video_state.latent[:1])

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        stage_2_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=enc,
                dtype=self.dtype,
                device=self.device,
            )
        )

        video_state, audio_state = self.stage_2(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=distilled_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage_2_conditionings,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
            streaming_prefetch_count=streaming_prefetch_count,
        )

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio

    def _create_conditionings(
        self,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        conditioning_attention_strength: float = 1.0,
        conditioning_attention_mask: torch.Tensor | None = None,
    ) -> list[ConditioningItem]:
        conditionings = combined_image_conditionings(
            images=images,
            height=height,
            width=width,
            video_encoder=video_encoder,
            dtype=self.dtype,
            device=self.device,
        )



        scale = self.reference_downscale_factor
        if scale != 1 and (height % scale != 0 or width % scale != 0):
            raise ValueError(
                f"Output dimensions ({height}x{width}) must be divisible by reference_downscale_factor ({scale})"
            )
        ref_height = height // scale
        ref_width = width // scale

        for video_path, strength in video_conditioning:

            frame_gen = decode_video_by_frame(path=video_path, frame_cap=num_frames, device=self.device)
            video = video_preprocess(frame_gen, ref_height, ref_width, self.dtype, self.device)
            encoded_video = video_encoder(video)
            reference_video_shape = VideoLatentShape.from_torch_shape(encoded_video.shape)


            if conditioning_attention_mask is not None:

                latent_mask = self._downsample_mask_to_latent(
                    mask=conditioning_attention_mask,
                    target_latent_shape=reference_video_shape,
                )
                attn_mask = latent_mask * conditioning_attention_strength
            elif conditioning_attention_strength < 1.0:

                attn_mask = conditioning_attention_strength
            else:
                attn_mask = None

            cond = VideoConditionByReferenceLatent(
                latent=encoded_video,
                downscale_factor=scale,
                strength=strength,
            )
            if attn_mask is not None:
                cond = ConditioningItemAttentionStrengthWrapper(cond, attention_mask=attn_mask)
            conditionings.append(cond)

        if video_conditioning:
            logging.info(f"[IC-LoRA] Added {len(video_conditioning)} video conditioning(s)")

        return conditionings

    @staticmethod
    def _downsample_mask_to_latent(
        mask: torch.Tensor,
        target_latent_shape: VideoLatentShape,
    ) -> torch.Tensor:
        b = mask.shape[0]
        f_lat = target_latent_shape.frames
        h_lat = target_latent_shape.height
        w_lat = target_latent_shape.width


        f_pix = mask.shape[2]
        spatial_down = torch.nn.functional.interpolate(
            rearrange(mask, "b 1 f h w -> (b f) 1 h w"),
            size=(h_lat, w_lat),
            mode="area",
        )
        spatial_down = rearrange(spatial_down, "(b f) 1 h w -> b 1 f h w", b=b)



        first_frame = spatial_down[:, :, :1, :, :]

        if f_pix > 1 and f_lat > 1:

            t = (f_pix - 1) // (f_lat - 1)
            assert (f_pix - 1) % (f_lat - 1) == 0, (
                f"Pixel frames ({f_pix}) not compatible with latent frames ({f_lat}): "
                f"(f_pix - 1) must be divisible by (f_lat - 1)"
            )
            rest = rearrange(spatial_down[:, :, 1:, :, :], "b 1 (f t) h w -> b 1 f t h w", t=t)
            rest = rest.mean(dim=3)
            latent_mask = torch.cat([first_frame, rest], dim=2)
        else:
            latent_mask = first_frame


        return rearrange(latent_mask, "b 1 f h w -> b (f h w)")


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    parser.add_argument(
        "--video-conditioning",
        action=VideoConditioningAction,
        nargs=2,
        metavar=("PATH", "STRENGTH"),
        required=True,
    )
    parser.add_argument(
        "--conditioning-attention-mask",
        action=VideoMaskConditioningAction,
        nargs=2,
        metavar=("MASK_PATH", "STRENGTH"),
        default=None,
        help=(
            "Optional spatial attention mask: path to a grayscale mask video and "
            "attention strength. The mask video pixel values in [0,1] control "
            "per-region conditioning attention strength. The strength scalar is "
            "multiplied with the spatial mask. "
            "0.0 = ignore IC-LoRA conditioning, 1.0 = full conditioning influence. "
            "When not provided, full conditioning strength (1.0) is used. "
            "Example: --conditioning-attention-mask path/to/mask.mp4 0.5"
        ),
    )
    parser.add_argument(
        "--skip-stage-2",
        action="store_true",
        help=(
            "Skip Stage 2 upsampling and refinement. Output will be at half resolution "
            "(height//2, width//2). Useful for faster iteration or when GPU memory is limited."
        ),
    )
    args = parser.parse_args()


    conditioning_attention_mask = None
    conditioning_attention_strength = 1.0
    if args.conditioning_attention_mask is not None:
        mask_path, mask_strength = args.conditioning_attention_mask
        conditioning_attention_strength = mask_strength
        conditioning_attention_mask = _load_mask_video(
            mask_path=mask_path,
            height=args.height // 2,
            width=args.width // 2,
            num_frames=args.num_frames,
        )

    pipeline = ICLoraPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        torch_compile=args.compile,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        video_conditioning=args.video_conditioning,
        tiling_config=tiling_config,
        conditioning_attention_strength=conditioning_attention_strength,
        skip_stage_2=args.skip_stage_2,
        conditioning_attention_mask=conditioning_attention_mask,
        streaming_prefetch_count=args.streaming_prefetch_count,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


def _load_mask_video(
    mask_path: str,
    height: int,
    width: int,
    num_frames: int,
) -> torch.Tensor:
    device = get_device()
    frame_gen = decode_video_by_frame(path=mask_path, frame_cap=num_frames, device=device)
    mask_video = video_preprocess(frame_gen, height, width, torch.bfloat16, device)

    mask = mask_video.mean(dim=1, keepdim=True)


    mask = (mask + 1.0) / 2.0
    return mask.clamp(0.0, 1.0)


def _read_lora_reference_downscale_factor(lora_path: str) -> int:
    try:
        with safe_open(lora_path, framework="pt") as f:
            metadata = f.metadata() or {}
            return int(metadata.get("reference_downscale_factor", 1))
    except Exception as e:
        logging.warning(f"Failed to read metadata from LoRA file '{lora_path}': {e}")
        return 1


if __name__ == "__main__":
    main()
