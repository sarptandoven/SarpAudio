
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import replace
from typing import Callable, TypeVar

import torch

from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import Noiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.layer_streaming import LayerStreamingWrapper
from ltx_core.loader import SDOps
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae import (
    decode_audio as vae_decode_audio,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.transformer.compiling import COMPILE_TRANSFORMER, modify_sd_ops_for_compilation
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.utils import find_matching_file
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    create_noised_state,
    generate_enhanced_prompt,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import Denoiser, ModalitySpec

logger = logging.getLogger(__name__)

T = TypeVar("T")
_M = TypeVar("_M", bound=torch.nn.Module)







@contextmanager
def _streaming_model(
    model: _M,
    layers_attr: str,
    target_device: torch.device,
    prefetch_count: int,
) -> Iterator[_M]:
    wrapped = LayerStreamingWrapper(
        model,
        layers_attr=layers_attr,
        target_device=target_device,
        prefetch_count=prefetch_count,
    )
    try:
        yield wrapped
    finally:
        wrapped.teardown()
        wrapped.to("meta")
        cleanup_memory()




        torch.cuda.synchronize(device=target_device)
        try:
            if hasattr(torch._C, "_host_emptyCache"):
                torch._C._host_emptyCache()
        except Exception:
            logger.warning("Host empty cache cleanup failed; ignoring.", exc_info=True)


def _build_state(
    spec: ModalitySpec,
    tools: LatentTools,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
) -> LatentState:
    state = create_noised_state(
        tools=tools,
        conditionings=spec.conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=spec.noise_scale,
        initial_latent=spec.initial_latent,
    )
    if spec.frozen:
        state = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
    return state


def _cleanup_iter(it: Iterator[torch.Tensor], model: torch.nn.Module) -> Iterator[torch.Tensor]:
    with gpu_model(model):
        yield from it







class DiffusionStage:

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._quantization = quantization
        self._torch_compile = torch_compile
        self._transformer_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
            loras=tuple(loras),
            registry=registry or DummyRegistry(),
        )

    def _build_transformer(self, *, device: torch.device | None = None, **kwargs: object) -> X0Model:
        target = device or self._device
        sd_ops = self._transformer_builder.model_sd_ops
        module_ops = self._transformer_builder.module_ops
        loras = self._transformer_builder.loras
        if self._torch_compile:
            module_ops = (*module_ops, COMPILE_TRANSFORMER)
            number_of_layers = self._transformer_builder.model_config()["transformer"]["num_layers"]
            sd_ops = modify_sd_ops_for_compilation(sd_ops, number_of_layers)
            loras = tuple(
                LoraPathStrengthAndSDOps(
                    lora.path,
                    lora.strength,
                    modify_sd_ops_for_compilation(
                        lora.sd_ops if lora.sd_ops is not None else SDOps(name="identity"), number_of_layers
                    ),
                )
                for lora in loras
            )
        if self._quantization is not None:
            module_ops = (*module_ops, *self._quantization.module_ops)
            sd_ops = SDOps(
                name=f"sd_ops_chain_{sd_ops.name}+{self._quantization.sd_ops.name}",
                mapping=(*sd_ops.mapping, *self._quantization.sd_ops.mapping),
            )

        builder = self._transformer_builder.with_module_ops(module_ops).with_sd_ops(sd_ops).with_loras(loras)
        return X0Model(builder.build(device=target, **kwargs)).to(target).eval()

    def _transformer_ctx(
        self,
        streaming_prefetch_count: int | None,
        **kwargs: object,
    ) -> AbstractContextManager:
        if streaming_prefetch_count is not None:
            return _streaming_model(
                self._build_transformer(device=torch.device("cpu"), **kwargs),
                layers_attr="velocity_model.transformer_blocks",
                target_device=self._device,
                prefetch_count=streaming_prefetch_count,
            )
        return gpu_model(self._build_transformer(**kwargs))

    def __call__(
        self,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        streaming_prefetch_count: int | None = None,
        max_batch_size: int = 1,
    ) -> tuple[LatentState | None, LatentState | None]:
        if video is None and audio is None:
            raise ValueError("At least one of `video` or `audio` must be provided")

        if loop is None:
            loop = euler_denoising_loop

        if stepper is None:
            stepper = EulerDiffusionStep()

        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)

        video_state: LatentState | None = None
        video_tools: LatentTools | None = None
        if video is not None:
            v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
            video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)
            video_state = _build_state(video, video_tools, noiser, self._dtype, self._device)

        audio_state: LatentState | None = None
        audio_tools: LatentTools | None = None
        if audio is not None:
            a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
            audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
            audio_state = _build_state(audio, audio_tools, noiser, self._dtype, self._device)

        with self._transformer_ctx(streaming_prefetch_count, video_tools=video_tools) as base_transformer:
            transformer = BatchSplitAdapter(base_transformer, max_batch_size=max_batch_size)
            video_state, audio_state = loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                transformer=transformer,
                denoiser=denoiser,
            )


        if video_state is not None and video_tools is not None:
            video_state = video_tools.clear_conditioning(video_state)
            video_state = video_tools.unpatchify(video_state)

        if audio_state is not None and audio_tools is not None:
            audio_state = audio_tools.clear_conditioning(audio_state)
            audio_state = audio_tools.unpatchify(audio_state)

        return video_state, audio_state







class PromptEncoder:

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        warm: bool = False,
        use_bnb_4bit: bool = False,
        audio_only: bool = False,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._warm = warm
        self._use_bnb_4bit = use_bnb_4bit
        self._audio_only = audio_only

        module_ops = module_ops_from_gemma_root(gemma_root)
        model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
        weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]

        self._gemma_root = gemma_root
        self._text_encoder_builder = Builder(
            model_path=tuple(weight_paths),
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GEMMA_LLM_KEY_OPS,
            module_ops=(GEMMA_MODEL_OPS, *module_ops),
            registry=registry or DummyRegistry(),
        )
        self._embeddings_processor_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            registry=registry or DummyRegistry(),
        )


        self._warm_text_encoder = None
        self._warm_embeddings_processor = None
        if warm:
            if use_bnb_4bit:
                self._warm_text_encoder = self._load_bnb_4bit_encoder(gemma_root)
            else:
                self._warm_text_encoder = self._text_encoder_builder.build(
                    device=self._device, dtype=self._dtype
                ).eval()
            built_ep = self._embeddings_processor_builder.build(
                device=self._device, dtype=self._dtype
            )






            if audio_only:
                import logging as _log
                ep = built_ep
                freed = 0


                if ep.video_connector is not None:
                    try:
                        freed += sum(p.numel() * p.element_size() for p in ep.video_connector.parameters() if not p.is_meta)
                    except Exception:
                        pass
                    del ep.video_connector
                    ep.video_connector = None


                fe = ep.feature_extractor
                if hasattr(fe, 'video_aggregate_embed') and fe.video_aggregate_embed is not None:
                    try:
                        freed += sum(p.numel() * p.element_size() for p in fe.video_aggregate_embed.parameters() if not p.is_meta)
                    except Exception:
                        pass
                    out_features = fe.video_aggregate_embed.out_features
                    del fe.video_aggregate_embed

                    class _DummyVideoEmbed(torch.nn.Module):
                        def __init__(self, out_f):
                            super().__init__()
                            self.out_features = out_f
                        def forward(self, x):
                            return torch.zeros(x.shape[0], x.shape[1], self.out_features,
                                             device=x.device, dtype=x.dtype)
                    fe.video_aggregate_embed = _DummyVideoEmbed(out_features)


            self._warm_embeddings_processor = built_ep.to(self._device).eval()

            if audio_only and self._warm_embeddings_processor is not None:
                ep = self._warm_embeddings_processor


                _orig_create = ep.create_embeddings
                def _audio_only_create(video_features, audio_features, additive_attention_mask,
                                      _ep=ep):



                    m = additive_attention_mask
                    while m.dim() > 2:
                        m = m[:, 0]

                    binary_mask = (m >= -1.0).to(torch.int64)

                    audio_encoded = None
                    if _ep.audio_connector is not None:
                        audio_encoded, _ = _ep.audio_connector(audio_features, additive_attention_mask)
                    return video_features, audio_encoded, binary_mask
                ep.create_embeddings = _audio_only_create

                torch.cuda.empty_cache()
                import gc; gc.collect()
                _log.info(f"Audio-only mode: freed video components, saved {freed/1e9:.1f}GB VRAM")

    def _load_bnb_4bit_encoder(self, gemma_root: str):
        import json
        import logging
        import os
        from transformers import Gemma3ForConditionalGeneration, BitsAndBytesConfig
        from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
        from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder


        prequantized = False
        cfg_path = os.path.join(gemma_root, "config.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
                prequantized = "quantization_config" in cfg
            except Exception:
                pass

        from_kwargs = {
            "device_map": str(self._device),
            "torch_dtype": self._dtype,
        }
        if prequantized:
            logging.info(
                "Loading pre-quantized Gemma (bnb-4bit) — using checkpoint's own quantization_config"
            )
        else:
            logging.info("Loading Gemma with runtime bitsandbytes 4-bit quantization...")
            from_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._dtype,
            )
        hf_model = Gemma3ForConditionalGeneration.from_pretrained(gemma_root, **from_kwargs)
        tokenizer = LTXVGemmaTokenizer(
            str(find_matching_file(gemma_root, "tokenizer.model").parent), 1024
        )
        encoder = GemmaTextEncoder(model=hf_model, tokenizer=tokenizer, dtype=self._dtype)
        mem_gb = torch.cuda.memory_allocated(self._device) / 1e9
        logging.info(f"Gemma 4-bit loaded: {mem_gb:.1f}GB VRAM")
        return encoder

    def _text_encoder_ctx(
        self,
        streaming_prefetch_count: int | None,
    ) -> AbstractContextManager:
        if streaming_prefetch_count is not None:
            return _streaming_model(
                self._text_encoder_builder.build(device=torch.device("cpu"), dtype=self._dtype).eval(),
                layers_attr="model.model.language_model.layers",
                target_device=self._device,
                prefetch_count=streaming_prefetch_count,
            )
        return gpu_model(self._text_encoder_builder.build(device=self._device, dtype=self._dtype).eval())

    @contextmanager
    def _noop_ctx(self, model):
        yield model

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
        streaming_prefetch_count: int | None = None,
    ) -> list[EmbeddingsProcessorOutput]:
        if self._warm and self._warm_text_encoder is not None:

            text_encoder = self._warm_text_encoder
            raw_outputs = [text_encoder.encode(p) for p in prompts]
            embeddings_processor = self._warm_embeddings_processor
            return [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]


        with self._text_encoder_ctx(streaming_prefetch_count) as text_encoder:
            if enhance_first_prompt:
                prompts = list(prompts)
                prompts[0] = generate_enhanced_prompt(
                    text_encoder, prompts[0], enhance_prompt_image, seed=enhance_prompt_seed
                )
            raw_outputs = [text_encoder.encode(p) for p in prompts]

        with gpu_model(
            self._embeddings_processor_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        ) as embeddings_processor:
            return [embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs]







class ImageConditioner:

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def _build_encoder(self) -> VideoEncoder:
        return self._encoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()

    def __call__(self, fn: Callable[[VideoEncoder], T]) -> T:
        with gpu_model(self._build_encoder()) as encoder:
            return fn(encoder)







class VideoUpsampler:

    def __init__(
        self,
        checkpoint_path: str,
        upsampler_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._upsampler_builder = Builder(
            model_path=upsampler_path,
            model_class_configurator=LatentUpsamplerConfigurator,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        with (
            gpu_model(
                self._encoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
            ) as encoder,
            gpu_model(
                self._upsampler_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
            ) as upsampler,
        ):
            return upsample_video(latent=latent, video_encoder=encoder, upsampler=upsampler)







class VideoDecoder:

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._decoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoDecoderConfigurator,
            model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def __call__(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        return _cleanup_iter(decoder.decode_video(latent, tiling_config, generator), decoder)







class AudioDecoder:

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        warm: bool = False,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._warm = warm
        self._decoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._vocoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._warm_decoder = None
        self._warm_vocoder = None
        if warm:
            self._warm_decoder = self._decoder_builder.build(device=device, dtype=dtype).to(device).eval()
            self._warm_vocoder = self._vocoder_builder.build(device=device, dtype=dtype).to(device).eval()

    def __call__(self, latent: torch.Tensor) -> Audio:
        if self._warm and self._warm_decoder is not None:
            return vae_decode_audio(latent, self._warm_decoder, self._warm_vocoder)
        with (
            gpu_model(
                self._decoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
            ) as decoder,
            gpu_model(
                self._vocoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
            ) as vocoder,
        ):
            return vae_decode_audio(latent, decoder, vocoder)







class AudioConditioner:

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        warm: bool = False,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._warm = warm
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._warm_encoder = None
        if warm:
            self._warm_encoder = self._encoder_builder.build(device=device, dtype=dtype).to(device).eval()

    def __call__(self, fn: Callable[[torch.nn.Module], T]) -> T:
        if self._warm and self._warm_encoder is not None:
            return fn(self._warm_encoder)
        with gpu_model(
            self._encoder_builder.build(device=self._device, dtype=self._dtype).to(self._device).eval()
        ) as encoder:
            return fn(encoder)
