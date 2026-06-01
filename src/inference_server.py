
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torchaudio


APP_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(APP_DIR / "ltx2"))
sys.path.insert(0, str(APP_DIR / "src"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from audio_conditioning import AudioConditionByReferenceLatent
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.loader import DummyRegistry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.transformer.model import LTXModel, LTXModelType, X0Model
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.text_projection import create_caption_projection
from ltx_core.model.transformer.attention import AttentionFunction
from ltx_core.model.model_protocol import ModelConfigurator
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_pipelines.utils.blocks import AudioConditioner, AudioDecoder, PromptEncoder
from ltx_pipelines.utils.media_io import decode_audio_from_file
from ltx_pipelines.utils.denoisers import GuidedDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop
from safetensors import safe_open


DEFAULT_NEG = "worst quality, inconsistent, robotic, distorted, noise, static, muffled, unclear, unnatural, monotone"


def estimate_duration(prompt, multiplier=1.1):
    from duration_estimator import estimate_speech_duration
    base = estimate_speech_duration(prompt)
    return max(3.0, round(base * multiplier, 1))


def _equal_power_crossfade(prev: torch.Tensor, nxt: torch.Tensor,
                           sample_rate: int, fade_ms: float = 50.0) -> torch.Tensor:
    fade_samples = int(round(fade_ms * 1e-3 * sample_rate))
    fade_samples = max(1, min(fade_samples, prev.shape[-1], nxt.shape[-1]))
    if fade_samples <= 1:
        return torch.cat([prev, nxt], dim=-1)

    t = torch.linspace(0.0, 1.0, fade_samples, device=prev.device, dtype=prev.dtype)
    fade_out = torch.cos(t * torch.pi / 2)
    fade_in = torch.sin(t * torch.pi / 2)

    prev_tail = prev[..., -fade_samples:] * fade_out
    nxt_head = nxt[..., :fade_samples] * fade_in
    mixed = prev_tail + nxt_head
    return torch.cat([prev[..., :-fade_samples], mixed, nxt[..., fade_samples:]], dim=-1)


def auto_rescale_for_cfg(cfg: float) -> float:
    if cfg <= 2.0:
        return 0.0
    if cfg <= 3.0:
        return 0.6 * (cfg - 2.0)
    if cfg <= 4.0:
        return 0.6 + 0.2 * (cfg - 3.0)
    if cfg <= 8.0:
        return 0.8
    return min(1.0, 0.8 + 0.1 * (cfg - 8.0))


class TTSServer:
    def __init__(self, checkpoint=None, full_checkpoint=None, gemma_root=None,
                 device="cuda", dtype="bf16", compile_model=True, bnb_4bit=True):
        MODELS = APP_DIR / "models"
        self.checkpoint = checkpoint or str(MODELS / "ltx-2.3-22b-dev-audio-only-v13-merged.safetensors")
        self.full_checkpoint = full_checkpoint or os.environ.get(
            "LTX_FULL_CHECKPOINT", str(MODELS / "ltx-2.3-22b-dev.safetensors"))
        if gemma_root is None and not os.environ.get("GEMMA_DIR"):
            from model_downloader import get_gemma_path
            gemma_root = get_gemma_path()
        self.gemma_root = gemma_root or os.environ["GEMMA_DIR"]
        self.device = torch.device(device)
        self.dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        self.compile_model = compile_model
        self.bnb_4bit = bnb_4bit
        self.patchifier = AudioPatchifier(patch_size=1)


        self._prompt_encoder = None
        self._velocity_model = None
        self._audio_conditioner = None
        self._audio_decoder = None



        self._ref_denoiser = None
        self._ref_denoise_cache: dict[tuple, "torch.Tensor"] = {}

        logging.info(f"TTSServer loading on {device}...")
        t0 = time.time()
        self._load_all()
        logging.info(f"All models loaded in {time.time()-t0:.1f}s — ready for requests")

    def _load_all(self):

        t0 = time.time()
        self._prompt_encoder = PromptEncoder(
            checkpoint_path=self.full_checkpoint,
            gemma_root=self.gemma_root,
            dtype=self.dtype, device=self.device,
            warm=True,
            use_bnb_4bit=self.bnb_4bit,
            audio_only=True,
        )
        logging.info(f"  PromptEncoder (warm): {time.time()-t0:.1f}s")


        t0 = time.time()
        self._audio_conditioner = AudioConditioner(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=self.device,
            warm=True,
        )
        logging.info(f"  AudioConditioner (warm): {time.time()-t0:.1f}s")


        t0 = time.time()
        with safe_open(self.checkpoint, framework="pt") as f:
            config = json.loads(f.metadata()["config"])

        t = config.get("transformer", {})

        class AudioOnlyConfigurator(ModelConfigurator[LTXModel]):
            @classmethod
            def from_config(cls, cfg):
                t = cfg.get("transformer", {})
                cp = None
                if not t.get("caption_proj_before_connector", False):
                    with torch.device("meta"):
                        cp = create_caption_projection(t, audio=True)
                return LTXModel(
                    model_type=LTXModelType.AudioOnly,
                    audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
                    audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
                    audio_in_channels=t.get("audio_in_channels", 128),
                    audio_out_channels=t.get("audio_out_channels", 128),
                    num_layers=t.get("num_layers", 48),
                    audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
                    norm_eps=t.get("norm_eps", 1e-6),
                    attention_type=AttentionFunction(t.get("attention_type", "default")),
                    positional_embedding_theta=10000.0,
                    audio_positional_embedding_max_pos=[20.0],
                    timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                    use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                    rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
                    double_precision_rope=t.get("frequencies_precision", False) == "float64",
                    apply_gated_attention=t.get("apply_gated_attention", False),
                    audio_caption_projection=cp,
                    cross_attention_adaln=t.get("cross_attention_adaln", False),
                )

        audio_sd_ops = SDOps("AO").with_matching(prefix="model.diffusion_model.").with_replacement(
            "model.diffusion_model.", "")
        builder = Builder(
            model_path=self.checkpoint,
            model_class_configurator=AudioOnlyConfigurator,
            model_sd_ops=audio_sd_ops,
            registry=DummyRegistry(),
        )
        self._velocity_model = builder.build(device=self.device, dtype=self.dtype).to(self.device).eval()
        n_params = sum(p.numel() for p in self._velocity_model.parameters()) / 1e9
        vram_gb = sum(p.numel() * p.element_size() for p in self._velocity_model.parameters()) / 1e9
        logging.info(f"  Transformer: {time.time()-t0:.1f}s ({n_params:.1f}B params, {vram_gb:.1f}GB VRAM, {self.dtype})")


        if self.compile_model:
            t0 = time.time()
            logging.info("  Compiling transformer with torch.compile (default mode)...")
            self._velocity_model = torch.compile(self._velocity_model, mode="default", dynamic=True)
            logging.info(f"  Compiled: {time.time()-t0:.1f}s (first call triggers actual compilation)")


        t0 = time.time()
        self._audio_decoder = AudioDecoder(
            checkpoint_path=self.full_checkpoint,
            dtype=self.dtype, device=self.device,
            warm=True,
        )
        logging.info(f"  AudioDecoder (warm): {time.time()-t0:.1f}s")

    def _denoise_voice_ref(self, voice, voice_ref_path: str, ref_duration: float):
        cache_key = (voice_ref_path, float(ref_duration), int(voice.sampling_rate))
        if cache_key in self._ref_denoise_cache:
            return Audio(
                waveform=self._ref_denoise_cache[cache_key],
                sampling_rate=voice.sampling_rate,
            )





        if self._ref_denoiser is None:
            from super_resolution import REUSEUpsampler
            try:
                self._ref_denoiser = REUSEUpsampler(
                    target_sr=int(voice.sampling_rate),
                    device=self.device,
                    chunk_size_s=1.0,
                )
            except Exception as e:


                logging.warning(f"Voice-ref denoise disabled (RE-USE unavailable: {e})")
                self._ref_denoiser = False
                return voice

        if self._ref_denoiser is False:
            return voice

        w = voice.waveform


        if w.dim() == 3:
            mono = w[0].mean(dim=0)
        elif w.dim() == 2:
            mono = w.mean(dim=0)
        else:
            mono = w
        mono = mono.contiguous()

        t0 = time.time()
        cleaned, _ = self._ref_denoiser(mono, in_sr=int(voice.sampling_rate))
        if cleaned.dim() == 2 and cleaned.shape[0] == 1:
            cleaned = cleaned[0]


        cleaned = cleaned.unsqueeze(0).unsqueeze(0).to(self.device, dtype=w.dtype)
        logging.info(f"Voice-ref denoise (RE-USE): {time.time() - t0:.2f}s")

        self._ref_denoise_cache[cache_key] = cleaned
        return Audio(waveform=cleaned, sampling_rate=voice.sampling_rate)

    @torch.inference_mode()
    def generate(self, prompt, voice_ref=None, cfg_scale=2.5, stg_scale=1.5,
                 duration_multiplier=1.1, seed=42, ref_duration=10.0,
                 rescale_scale="auto", gen_duration: float = 0.0,
                 denoise_ref: bool = True):
        t_total = time.time()


        if gen_duration and gen_duration > 0:
            gen_dur = float(gen_duration)
        else:
            gen_dur = estimate_duration(prompt, duration_multiplier)
        fps = 25.0
        n_frames = int(round(gen_dur * fps)) + 1
        n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
        pixel_shape = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
        target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(patchifier=self.patchifier, target_shape=target_shape)


        state = audio_tools.create_initial_state(device=self.device, dtype=self.dtype)


        if voice_ref and os.path.exists(voice_ref):
            t0 = time.time()
            voice = decode_audio_from_file(voice_ref, self.device, 0.0, ref_duration)
            if denoise_ref:
                voice = self._denoise_voice_ref(voice, voice_ref, ref_duration)
            w = voice.waveform
            if w.dim() == 2:
                if w.shape[0] == 1:
                    w = w.repeat(2, 1)
                w = w.unsqueeze(0)
            elif w.dim() == 3 and w.shape[1] == 1:
                w = w.repeat(1, 2, 1)
            target_samples = int(ref_duration * voice.sampling_rate)
            if w.shape[-1] < target_samples:
                w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
            w = w[..., :target_samples]
            peak = w.abs().max()
            if peak > 0:
                w = w * (10 ** (-4.0 / 20) / peak)
            voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)
            ref_latent = self._audio_conditioner(lambda enc: vae_encode_audio(voice, enc, None))
            cond = AudioConditionByReferenceLatent(latent=ref_latent.to(self.device, self.dtype), strength=1.0)
            state = cond.apply_to(state, audio_tools)
            logging.info(f"Voice ref: {time.time()-t0:.2f}s")


        gen = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=gen)
        state = noiser(state, noise_scale=1.0)


        t0 = time.time()
        prompts = [prompt, DEFAULT_NEG] if cfg_scale > 1.0 else [prompt]
        ctx = self._prompt_encoder(prompts, streaming_prefetch_count=None)
        a_ctx = ctx[0].audio_encoding
        a_ctx_neg = ctx[1].audio_encoding if cfg_scale > 1.0 else None
        logging.info(f"Prompt: {time.time()-t0:.2f}s")


        resc = auto_rescale_for_cfg(cfg_scale) if rescale_scale == "auto" else float(rescale_scale)
        if rescale_scale == "auto":
            logging.info(f"Auto rescale_scale = {resc:.2f} for cfg={cfg_scale}")
        guider = MultiModalGuider(
            params=MultiModalGuiderParams(
                cfg_scale=cfg_scale, stg_scale=stg_scale,
                stg_blocks=[29], rescale_scale=resc, modality_scale=1.0,
            ),
            negative_context=a_ctx_neg,
        )
        denoiser = GuidedDenoiser(
            v_context=None, a_context=a_ctx,
            video_guider=None, audio_guider=guider,
        )


        sigmas = LTX2Scheduler().execute(steps=30, latent=state.latent).to(self.device)


        t0 = time.time()
        x0 = X0Model(self._velocity_model)
        _, audio_state = euler_denoising_loop(
            sigmas=sigmas, video_state=None, audio_state=state,
            stepper=EulerDiffusionStep(), transformer=x0, denoiser=denoiser,
        )
        logging.info(f"Denoise (30 steps): {time.time()-t0:.2f}s")


        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)











        latent = audio_state.latent
        if latent.shape[2] > 513:
            f0, f1 = 511, 514
            n = f1 - f0
            patched = latent.clone()
            for f in (512, 513):
                t = (f - f0) / n
                patched[:, :, f, :] = (1.0 - t) * latent[:, :, f0, :] + t * latent[:, :, f1, :]
            latent = patched

        t0 = time.time()
        decoded = self._audio_decoder(latent)
        out_waveform, out_sr = decoded.waveform, decoded.sampling_rate
        logging.info(f"Decode (LTX BWE): {time.time()-t0:.2f}s")

        total = time.time() - t_total
        dur = out_waveform.shape[-1] / out_sr
        logging.info(f"Total: {total:.2f}s for {dur:.1f}s audio")
        return out_waveform, out_sr

    @torch.inference_mode()
    def generate_long(self, prompt, max_chunk_duration: float = 45.0,
                      target_chunk_duration: float = 37.0,
                      crossfade_ms: float = 50.0,
                      progress_callback=None,
                      **kwargs):
        from text_chunker import chunk_prompt_for_duration



        per_chunk_mul = float(kwargs.pop("duration_multiplier", 1.1))


        kwargs.pop("gen_duration", None)

        chunks = chunk_prompt_for_duration(
            prompt,
            max_duration_s=max_chunk_duration,
            target_duration_s=target_chunk_duration,
            duration_multiplier=per_chunk_mul,
        )
        logging.info(f"Long-form: {len(chunks)} chunks (target {target_chunk_duration:.0f}s, "
                     f"max {max_chunk_duration:.0f}s)")

        out_waveform: Optional[torch.Tensor] = None
        out_sr: Optional[int] = None
        t_total = time.time()
        for idx, chunk in enumerate(chunks):
            logging.info(f"  Chunk {idx + 1}/{len(chunks)}: est {chunk.est_duration_s:.1f}s, "
                         f"{len(chunk.text)} chars")
            if progress_callback is not None:
                try:
                    progress_callback(idx, len(chunks), chunk.est_duration_s)
                except Exception as e:
                    logging.warning(f"progress_callback raised, ignoring: {e}")
            wav, sr = self.generate(
                chunk.text,
                duration_multiplier=per_chunk_mul,
                **kwargs,
            )
            wav = wav.cpu().float()
            if out_waveform is None:
                out_waveform, out_sr = wav, sr
            else:
                if sr != out_sr:
                    raise RuntimeError(f"Sample-rate mismatch between chunks: {out_sr} vs {sr}")


                if wav.shape[0] != out_waveform.shape[0]:
                    if wav.shape[0] == 1:
                        wav = wav.repeat(out_waveform.shape[0], 1)
                    elif out_waveform.shape[0] == 1:
                        out_waveform = out_waveform.repeat(wav.shape[0], 1)
                out_waveform = _equal_power_crossfade(out_waveform, wav, out_sr,
                                                      fade_ms=crossfade_ms)

        total_dur = out_waveform.shape[-1] / out_sr
        logging.info(f"Long-form total: {time.time() - t_total:.2f}s wall, {total_dur:.1f}s audio")
        return out_waveform, out_sr

    def generate_to_file(self, prompt, output, watermark: bool = True,
                         max_chunk_duration: float = 45.0,
                         target_chunk_duration: float = 37.0,
                         crossfade_ms: float = 50.0,
                         progress_callback=None,
                         **kwargs):




        explicit_dur = float(kwargs.get("gen_duration") or 0.0)
        est_dur = explicit_dur if explicit_dur > 0 else estimate_duration(
            prompt, kwargs.get("duration_multiplier", 1.1))

        if est_dur > max_chunk_duration:
            waveform, sr = self.generate_long(
                prompt,
                max_chunk_duration=max_chunk_duration,
                target_chunk_duration=target_chunk_duration,
                crossfade_ms=crossfade_ms,
                progress_callback=progress_callback,
                **kwargs,
            )
        else:
            if progress_callback is not None:
                try:
                    progress_callback(0, 1, est_dur)
                except Exception:
                    pass
            waveform, sr = self.generate(prompt, **kwargs)
        wav_cpu = waveform.cpu().float()
        if watermark:
            try:
                import numpy as np, perth
                if not hasattr(self, "_perth"):
                    self._perth = perth.PerthImplicitWatermarker()
                mono = wav_cpu.mean(dim=0).numpy() if wav_cpu.shape[0] > 1 else wav_cpu[0].numpy()
                mono_wm = self._perth.apply_watermark(mono, sample_rate=sr)
                mono_wm_t = torch.from_numpy(np.asarray(mono_wm, dtype=np.float32)).unsqueeze(0)
                wav_cpu = mono_wm_t if wav_cpu.shape[0] == 1 else mono_wm_t.repeat(wav_cpu.shape[0], 1)
            except Exception as e:
                logging.warning(f"Perth watermark skipped ({e})")
        torchaudio.save(output, wav_cpu, sr)
        logging.info(f"Saved: {output}")
        return output


if __name__ == "__main__":
    import argparse
    import tempfile
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--no-compile", action="store_true")
    p.add_argument("--no-bnb-4bit", action="store_true",
                   help="Disable bitsandbytes 4-bit path (default: on, since the default "
                        "unsloth Gemma checkpoint is pre-quantized).")
    p.add_argument("--warmup-voice-ref-1", default=os.environ.get("WARMUP_VOICE_REF_1"),
                   help="Voice reference for the first warmup pass. If unset, the warmup "
                        "pass is skipped. Env var: WARMUP_VOICE_REF_1.")
    p.add_argument("--warmup-voice-ref-2", default=os.environ.get("WARMUP_VOICE_REF_2"),
                   help="Voice reference for the second warmup pass. If unset, the warmup "
                        "pass is skipped. Env var: WARMUP_VOICE_REF_2.")
    p.add_argument("--warmup-output-dir", default=os.environ.get("WARMUP_OUTPUT_DIR", tempfile.gettempdir()),
                   help="Directory to write warmup test outputs. "
                        "Defaults to the system temp dir. Env var: WARMUP_OUTPUT_DIR.")
    args = p.parse_args()

    server = TTSServer(device=args.device, dtype=args.dtype, compile_model=not args.no_compile,
                       bnb_4bit=not args.no_bnb_4bit)


    if args.warmup_voice_ref_1:
        logging.info("=== First request ===")
        server.generate_to_file(
            prompt='A woman speaks clearly, "The weather today will be sunny."',
            output=os.path.join(args.warmup_output_dir, "warm_test1.wav"),
            voice_ref=args.warmup_voice_ref_1,
        )
    else:
        logging.info("Skipping first warmup pass (no --warmup-voice-ref-1 / WARMUP_VOICE_REF_1).")


    if args.warmup_voice_ref_2:
        logging.info("\n=== Second request (warm) ===")
        server.generate_to_file(
            prompt='A man speaks excitedly, "This is amazing, I cannot believe it!"',
            output=os.path.join(args.warmup_output_dir, "warm_test2.wav"),
            voice_ref=args.warmup_voice_ref_2,
        )
    else:
        logging.info("Skipping second warmup pass (no --warmup-voice-ref-2 / WARMUP_VOICE_REF_2).")
