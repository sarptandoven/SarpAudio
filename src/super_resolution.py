from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch





_REUSE_DIR: Optional[Path] = None


def _resolve_reuse_dir() -> Path:
    global _REUSE_DIR
    if _REUSE_DIR is None:
        from model_downloader import get_reuse_code_path
        _REUSE_DIR = Path(get_reuse_code_path())
    return _REUSE_DIR






_REUSE_OPT_DEPS = ("librosa", "resampy", "mamba_ssm")


def _check_reuse_deps() -> None:
    missing = [m for m in _REUSE_OPT_DEPS if _module_missing(m)]
    if missing:


        pkgs = [m.replace("_", "-") for m in missing]
        raise ImportError(
            "Voice-reference denoising (RE-USE) requires optional dependencies "
            f"not currently installed: {', '.join(pkgs)}.\n\n"
            "    pip install -r requirements-reuse.txt\n\n"
            "These are opt-in because mamba-ssm and causal-conv1d have no "
            "pre-built wheels on macOS / Windows and require matching CUDA + "
            "nvcc on Linux. To run without RE-USE, pass denoise_ref=False or "
            "untick 'Denoise voice reference' in the Gradio UI."
        )


def _module_missing(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is None


class REUSEUpsampler:

    def __init__(
        self,
        target_sr: int = 48000,
        config_path: Optional[str] = None,
        chunk_size_s: float = 1.0,
        hop_portion: float = 0.5,
        device: str | torch.device = "cuda",
    ) -> None:




        _check_reuse_deps()




        self.device = torch.device(device)
        self.target_sr = int(target_sr)
        self.chunk_size_s = float(chunk_size_s)
        self.hop_portion = float(hop_portion)


        self._config_path_override = Path(config_path) if config_path else None
        self.config_path: Optional[Path] = None
        self._model = None
        self._cfg = None
        self._stft_fns = None

    @staticmethod
    def _ensure_mamba_ssm_importable() -> None:
        try:
            import selective_scan_cuda
            import mamba_ssm
            return
        except ImportError:
            pass

        import types
        if "selective_scan_cuda" not in sys.modules:
            stub = types.ModuleType("selective_scan_cuda")
            def _missing(*a, **kw):
                raise NotImplementedError(
                    "selective_scan_cuda kernel missing; the call should have "
                    "been routed to selective_scan_ref via the runtime patch."
                )
            stub.fwd = _missing
            stub.bwd = _missing
            sys.modules["selective_scan_cuda"] = stub

        from mamba_ssm.ops import selective_scan_interface as ssi
        from mamba_ssm.modules import mamba_simple
        if getattr(ssi, "_dramabox_kernel_free_patch_applied", False):
            return
        ssi.selective_scan_fn = ssi.selective_scan_ref
        ssi.mamba_inner_fn = ssi.mamba_inner_ref


        mamba_simple.selective_scan_fn = ssi.selective_scan_ref
        mamba_simple.mamba_inner_fn = ssi.mamba_inner_ref
        ssi._dramabox_kernel_free_patch_applied = True
        logging.info(
            "mamba_ssm kernel missing - using kernel-free fallback "
            "(selective_scan_fn -> selective_scan_ref). Expect ~5-10x slowdown."
        )

    def _lazy_load(self) -> None:
        if self._model is not None:
            return


        self._ensure_mamba_ssm_importable()



        reuse_dir = _resolve_reuse_dir()
        if str(reuse_dir) not in sys.path:
            sys.path.insert(0, str(reuse_dir))

        if self.config_path is None:
            self.config_path = self._config_path_override or (
                reuse_dir / "recipes" /
                "USEMamba_30x1_lr_00002_norm_05_vq_065_nfft_320_hop_40_NRIR_012_pha_0005_com_04_early_001.yaml"
            )

        from models.generator_SEMamba_time_d4 import SEMamba
        from models.stfts import mag_phase_stft, mag_phase_istft
        from utils.util import load_config, pad_or_trim_to_match

        self._cfg = load_config(str(self.config_path))
        compress_factor = self._cfg["model_cfg"]["compress_factor"]
        self._stft_fns = (mag_phase_stft, mag_phase_istft, compress_factor, pad_or_trim_to_match)


        model = SEMamba.from_pretrained("nvidia/RE-USE", cfg=self._cfg).to(self.device)
        model.train(False)
        self._model = model
        n_params = sum(p.numel() for p in model.parameters())
        logging.info(f"RE-USE loaded: SEMamba ({n_params / 1e6:.1f}M params) -> {self.target_sr} Hz")

    @staticmethod
    def _make_even(v: float) -> int:
        v = int(round(v))
        return v if v % 2 == 0 else v + 1

    @torch.inference_mode()
    def __call__(self, waveform: torch.Tensor, in_sr: int = 16000) -> Tuple[torch.Tensor, int]:
        import math
        self._lazy_load()
        import librosa
        mag_phase_stft, mag_phase_istft, compress_factor, pad_or_trim_to_match = self._stft_fns


        base_n_fft = self._cfg["stft_cfg"]["n_fft"]
        base_hop = self._cfg["stft_cfg"]["hop_size"]
        base_win = self._cfg["stft_cfg"]["win_size"]
        base_sr = self._cfg["stft_cfg"]["sampling_rate"]

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)


        if self.target_sr != in_sr:
            wav_np = waveform.cpu().float().numpy()
            wav_np = librosa.resample(
                wav_np, orig_sr=in_sr, target_sr=self.target_sr, res_type="kaiser_best"
            )
            wav = torch.from_numpy(wav_np).to(self.device, dtype=torch.float32)
        else:
            wav = waveform.to(self.device, dtype=torch.float32)

        op_sr = self.target_sr
        n_fft = self._make_even(base_n_fft * op_sr // base_sr)
        hop = self._make_even(base_hop * op_sr // base_sr)
        win = self._make_even(base_win * op_sr // base_sr)


        chunk_size = int(self.chunk_size_s * op_sr)
        hop_length = int(self.hop_portion * chunk_size)
        window = torch.hann_window(chunk_size, device=self.device)

        n_ch, total = wav.shape
        enhanced = torch.zeros_like(wav)
        window_sum = torch.zeros_like(wav)
        n_chunks = max(1, math.ceil((total - chunk_size) / hop_length) + 1) if total > chunk_size else 1

        for c in range(n_ch):
            ch_in = wav[c : c + 1]
            for i in range(n_chunks):
                start = i * hop_length
                end = min(start + chunk_size, total)
                chunk = ch_in[:, start:end]
                if chunk.shape[-1] < 2:
                    continue
                noisy_mag, noisy_pha, _ = mag_phase_stft(
                    chunk, n_fft=n_fft, hop_size=hop, win_size=win,
                    compress_factor=compress_factor, center=True, addeps=False,
                )
                amp_g, pha_g, _ = self._model(noisy_mag, noisy_pha)

                mag = torch.expm1(torch.relu(amp_g))
                zero_portion = (mag == 0).sum(dim=1) / mag.shape[1]
                amp_g[:, :, (zero_portion > 0.5)[0]] = 0

                audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop, win, compress_factor)
                audio_g = pad_or_trim_to_match(chunk.detach(), audio_g, pad_value=1e-8)

                w_slice = window[: audio_g.shape[-1]]
                enhanced[c : c + 1, start : start + audio_g.shape[-1]] += audio_g * w_slice
                window_sum[c : c + 1, start : start + audio_g.shape[-1]] += w_slice


        mask = window_sum > 1e-8
        enhanced[mask] = enhanced[mask] / window_sum[mask]
        return enhanced.clamp(-1.0, 1.0).cpu().float(), op_sr
