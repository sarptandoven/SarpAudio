
from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.modality import Modality


def _split_perturbations(config: BatchedPerturbationConfig, sizes: list[int]) -> list[BatchedPerturbationConfig]:
    it = iter(config.perturbations)
    return [BatchedPerturbationConfig([next(it) for _ in range(s)]) for s in sizes]


def _merge_tensors(tensors: list[torch.Tensor | None]) -> torch.Tensor | None:
    non_none = [t for t in tensors if t is not None]
    if not non_none:
        return None
    return torch.cat(non_none, dim=0)


class BatchSplitAdapter(nn.Module):

    def __init__(self, model: nn.Module, max_batch_size: int) -> None:
        if max_batch_size < 1:
            raise ValueError(f"max_batch_size must be >= 1, got {max_batch_size}")
        super().__init__()
        self._model = model
        self._max_batch_size = max_batch_size

    def _get_chunk_sizes(self, batch_size: int) -> list[int]:
        full, remainder = divmod(batch_size, self._max_batch_size)
        sizes = [self._max_batch_size] * full
        if remainder:
            sizes.append(remainder)
        return sizes

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        batch_size = (video or audio).latent.shape[0]

        if batch_size <= self._max_batch_size:
            return self._model(video=video, audio=audio, perturbations=perturbations)

        sizes = self._get_chunk_sizes(batch_size)
        n = len(sizes)

        v_chunks = video.split(sizes) if video is not None else [None] * n
        a_chunks = audio.split(sizes) if audio is not None else [None] * n
        p_chunks = _split_perturbations(perturbations, sizes)

        chunk_results = [
            self._model(video=vc, audio=ac, perturbations=pc)
            for vc, ac, pc in zip(v_chunks, a_chunks, p_chunks, strict=True)
        ]

        results_v, results_a = zip(*chunk_results, strict=True)
        return _merge_tensors(list(results_v)), _merge_tensors(list(results_a))

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._model, name)
