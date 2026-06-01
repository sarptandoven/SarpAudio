
import torch

from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import AudioLatentTools
from ltx_core.types import AudioLatentShape, LatentState


class AudioConditionByReferenceLatent(ConditioningItem):

    def __init__(self, latent: torch.Tensor, strength: float = 1.0):
        self.latent = latent
        self.strength = strength

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: AudioLatentTools,
    ) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)





        ref_shape = AudioLatentShape(
            batch=self.latent.shape[0],
            channels=self.latent.shape[1],
            frames=self.latent.shape[2],
            mel_bins=self.latent.shape[3],
        )
        positions = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=ref_shape,
            device=self.latent.device,
        )

        positions = positions + 0.5


        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=torch.float32,
        )















        batch_size = tokens.shape[0]
        num_target = latent_state.latent.shape[1]
        num_ref = tokens.shape[1]
        total = num_target + num_ref



        mask = torch.zeros(
            (batch_size, total, total),
            device=self.latent.device,
            dtype=torch.float32,
        )


        if latent_state.attention_mask is not None:
            mask[:, :num_target, :num_target] = latent_state.attention_mask
        else:
            mask[:, :num_target, :num_target] = 1.0


        mask[:, :num_target, num_target:] = 1.0





        mask[:, num_target:, num_target:] = 1.0

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            attention_mask=mask,
        )
