from dataclasses import dataclass

from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.conditioning.item import ConditioningItem
from ltx_core.tools import LatentTools, SpatioTemporalScaleFactors
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape


@dataclass(frozen=True)
class TemporalRegionMask(ConditioningItem):

    start_time: float
    end_time: float
    fps: float

    def apply_to(self, latent_state: LatentState, latent_tools: LatentTools) -> LatentState:
        coords = latent_tools.patchifier.get_patch_grid_bounds(
            latent_tools.target_shape, device=latent_state.denoise_mask.device
        )
        if isinstance(latent_tools.target_shape, AudioLatentShape):

            t_boundaries = coords[:, 0]
        elif isinstance(latent_tools.target_shape, VideoLatentShape):

            scale_factors = getattr(latent_tools, "scale_factors", SpatioTemporalScaleFactors.default())
            pixel_bounds = get_pixel_coords(coords, scale_factors, causal_fix=getattr(latent_tools, "causal_fix", True))

            t_boundaries = pixel_bounds[:, 0] / self.fps
        else:
            raise ValueError("Unsupported LatentShape type, expected AudioLatentShape or VideoLatentShape")
        t_start, t_end = t_boundaries.unbind(dim=-1)
        in_region = (t_end > self.start_time) & (t_start < self.end_time)
        state = latent_state.clone()
        mask_val = in_region.to(state.denoise_mask.dtype)
        if state.denoise_mask.dim() == 3:
            mask_val = mask_val.unsqueeze(-1)
        state.denoise_mask.copy_(mask_val)
        return state
