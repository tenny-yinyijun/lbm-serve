"""Diffusion noise schedulers.

TRIMMED FOR SERVING (lbm-serve). Upstream this module also exports `StableDiffusion`,
`UNet`, `UNetDiffusers` and their blocks, and registers the "stable_diffusion" model. None
is reachable from `diffusion_policy`, which only calls `create_noise_scheduler`.

`diffusers` is NOT removable here: `FlowMatchingScheduler` lives in
`noise_scheduler_diffusers`, which imports `DDPMScheduler` at module scope, and the
open-world checkpoints set `use_flow_matching_scheduler: true`.
"""

from vla_foundry.models.diffusion.noise_scheduler import NoiseSchedulerDDPM
from vla_foundry.models.diffusion.noise_scheduler_diffusers import (
    FlowMatchingScheduler,
    NoiseSchedulerDDPMDiffusers,
)


def create_noise_scheduler(model_params):
    """Create noise scheduler based on model parameters."""
    if model_params.use_diffusers_scheduler and model_params.use_flow_matching_scheduler:
        raise ValueError("use_flow_matching_scheduler and use_diffusers_scheduler are mutually exclusive")
    if model_params.use_diffusers_scheduler:
        noise_scheduler = NoiseSchedulerDDPMDiffusers(model_params.noise_scheduler)
    elif model_params.use_flow_matching_scheduler:
        noise_scheduler = FlowMatchingScheduler(model_params.noise_scheduler)
    else:
        noise_scheduler = NoiseSchedulerDDPM(model_params.noise_scheduler)
    return noise_scheduler


__all__ = [
    "NoiseSchedulerDDPM",
    "NoiseSchedulerDDPMDiffusers",
    "FlowMatchingScheduler",
    "create_noise_scheduler",
]
