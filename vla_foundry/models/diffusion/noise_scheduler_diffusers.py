from diffusers import DDPMScheduler

from vla_foundry.models.diffusion.noise_scheduler import NoiseScheduler
from vla_foundry.params.model_params import NoiseSchedulerParams


class NoiseSchedulerDDPMDiffusers(NoiseScheduler):
    def __init__(self, params: NoiseSchedulerParams):
        super().__init__(params)
        if params.num_timesteps is None:
            raise ValueError("NoiseSchedulerDDPMDiffusers requires noise_scheduler.num_timesteps")
        self.num_timesteps = params.num_timesteps
        if params.clamp_range is not None:
            assert -params.clamp_range[0] == params.clamp_range[1], (
                "clamp_range must be symmetric for DDPM Diffusers scheduler"
            )
            clip_params = {
                "clip_sample": True,
                "clip_sample_range": params.clamp_range[1],
            }
        else:
            clip_params = {
                "clip_sample": False,
                "clip_sample_range": None,
            }
        self.clamp_range = params.clamp_range
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.num_timesteps,
            **clip_params,
        )

    def add_noise(self, x_start, noise, timesteps, mask=None):
        output = self.scheduler.add_noise(x_start, noise, timesteps)
        if mask is not None:
            mask_expanded = mask
            while mask_expanded.ndim < x_start.ndim:
                mask_expanded = mask_expanded.unsqueeze(-1)
            mask_expanded = mask_expanded.to(dtype=x_start.dtype)
            output = mask_expanded * output + (1 - mask_expanded) * x_start
        if self.clamp_range is not None:
            output = output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output

    def step(self, model_output, timestep, sample, step_size=None):
        # step_size is unused (DDPM uses the diffusers variance schedule)
        output = self.scheduler.step(model_output, timestep, sample).prev_sample
        if self.clamp_range is not None:
            output = output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output


class FlowMatchingScheduler(NoiseScheduler):
    def __init__(self, params: NoiseSchedulerParams):
        super().__init__(params)
        self.clamp_range = params.clamp_range
        self.continuous_time = params.continuous_time
        self.num_timesteps = params.num_timesteps

    def add_noise(self, x_start, noise, timesteps, mask=None):
        """
        Add noise to the input data. If mask is provided, only add noise to the unmasked parts.
        Args:
            x_start: The input data.
            noise: The noise to add.
            timesteps: The timesteps used to scale the noise.
            mask: The mask 1: should add noise, 0: should not add noise.
        Returns:
            The noisy data.
        """
        # Determine how many dimensions to add by comparing timesteps and x_start shapes
        # timesteps is typically (batch_size,) and x_start is (batch_size, ..., feature)
        # We need to add dimensions to timesteps to match x_start's broadcasting requirements
        target_ndim = x_start.ndim
        current_ndim = timesteps.ndim
        dims_to_add = target_ndim - current_ndim

        # Create the view shape: keep first dimension (-1), add 1s for the remaining dimensions
        view_shape = [-1] + [1] * dims_to_add
        # Compute a scale in the same dtype as inputs to avoid dtype promotion (e.g., bf16 -> fp32)
        scale = timesteps.view(*view_shape).to(dtype=x_start.dtype)
        if not self.continuous_time:
            scale = scale / self.num_timesteps
        if mask is not None:
            # Ensure mask can broadcast with x_start by expanding missing dimensions
            # mask should have first dimensions matching x_start, and we'll expand the rest
            mask_expanded = mask
            while mask_expanded.ndim < x_start.ndim:
                mask_expanded = mask_expanded.unsqueeze(-1)

            output = x_start + scale * (noise - x_start) * mask_expanded
        else:
            output = x_start + scale * (noise - x_start)
        if self.clamp_range is not None:
            output = output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output

    def step(self, model_output, timestep=None, sample=None, step_size=None):
        # The absolute timestep is not used: the increment is fully described by step_size.
        # Continuous time: step_size is the explicit dt from the inference time range.
        # Discrete time: step_size is an integer stride normalized by num_timesteps
        # (step_size defaults to 1 -> a single 1/num_timesteps increment).
        if sample is None:
            raise ValueError("FlowMatchingScheduler.step requires sample to be provided.")
        if self.continuous_time:
            if step_size is None:
                raise ValueError("FlowMatchingScheduler.step requires step_size when continuous_time=True.")
            time_increment = step_size
        else:
            time_increment = (1 if step_size is None else step_size) / self.num_timesteps
        output = sample - model_output * time_increment
        if self.clamp_range is not None:
            output = output.clamp(self.clamp_range[0], self.clamp_range[1])
        return output
