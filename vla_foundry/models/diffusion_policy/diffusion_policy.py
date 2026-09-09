import torch

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.diffusion.noise_scheduler import NoiseScheduler
from vla_foundry.models.diffusion.noise_scheduler_diffusers import FlowMatchingScheduler
from vla_foundry.models.diffusion.unet import SinusoidalPositionEmbeddings
from vla_foundry.models.diffusion_policy.time_embedding import TimeEmbedding
from vla_foundry.models.transformer import Transformer
from vla_foundry.models.transformer_hf import TransformerHF
from vla_foundry.models.vision_language_backbones import BaseBackboneWrapper
from vla_foundry.params.model_params import DiffusionPolicyParams


class DiffusionPolicy(BaseModel):
    def __init__(
        self,
        model_params: DiffusionPolicyParams,
        vision_language_backbone: BaseBackboneWrapper,
        transformer: Transformer | TransformerHF,
        noise_scheduler: NoiseScheduler,
    ):
        super().__init__(model_params)
        self.vision_language_backbone = vision_language_backbone
        self.transformer = transformer
        self.scheduler = noise_scheduler
        self.use_flow_matching_scheduler = model_params.use_flow_matching_scheduler
        if self.use_flow_matching_scheduler and not isinstance(self.scheduler, FlowMatchingScheduler):
            raise ValueError("use_flow_matching_scheduler=True requires a FlowMatchingScheduler")
        self.continuous_time = model_params.noise_scheduler.continuous_time
        if self.continuous_time and not self.use_flow_matching_scheduler:
            raise ValueError("noise_scheduler.continuous_time=True requires use_flow_matching_scheduler=True")
        self.proprioception_dim = model_params.proprioception_dim

        backbone_dim = vision_language_backbone.get_conditioning_embeddings_dim()
        if self.continuous_time:
            # Continuous flow matching uses scalar times in [0, 1] rather than a fixed table
            # of discrete DDPM timestep embeddings.
            self.time_embedding = TimeEmbedding(backbone_dim)
            self.time_encoding = None
            self.sinusoidal_position_embeddings = None
        else:
            self.time_embedding = None
            self.time_encoding = torch.nn.Embedding(noise_scheduler.num_timesteps, backbone_dim)
            self.sinusoidal_position_embeddings = SinusoidalPositionEmbeddings(backbone_dim)
        self.output_layer = torch.nn.Linear(transformer.hidden_dim, model_params.action_dim)
        self.action_encode = torch.nn.Linear(model_params.action_dim, transformer.hidden_dim)
        self.condition_encode = torch.nn.Linear(backbone_dim, transformer.hidden_dim)
        self.proprioception_encode = (
            torch.nn.Linear(self.proprioception_dim, transformer.hidden_dim) if self.proprioception_dim > 0 else None
        )

        self.diffusion_step_conditioning = model_params.diffusion_step_conditioning
        self.input_noise_std = model_params.input_noise_std
        self.num_action_head_repeats = model_params.num_action_head_repeats
        self.mask_padded_actions = model_params.mask_padded_actions
        self.initialize_weights()

    def initialize_weights(self):
        if self.time_encoding is not None:
            # Initialize time encoding weights with sinusoidal position embeddings.
            timesteps = torch.arange(self.time_encoding.weight.shape[0])
            with torch.no_grad():
                self.time_encoding.weight.copy_(self.sinusoidal_position_embeddings.forward(timesteps))

        # Initialize output layer weights with Xavier initialization
        torch.nn.init.xavier_uniform_(self.output_layer.weight)
        if self.proprioception_encode is not None:
            torch.nn.init.xavier_uniform_(self.proprioception_encode.weight)

    def _build_conditioning_mask(self, backbone_embeddings, backbone_attention_mask=None):
        """Build the attention mask for the conditioning part of the transformer input.

        Must match the sequence structure produced by _build_transformer_input:
        - ADD: [backbone] -> backbone_mask
        - CONCAT: [time, backbone] -> [time_mask, backbone_mask]

        Returns:
            List of mask tensors to be extended with proprio/action masks and concatenated.
        """
        batch_size = backbone_embeddings.shape[0]
        device = backbone_embeddings.device

        if backbone_attention_mask is not None:
            backbone_mask = backbone_attention_mask.to(device=device, dtype=torch.bool)
        else:
            backbone_mask = torch.ones(batch_size, backbone_embeddings.shape[1], dtype=torch.bool, device=device)

        mask_parts = []
        if self.diffusion_step_conditioning == "concat":
            # Time token is prepended as a separate token and is always visible.
            mask_parts.append(torch.ones(batch_size, 1, dtype=torch.bool, device=device))
        mask_parts.append(backbone_mask)
        return mask_parts

    @staticmethod
    def _leading_padding_from_past_mask(past_mask):
        """Locate padded action slots when only past_mask is known (inference).

        The future region at inference is the chunk being denoised, so it is never
        padded; padding is exactly the leading run of past slots that the episode has
        not produced yet, which PolicyDataAdapter fills in from the right. A row whose
        past_mask is empty has no past region at all (num_past == 0) and therefore no
        padding -- without that guard the cumulative product would mask everything.

        Returns:
            Bool tensor [B, T], True where the slot is padding.
        """
        head_padding = torch.cumprod((~past_mask).to(torch.uint8), dim=1).bool()
        return head_padding & past_mask.any(dim=1, keepdim=True)

    def _build_action_mask(self, batch_size, action_seq_len, device, is_padding=None):
        """Attention mask for the action block: True where a token may be attended to.

        Padded slots are hidden only when mask_padded_actions is set, and only from the
        attention keys. What those slots contain is decided separately in forward().
        """
        if self.mask_padded_actions and is_padding is not None:
            return ~is_padding.to(device=device, dtype=torch.bool)
        return torch.ones(batch_size, action_seq_len, dtype=torch.bool, device=device)

    def _get_time_embeddings(self, timesteps):
        if self.continuous_time:
            return self.time_embedding(timesteps).unsqueeze(1)
        return self.time_encoding(timesteps).unsqueeze(1)

    def _build_transformer_input(self, backbone_embeddings, time_embeddings, noisy_action, proprio_embeddings=None):
        """Build transformer input by combining conditioning, time, and action embeddings.

        Supports two time conditioning strategies:
        - CONCAT: Prepend time as a separate token [time, backbone] → [B, 1+N, D]
        - ADD: Add time to backbone embeddings element-wise → [B, N, D]

        Args:
            backbone_embeddings: [B, N, backbone_dim] from vision-language backbone
            time_embeddings: [B, 1, backbone_dim] from time encoding
            noisy_action: [B, T, transformer_dim] encoded noisy actions
            proprio_embeddings: Optional [B, P, transformer_dim] encoded proprioception

        Returns:
            transformer_input: [B, C+P+T, transformer_dim]
        """
        if self.diffusion_step_conditioning == "add":
            conditional_embeddings = backbone_embeddings + time_embeddings
        elif self.diffusion_step_conditioning == "concat":
            conditional_embeddings = torch.cat([time_embeddings, backbone_embeddings], dim=1)
        else:
            raise ValueError(f"Unknown diffusion_step_conditioning: {self.diffusion_step_conditioning}")

        conditional_embeddings = self.condition_encode(conditional_embeddings)

        parts = [conditional_embeddings]
        if proprio_embeddings is not None:
            parts.append(proprio_embeddings)
        parts.append(noisy_action)

        return torch.cat(parts, dim=1)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask,
        attention_mask_images,
        actions,
        noise,
        past_mask,
        future_mask,
        proprioception=None,
        **kwargs,
    ):
        # Sample random timesteps
        if self.continuous_time:
            # Continuous flow matching is trained with timesteps in [0, 1).
            timesteps = torch.rand((actions.shape[0],), device=actions.device)
        else:
            timesteps = torch.randint(0, self.scheduler.num_timesteps, (actions.shape[0],), device=actions.device)

        # Clean iff in the past region = real past + the LEADING padding run, matching
        # inference (past slots clean, everything else noise). Tail padding is edge-copied,
        # so clean there would leak the episode goal into this bidirectional block.
        is_padding = ~past_mask & ~future_mask
        head_padding = torch.cumprod(is_padding.to(torch.uint8), dim=1).bool()
        clean_mask = past_mask | head_padding
        noisy_action = self.scheduler.add_noise(actions, noise, timesteps, mask=~clean_mask)
        noisy_action = torch.where(clean_mask.unsqueeze(-1), actions, noisy_action)
        if self.input_noise_std > 0:
            # Add input noise to the past (conditioning) actions only - they are
            # unsupervised conditioning, so this is input augmentation that
            # helps cross-chunk continuity. Not the future/current positions:
            # those are the diffusion denoising target, and the regression target
            # (noise - actions) does not account for extra input noise, so
            # noising them would corrupt the supervision.
            #
            # Not padding either: it already carries scheduler noise at the sampled
            # timestep, and a fixed-std jitter on top matches nothing at inference.
            noise_term = torch.randn_like(noisy_action) * self.input_noise_std
            noisy_action = torch.where(past_mask.unsqueeze(-1), noisy_action + noise_term, noisy_action)
        noisy_action = self.action_encode(noisy_action)

        # Get backbone embeddings (handles text+image concatenation)
        backbone_output = self.vision_language_backbone.get_action_conditioning(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )

        backbone_embeddings = backbone_output.embeddings
        backbone_attention_mask = backbone_output.attention_mask
        num_repeats = self.num_action_head_repeats
        if num_repeats is not None and num_repeats > 1:
            # Verify action-side inputs were tiled to [B*N] by the batch handler
            vlm_batch_size = input_ids.shape[0]
            assert actions.shape[0] == vlm_batch_size * num_repeats, (
                f"Expected actions batch size {vlm_batch_size * num_repeats} (vlm_batch={vlm_batch_size} * "
                f"num_repeats={num_repeats}), got {actions.shape[0]}"
            )
            assert noise.shape[0] == vlm_batch_size * num_repeats, (
                f"Expected noise batch size {vlm_batch_size * num_repeats}, got {noise.shape[0]}"
            )
            assert future_mask.shape[0] == vlm_batch_size * num_repeats, (
                f"Expected future_mask batch size {vlm_batch_size * num_repeats}, got {future_mask.shape[0]}"
            )
            if proprioception is not None:
                assert proprioception.shape[0] == vlm_batch_size * num_repeats, (
                    f"Expected proprioception batch size {vlm_batch_size * num_repeats}, got {proprioception.shape[0]}"
                )
            # Tile backbone-side conditioning to match the action batch size [B*N].
            backbone_embeddings = backbone_embeddings.repeat_interleave(num_repeats, dim=0)
            if backbone_attention_mask is not None:
                backbone_attention_mask = backbone_attention_mask.repeat_interleave(num_repeats, dim=0)

        # Time embeddings (batch, 1, backbone_dim) - batch is [B*N] when repeating, else [B]
        time_embeddings = self._get_time_embeddings(timesteps)

        # Proprioception embeddings (already tiled to [B*N] by the batch handler when num_repeats > 1)
        proprio_embeddings = None
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_embeddings = self.proprioception_encode(proprioception)
            # input_noise_std also regularizes the proprioception input (in
            # addition to the past actions above).
            if self.input_noise_std > 0:
                proprio_embeddings = proprio_embeddings + torch.randn_like(proprio_embeddings) * self.input_noise_std

        # Build transformer input using time conditioning strategy
        transformer_input = self._build_transformer_input(
            backbone_embeddings=backbone_embeddings,
            time_embeddings=time_embeddings,
            noisy_action=noisy_action,
            proprio_embeddings=proprio_embeddings,
        )

        # Build attention mask matching transformer input: [conditioning, proprio, actions]
        full_attention_mask = self._build_conditioning_mask(
            backbone_embeddings, backbone_attention_mask=backbone_attention_mask
        )

        # Add proprioception mask if present
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_len = proprioception.shape[1]
            proprio_mask = torch.ones(
                backbone_embeddings.shape[0], proprio_len, dtype=torch.bool, device=backbone_embeddings.device
            )
            full_attention_mask.append(proprio_mask)

        # Action padding is value/loss-side metadata unless mask_padded_actions is set,
        # in which case padded slots are also hidden from the attention keys. Proprioception
        # is never padded, so its mask stays all-ones either way.
        full_attention_mask.append(
            self._build_action_mask(
                noisy_action.shape[0], noisy_action.shape[1], noisy_action.device, is_padding=is_padding
            )
        )
        full_attention_mask = torch.cat(full_attention_mask, dim=1)
        if full_attention_mask.shape != transformer_input.shape[:2]:
            raise ValueError(
                f"Transformer attention mask shape {full_attention_mask.shape} must match "
                f"input token shape {transformer_input.shape[:2]}."
            )

        # Pass through transformer
        transformer_output = self.transformer(
            inputs_embeds=transformer_input,
            output_hidden_states=True,
            use_cache=False,
            attention_mask=full_attention_mask,
        )

        # Extract predicted direction to denoise the action (B, 1+N+P+T, D) -> (B, T, D)
        action_seq_len = noise.shape[1]
        predicted_direction = self.output_layer(transformer_output.hidden_states[-1][:, -action_seq_len:, :])

        return predicted_direction

    @torch.no_grad()
    def generate_actions(
        self,
        input_ids,
        pixel_values,
        actions,
        attention_mask=None,
        attention_mask_images=None,
        num_inference_steps: int | None = None,
        past_mask=None,
        proprioception=None,
        guidance_target: torch.Tensor | None = None,
        guidance_scale: float = 0.0,
        guidance_mask: torch.Tensor | None = None,
        sigma_d_obs: float = 0.2,
        **kwargs,  # Ignore extra params like point_cloud (used by other models)
    ):
        """
        Generate actions using iterative denoising through the diffusion process.

        Args:
            input_ids: Text input token IDs
            pixel_values: Input images/pixel values
            actions: Input actions (past timesteps are given in the same sequence, others can be noise)
            attention_mask: Optional attention mask for text
            attention_mask_images: Optional attention mask for camera images
            num_inference_steps: Number of denoising steps. Required for flow matching; defaults to
                scheduler.num_timesteps for DDPM.
            past_mask: Optional mask indicating which actions are from past (1) vs future (0)
            proprioception: Optional proprioception input
            guidance_target: Target actions for Pi-GDM guidance (previous chunk predictions)
            guidance_scale: Multiplier for β = guidance_scale * n (1.0 = recommended default)
            guidance_mask: Per-timestep mask weighting the guidance residual
            sigma_d_obs: Observation noise std for Pi-GDM (default 0.2)
            **kwargs: Model-specific args

        Returns:
            Generated actions
        """
        if num_inference_steps is None and self.continuous_time:
            raise ValueError("num_inference_steps is required when noise_scheduler.continuous_time=True")
        if num_inference_steps is None:
            # DDPM / flow matching with discrete time_step can fall back to the full training schedule.
            num_inference_steps = self.scheduler.num_timesteps
        num_inference_steps = int(num_inference_steps)
        if num_inference_steps <= 0:
            raise ValueError(f"num_inference_steps must be > 0, got {num_inference_steps}")

        use_guidance = guidance_scale > 0 and guidance_target is not None
        # β = guidance_scale * n (num_inference_steps) as in the RTC paper.
        # This gives critically-damped guidance: β·dt = guidance_scale ≈ 1.
        beta = guidance_scale * num_inference_steps

        batch_size = actions.shape[0]
        device = actions.device

        # Action-block attention mask, constant across denoising steps.
        is_padding = (
            self._leading_padding_from_past_mask(past_mask.to(device=device, dtype=torch.bool))
            if self.mask_padded_actions and past_mask is not None
            else None
        )
        action_mask = self._build_action_mask(batch_size, actions.shape[1], device, is_padding=is_padding)

        # Precompute backbone embeddings (reused across all denoising steps)
        backbone_output = self.vision_language_backbone.get_action_conditioning(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            attention_mask_images=attention_mask_images,
            **kwargs,
        )

        # Initialize actions with noise (preserve past actions if past_mask provided)
        if past_mask is not None:
            original_past_actions = actions.clone() * past_mask[:, :, None].float()
            # Keep past actions, add noise to future actions.
            actions = actions * past_mask[:, :, None].float() + torch.randn_like(actions) * (
                1 - past_mask[:, :, None].float()
            )
        else:
            # All actions are noise.
            actions = torch.randn_like(actions)

        # Precompute proprioception embedding
        proprio_embeddings = None
        if self.proprioception_encode is not None and proprioception is not None:
            proprio_embeddings = self.proprioception_encode(proprioception)

        if self.continuous_time:
            # Iterative denoising - similar to flow VLM approach
            ## List of timesteps to denoise the actions, from 1 to 0. (Does not need to be equaly spaced.)
            step_list = torch.linspace(0, 1, num_inference_steps + 1, device=device).flip(dims=[0])[:-1]
        else:
            # Discrete denoising loop over integer timesteps (DDPM and discrete-time flow matching)
            step_size = max(1, self.scheduler.num_timesteps // num_inference_steps)
            step_list = range(self.scheduler.num_timesteps - 1, 0, -step_size)

        for i, step in enumerate(step_list):
            if self.continuous_time:
                timesteps = step.expand(batch_size)
            else:
                timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
            time_embeddings = self._get_time_embeddings(timesteps)

            # Encode current actions
            action_encoding = self.action_encode(actions)

            # Build transformer input using time conditioning strategy
            transformer_input = self._build_transformer_input(
                backbone_embeddings=backbone_output.embeddings,
                time_embeddings=time_embeddings,
                noisy_action=action_encoding,
                proprio_embeddings=proprio_embeddings,
            )

            # Build attention mask matching transformer input: [conditioning, proprio, actions]
            full_attention_mask = self._build_conditioning_mask(
                backbone_output.embeddings, backbone_attention_mask=backbone_output.attention_mask
            )

            # Add proprioception mask if present
            if proprio_embeddings is not None:
                proprio_len = proprio_embeddings.shape[1]
                proprio_mask = torch.ones(batch_size, proprio_len, dtype=torch.bool, device=device)
                full_attention_mask.append(proprio_mask)

            full_attention_mask.append(action_mask)

            # Concatenate all masks at once
            full_attention_mask = torch.cat(full_attention_mask, dim=1)
            if full_attention_mask.shape != transformer_input.shape[:2]:
                raise ValueError(
                    f"Transformer attention mask shape {full_attention_mask.shape} must match "
                    f"input token shape {transformer_input.shape[:2]}."
                )

            # Pass through transformer
            transformer_output = self.transformer(
                inputs_embeds=transformer_input,
                output_hidden_states=True,
                use_cache=False,
                attention_mask=full_attention_mask,
            )

            # Extract predicted direction to denoise the action
            action_seq_len = actions.shape[1]
            predicted_direction = self.output_layer(transformer_output.hidden_states[-1][:, -action_seq_len:, :])

            # Pi-GDM guidance (replace approximation, J ≈ I).
            # Weight = min(β, raw_weight) with β = guidance_scale (fixed).
            # Fixed β gives step-count invariance (see generate_actions docstring).
            if use_guidance:
                tau = step if self.continuous_time else step / self.scheduler.num_timesteps
                x0_hat = actions - tau * predicted_direction
                residual = guidance_target - x0_hat
                if guidance_mask is not None:
                    if guidance_mask.dim() == 1:
                        guidance_mask = guidance_mask.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
                    elif guidance_mask.dim() == 2:
                        guidance_mask = guidance_mask.unsqueeze(0)  # [1, T, D]
                    residual = residual * guidance_mask
                sigma_sq = sigma_d_obs**2
                denom = (1 - tau) * tau * sigma_sq + 1e-8
                raw_weight = (tau**2 + sigma_sq * (1 - tau) ** 2) / denom
                adaptive_scale = min(beta, raw_weight)
                predicted_direction = predicted_direction - adaptive_scale * residual

            # Denoise actions using scheduler step
            #   continuous time_size -> dt = gap to the next grid point
            #   discrete time step   -> integer stride (normalized by num_timesteps inside the scheduler)
            #   DDPM                 -> step_size is ignored (it uses its own variance schedule)
            if self.continuous_time:
                step_size = step_list[i] - step_list[i + 1] if i < len(step_list) - 1 else step_list[i]
            predicted_actions = self.scheduler.step(predicted_direction, step, actions, step_size=step_size)

            # Preserve past actions if mask provided
            if past_mask is not None:
                past_mask_expanded = past_mask[:, :, None].to(actions.dtype)
                actions = original_past_actions + predicted_actions * (1 - past_mask_expanded)
            else:
                actions = predicted_actions

        return actions
