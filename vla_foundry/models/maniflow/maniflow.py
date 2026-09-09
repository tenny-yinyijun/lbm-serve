# MIT License
#
# Copyright (c) 2025 Ge Yan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""ManiFlow model implementation for consistency flow training."""

import logging

import torch
import torch.nn.functional as F
from einops import reduce

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.maniflow.ditx import DiTX
from vla_foundry.models.maniflow.pointnet_encoder import DP3Encoder
from vla_foundry.models.maniflow.sample_util import (
    sample_beta,
    sample_cosmap,
    sample_logit_normal,
    sample_mode,
)
from vla_foundry.params.model_params import ManiFlowParams

logger = logging.getLogger(__name__)


class ManiFlow(BaseModel):
    """ManiFlow: Manifold Flow model with consistency training for robotic manipulation.

    This model combines flow matching with consistency training to learn action policies
    conditioned on point cloud observations. It uses a DiTX transformer backbone with
    DP3Encoder for point cloud processing.
    """

    def __init__(
        self,
        model_params: ManiFlowParams,
        encoder: DP3Encoder,
        action_predictor: DiTX,
    ):
        super().__init__(model_params)

        self.action_dim = model_params.action_dim
        self.horizon = model_params.horizon
        self.lowdim_past_timesteps = model_params.lowdim_past_timesteps
        self.lowdim_future_timesteps = model_params.lowdim_future_timesteps
        self.language_conditioned = model_params.action_predictor.language_conditioned

        # Flow and consistency training parameters
        self.num_inference_steps = model_params.num_inference_steps
        self.flow_batch_ratio = model_params.flow_batch_ratio
        self.consistency_batch_ratio = model_params.consistency_batch_ratio
        assert self.flow_batch_ratio + self.consistency_batch_ratio == 1.0, "Sum of batch ratios should equal 1.0"
        self.denoise_timesteps = model_params.denoise_timesteps
        self.sample_t_mode_flow = model_params.sample_t_mode_flow
        self.sample_t_mode_consistency = model_params.sample_t_mode_consistency
        self.sample_dt_mode_consistency = model_params.sample_dt_mode_consistency
        self.sample_target_t_mode = model_params.sample_target_t_mode
        assert self.sample_target_t_mode in ["absolute", "relative"], (
            "sample_target_t_mode must be either 'absolute' or 'relative'"
        )

        # Use pre-instantiated components
        self.obs_encoder = encoder
        self.action_predictor = action_predictor
        self.obs_feature_dim = encoder.output_shape()

        logger.info("[ManiFlow] Initialized with parameters:")
        logger.info(f"  - horizon: {self.horizon}")
        logger.info(f"  - lowdim_future_timesteps: {self.lowdim_future_timesteps}")
        logger.info(f"  - lowdim_past_timesteps: {self.lowdim_past_timesteps}")
        logger.info(f"  - num_inference_steps: {self.num_inference_steps}")
        logger.info(f"  - flow_batch_ratio: {self.flow_batch_ratio}")
        logger.info(f"  - consistency_batch_ratio: {self.consistency_batch_ratio}")
        logger.info(f"  - denoise_timesteps: {self.denoise_timesteps}")
        logger.info(f"  - sample_t_mode_flow: {self.sample_t_mode_flow}")
        logger.info(f"  - sample_t_mode_consistency: {self.sample_t_mode_consistency}")
        logger.info(f"  - sample_dt_mode_consistency: {self.sample_dt_mode_consistency}")
        logger.info(f"  - sample_target_t_mode: {self.sample_target_t_mode}")

    def sample_t(self, batch_size: int, mode: str = "uniform") -> torch.Tensor:
        """Sample timestep t for flow matching or consistency training.

        Args:
            batch_size: Number of samples
            mode: Sampling mode - "uniform", "lognorm", "mode", "cosmap", "beta", "discrete"

        Returns:
            Tensor of shape (batch_size,) with timesteps in [0, 1]
        """
        device = next(self.parameters()).device

        if mode == "uniform":
            t = torch.rand((batch_size,), device=device)
        elif mode == "lognorm":
            t = sample_logit_normal(batch_size, m=0.0, s=1.0, device=device)
        elif mode == "mode":
            t = sample_mode(batch_size, s=1.29, device=device)
        elif mode == "cosmap":
            t = sample_cosmap(batch_size, device=device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=device)
        elif mode == "discrete":
            t = torch.randint(low=0, high=self.denoise_timesteps, size=(batch_size,), device=device).float()
            t = t / self.denoise_timesteps
        else:
            raise ValueError(
                f"Unsupported sample_t_mode {mode}. Choose from 'uniform', 'lognorm', "
                f"'mode', 'cosmap', 'beta', 'discrete'."
            )
        return t

    def sample_dt(self, batch_size: int, sample_dt_mode: str = "uniform") -> torch.Tensor:
        """Sample delta timestep dt for consistency training.

        Args:
            batch_size: Number of samples
            sample_dt_mode: Sampling mode (currently only "uniform" supported)

        Returns:
            Tensor of shape (batch_size,) with delta timesteps
        """
        device = next(self.parameters()).device

        if sample_dt_mode == "uniform":
            dt = torch.rand((batch_size,), device=device)
        else:
            raise ValueError(f"Unsupported sample_dt_mode {sample_dt_mode}")

        return dt

    def linear_interpolate(
        self,
        noise: torch.Tensor,
        target: torch.Tensor,
        timestep: torch.Tensor,
        epsilon: float = 0.0,
    ) -> torch.Tensor:
        """Linear interpolation between noise and target data.

        Args:
            noise: Initial noise at t=0
            target: Target data point at t=1
            timestep: Interpolation parameter in [0, 1]
            epsilon: Noise preservation factor (default 0.0)

        Returns:
            Interpolated data point at given timestep
        """
        # Calculate noise coefficient with epsilon adjustment
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep

        # Linear combination: preserved_noise + scaled_target
        interpolated_data_point = noise_coeff * noise + timestep * target

        return interpolated_data_point

    def get_flow_velocity(
        self,
        actions: torch.Tensor,
        vis_cond: torch.Tensor,
        lang_cond: list | None = None,
    ) -> dict[str, torch.Tensor]:
        """Get flow velocity targets for training.

        Flow training is used to train the model to predict instantaneous velocity
        given a timestep.

        Args:
            actions: (B, T, action_dim) target actions
            vis_cond: (B, L, obs_feature_dim) visual conditioning
            lang_cond: Optional language conditioning (list of strings)

        Returns:
            Dictionary with keys: x_t, t, target_t, v_target, vis_cond, lang_cond
        """
        target_dict = {}
        flow_batchsize = actions.shape[0]
        device = actions.device

        # Sample t for flow (dt is zero for flow)
        t_flow = self.sample_t(flow_batchsize, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((flow_batchsize,), device=device)

        # Get target timestep
        if self.sample_target_t_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        elif self.sample_target_t_mode == "relative":
            target_t_flow = dt_flow

        # Compute interpolated data points at t and predict flow velocity
        x_0_flow = torch.randn_like(actions, device=device)
        x_1_flow = actions.to(device)
        x_t_flow = self.linear_interpolate(x_0_flow, x_1_flow, t_flow, epsilon=0.0)
        v_t_flow = x_1_flow - x_0_flow

        target_dict["x_t"] = x_t_flow
        target_dict["t"] = t_flow
        target_dict["target_t"] = target_t_flow
        target_dict["v_target"] = v_t_flow
        target_dict["vis_cond"] = vis_cond
        target_dict["lang_cond"] = lang_cond

        return target_dict

    def get_consistency_velocity(
        self,
        actions: torch.Tensor,
        vis_cond: torch.Tensor,
        lang_cond: list | None,
    ) -> dict[str, torch.Tensor]:
        """Get consistency velocity targets for training.

        Consistency training is used to train the model to be consistent across
        different timesteps using an EMA model.

        Args:
            actions: (B, T, action_dim) target actions
            vis_cond: (B, L, obs_feature_dim) visual conditioning
            lang_cond: Optional language conditioning (list of strings)

        Note: EMA model should be set via model.set_ema_model() before calling this.

        Returns:
            Dictionary with keys: x_t, t, target_t, v_target
        """
        target_dict = {}
        consistency_batchsize = actions.shape[0]
        device = actions.device

        # Sample t and dt for consistency training
        t_ct = self.sample_t(consistency_batchsize, mode=self.sample_t_mode_consistency).to(device)
        t_ct = t_ct.view(-1, 1, 1)
        delta_t1 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        delta_t2 = delta_t1.clone()  # Use the same delta_t

        # Compute next timestep
        t_next = t_ct.squeeze() + delta_t1
        t_next = torch.clamp(t_next, max=1.0)  # Clip t to ensure it does not exceed 1.0
        t_next = t_next.view(-1, 1, 1)

        # Compute target timestep for next step
        if self.sample_target_t_mode == "absolute":
            target_t_next = t_next.squeeze() + delta_t2
        elif self.sample_target_t_mode == "relative":
            target_t_next = delta_t2

        # Compute interpolated data points at timestep t and t_next
        x0_ct = torch.randn_like(actions, device=device)
        x1_ct = actions.to(device)
        x_t_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        x_t_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        if self.ema_model is None:
            raise RuntimeError("EMA model must be set via set_ema_model() before training")

        # Predict the average velocity from t_next toward next target (t_next + delta_t2)
        with torch.no_grad():
            v_avg_to_next_target = self.ema_model.model.action_predictor(
                sample=x_t_next,
                timestep=t_next.squeeze(),
                target_t=target_t_next.squeeze(),
                vis_cond=vis_cond,
                lang_cond=lang_cond,
            )

        # Predict the target data point using the average velocity
        pred_x1_ct = x_t_next + (1 - t_next) * v_avg_to_next_target
        # Estimate the velocity at t by using the predicted endpoint
        v_ct = (pred_x1_ct - x_t_ct) / (1 - t_ct)

        # Target_t_ct is the target timestep for the current timestep t
        target_t_ct = delta_t1 if self.sample_target_t_mode == "relative" else t_next.squeeze()

        target_dict["x_t"] = x_t_ct
        target_dict["t"] = t_ct
        target_dict["target_t"] = target_t_ct
        target_dict["v_target"] = v_ct

        return target_dict

    @torch.no_grad()
    def sample_ode(
        self,
        x0: torch.Tensor,
        vis_cond: torch.Tensor,
        lang_cond: list | None = None,
        N: int | None = None,
    ) -> list:
        """Sample trajectory using Euler ODE integration.

        Args:
            x0: (B, T, action_dim) initial noise
            vis_cond: (B, L, obs_feature_dim) visual conditioning
            lang_cond: Optional language conditioning
            N: Number of inference steps (default: self.num_inference_steps)

        Returns:
            List of tensors representing the trajectory from noise to data
        """
        if N is None:
            N = self.num_inference_steps

        dt = 1.0 / N
        traj = []
        x = x0.detach().clone()
        batchsize = x.shape[0]
        device = x.device

        t = torch.arange(0, N, device=device, dtype=x.dtype) / N
        traj.append(x.detach().clone())

        for i in range(N):
            ti = torch.ones((batchsize,), device=device) * t[i]
            if self.sample_target_t_mode == "absolute":
                target_t = ti + dt
            elif self.sample_target_t_mode == "relative":
                target_t = torch.full_like(ti, dt)

            pred = self.action_predictor(x, ti, target_t=target_t, vis_cond=vis_cond, lang_cond=lang_cond)
            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj

    def generate_actions(
        self,
        input_ids=None,
        pixel_values=None,
        actions=None,
        attention_mask=None,
        attention_mask_images=None,
        num_inference_steps=None,
        past_mask=None,
        proprioception=None,
        point_cloud=None,
        task_name=None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate actions using ODE sampling (inference interface compatible with DiffusionPolicy).

        This method provides a compatible interface with DiffusionPolicy for use in inference pipelines.
        ManiFlow uses point clouds instead of images, so pixel_values is ignored.

        Args:
            input_ids: Ignored (ManiFlow doesn't use text embeddings directly)
            pixel_values: Ignored (ManiFlow uses point clouds)
            actions: (B, T, action_dim) action tensor - past actions used for context, future ignored
            attention_mask: Ignored
            attention_mask_images: Ignored
            num_inference_steps: Number of ODE integration steps (overrides default if provided)
            past_mask: Ignored (ManiFlow doesn't use past masking)
            proprioception: (B, lowdim_past_timesteps, proprioception_dim) proprioceptive state
            point_cloud: (B, lowdim_past_timesteps, num_points, point_dim) point cloud observations
            task_name: Optional list of task names for language conditioning

        Returns:
            (B, T, action_dim) predicted action sequence where T = lowdim_past_timesteps + lowdim_future_timesteps
        """
        # Extract observations
        obs_dict = {
            "point_cloud": point_cloud,
            "proprioception": proprioception,
        }
        if task_name is not None:
            obs_dict["task_name"] = task_name

        # Call predict_action with optional custom num_inference_steps
        if num_inference_steps is not None:
            # Temporarily override num_inference_steps
            original_steps = self.num_inference_steps
            self.num_inference_steps = num_inference_steps
            predicted_actions = self.predict_action(obs_dict, lang_cond=task_name)
            self.num_inference_steps = original_steps
        else:
            predicted_actions = self.predict_action(obs_dict, lang_cond=task_name)

        # Pad with past actions to match expected output shape (B, T, action_dim)
        # where T = lowdim_past_timesteps + lowdim_future_timesteps
        if actions is not None:
            # Extract past actions from input
            past_actions = actions[:, : self.lowdim_past_timesteps, :]
            # Concatenate past and predicted future
            full_actions = torch.cat([past_actions, predicted_actions], dim=1)
        else:
            # If no past actions provided, just return predicted future with zero padding
            B = predicted_actions.shape[0]
            device = predicted_actions.device
            past_actions = torch.zeros(
                (B, self.lowdim_past_timesteps, self.action_dim),
                device=device,
                dtype=predicted_actions.dtype,
            )
            full_actions = torch.cat([past_actions, predicted_actions], dim=1)

        return full_actions

    def predict_action(
        self,
        obs_dict: dict[str, torch.Tensor],
        lang_cond: list | None = None,
    ) -> torch.Tensor:
        """Predict action from observations using ODE sampling.

        Args:
            obs_dict: Dictionary with keys:
                - 'point_cloud': (B, lowdim_past_timesteps, num_points, point_dim)
                - 'proprioception': (B, lowdim_past_timesteps, proprioception_dim)
                - optionally 'task_name': list of strings for language conditioning
            lang_cond: Optional language conditioning (overrides obs_dict['task_name'])

        Returns:
            (B, lowdim_future_timesteps, action_dim) predicted actions
        """
        device = next(self.parameters()).device

        # Validate point cloud input
        if obs_dict["point_cloud"] is None:
            raise ValueError(
                "ManiFlow requires point cloud data, but point_cloud is None. "
                "Please set 'use_point_cloud: true' in your data config."
            )

        # Extract observations
        point_cloud = obs_dict["point_cloud"][:, : self.lowdim_past_timesteps]  # (B, To, N, D)
        proprioception = obs_dict["proprioception"][:, : self.lowdim_past_timesteps]  # (B, To, P)

        B = point_cloud.shape[0]

        # Flatten temporal dimension for encoding
        point_cloud_flat = point_cloud.reshape(-1, *point_cloud.shape[2:])  # (B*To, N, D)
        proprioception_flat = proprioception.reshape(-1, proprioception.shape[2])  # (B*To, P)

        # Encode observations
        obs_features = self.obs_encoder(point_cloud_flat, proprioception_flat)  # (B*To, obs_feature_dim)
        vis_cond = obs_features.reshape(B, -1, self.obs_feature_dim)  # (B, To*L, obs_feature_dim)

        # Handle language conditioning
        if lang_cond is None and self.language_conditioned:
            lang_cond = obs_dict.get("task_name")

        # Initialize with noise
        noise = torch.randn(
            (B, self.horizon, self.action_dim),
            device=device,
            dtype=point_cloud.dtype,
        )

        # Sample trajectory
        traj = self.sample_ode(x0=noise, vis_cond=vis_cond, lang_cond=lang_cond)

        action_pred = traj[-1]  # (B, T, action_dim) where T = horizon
        start = self.lowdim_past_timesteps  # Start at anchor position
        end = start + self.lowdim_future_timesteps  # Extract future actions
        action = action_pred[:, start:end]  # (B, lowdim_future_timesteps, action_dim)

        return action

    def compute_loss(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined flow matching and consistency training loss.

        Args:
            batch: Dictionary with keys:
                - 'obs': Dictionary with 'point_cloud', 'proprioception', optionally 'task_name'
                - 'actions': (B, T, action_dim) target actions

        Returns:
            Tuple of (total_loss, loss_dict)
        """
        device = next(self.parameters()).device

        # Extract data
        obs_dict = batch["obs"]
        actions = batch["actions"].to(device)  # (B, T, action_dim)
        batch_size = actions.shape[0]

        # Validate point cloud input
        if obs_dict["point_cloud"] is None:
            raise ValueError(
                "ManiFlow requires point cloud data, but point_cloud is None. "
                "Please set 'use_point_cloud: true' in your data config."
            )

        # Extract observations
        point_cloud = obs_dict["point_cloud"][:, : self.lowdim_past_timesteps]  # (B, To, N, D)
        proprioception = obs_dict["proprioception"][:, : self.lowdim_past_timesteps]  # (B, To, P)

        # Flatten temporal dimension for encoding
        point_cloud_flat = point_cloud.reshape(-1, *point_cloud.shape[2:])  # (B*To, N, D)
        proprioception_flat = proprioception.reshape(-1, proprioception.shape[2])  # (B*To, P)

        # Encode observations
        obs_features = self.obs_encoder(point_cloud_flat, proprioception_flat)  # (B*To, obs_feature_dim)
        vis_cond = obs_features.reshape(batch_size, -1, self.obs_feature_dim)  # (B, To*L, obs_feature_dim)

        # Handle language conditioning
        lang_cond = None
        if self.language_conditioned:
            lang_cond = obs_dict.get("task_name", None)

        # Split batch for flow and consistency training
        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        consistency_batchsize = int(batch_size * self.consistency_batch_ratio)

        # Get flow velocity targets and compute flow loss
        flow_target_dict = self.get_flow_velocity(
            actions[:flow_batchsize],
            vis_cond=vis_cond[:flow_batchsize],
            lang_cond=lang_cond[:flow_batchsize] if lang_cond is not None else None,
        )
        v_flow_pred = self.action_predictor(
            sample=flow_target_dict["x_t"],
            timestep=flow_target_dict["t"].squeeze(),
            target_t=flow_target_dict["target_t"].squeeze(),
            vis_cond=vis_cond[:flow_batchsize],
            lang_cond=flow_target_dict["lang_cond"],
        )

        v_flow_target = flow_target_dict["v_target"]
        loss_flow = F.mse_loss(v_flow_pred, v_flow_target, reduction="none")
        loss_flow = reduce(loss_flow, "b ... -> b (...)", "mean")
        loss_flow_mean = loss_flow.mean()

        # Get consistency velocity targets and compute consistency loss
        consistency_target_dict = self.get_consistency_velocity(
            actions[flow_batchsize : flow_batchsize + consistency_batchsize],
            vis_cond=vis_cond[flow_batchsize : flow_batchsize + consistency_batchsize],
            lang_cond=lang_cond[flow_batchsize : flow_batchsize + consistency_batchsize]
            if lang_cond is not None
            else None,
        )
        v_ct_pred = self.action_predictor(
            sample=consistency_target_dict["x_t"],
            timestep=consistency_target_dict["t"].squeeze(),
            target_t=consistency_target_dict["target_t"].squeeze(),
            vis_cond=vis_cond[flow_batchsize : flow_batchsize + consistency_batchsize],
            lang_cond=lang_cond[flow_batchsize : flow_batchsize + consistency_batchsize]
            if lang_cond is not None
            else None,
        )

        v_ct_target = consistency_target_dict["v_target"]
        loss_ct = F.mse_loss(v_ct_pred, v_ct_target, reduction="none")
        loss_ct = reduce(loss_ct, "b ... -> b (...)", "mean")
        loss_ct_mean = loss_ct.mean()

        # Combine losses
        total_loss = loss_flow_mean + loss_ct_mean

        loss_dict = {
            "loss": total_loss.item(),
            "loss_flow": loss_flow_mean.item(),
            "loss_consistency": loss_ct_mean.item(),
        }

        return total_loss, loss_dict

    def forward(
        self,
        point_cloud: torch.Tensor | None = None,
        proprioception: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        task_name: list | None = None,
        **kwargs,
    ):
        """Forward pass - handles both training (with loss) and inference.

        For training: pass point_cloud, proprioception, actions, and optionally task_name
        For inference: pass point_cloud, proprioception, and optionally task_name

        Note: EMA model should be set via model.set_ema_model() before training.

        Returns:
            Training mode: (loss, loss_dict) tuple
            Inference mode: (B, lowdim_future_timesteps, action_dim) actions
        """
        if actions is not None:
            # Training mode: compute loss internally
            obs_dict = {"point_cloud": point_cloud, "proprioception": proprioception}
            if task_name is not None:
                obs_dict["task_name"] = task_name
            batch = {"obs": obs_dict, "actions": actions}
            return self.compute_loss(batch)
        else:
            # Inference mode: predict actions
            obs_dict = {"point_cloud": point_cloud, "proprioception": proprioception}
            if task_name is not None:
                obs_dict["task_name"] = task_name
            return self.predict_action(obs_dict, lang_cond=task_name)
