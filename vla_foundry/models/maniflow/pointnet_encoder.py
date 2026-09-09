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

"""PointNet encoder for 3D point cloud processing."""

import logging

import torch
import torch.nn as nn

from vla_foundry.params.model_params import DP3EncoderParams

logger = logging.getLogger(__name__)


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: list[int],
    activation_fn: type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> list[nn.Module]:
    """
    Create a multi layer perceptron (MLP).

    Args:
        input_dim: Dimension of the input vector
        output_dim: Dimension of output
        net_arch: Architecture of the neural net (number of units per layer)
        activation_fn: Activation function to use after each layer
        squash_output: Whether to squash output using Tanh

    Returns:
        List of modules
    """
    modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()] if len(net_arch) > 0 else []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class PointNetEncoderXYZ(nn.Module):
    """PointNet encoder for XYZ point clouds."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 256,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        pointwise: bool = False,
        **kwargs,
    ):
        """
        Args:
            in_channels: Feature size of input (3 for XYZ)
            out_channels: Output feature dimension
            use_layernorm: Whether to use LayerNorm
            final_norm: Final normalization ("none" or "layernorm")
            use_projection: Whether to use final projection
            pointwise: If True, return per-point features; otherwise max-pool
        """
        super().__init__()
        block_channel = [64, 128, 256]
        logging.info(f"[PointNetEncoderXYZ] use_layernorm: {use_layernorm}")
        logging.info(f"[PointNetEncoderXYZ] final_norm: {final_norm}")

        assert in_channels == 3, f"PointNetEncoderXYZ only supports 3 channels, got {in_channels}"

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            logging.info("[PointNetEncoderXYZ] not using projection")

        self.pointwise = pointwise

    def forward(self, x):
        """
        Args:
            x: (B, N, 3) point cloud

        Returns:
            (B, out_channels) if pointwise=False
            (B, N, out_channels) if pointwise=True
        """
        x = self.mlp(x)
        if not self.pointwise:
            x = torch.max(x, 1)[0]  # Max pooling over points
        x = self.final_projection(x)
        return x


class PointNetEncoderXYZRGB(nn.Module):
    """PointNet encoder for XYZRGB point clouds."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 256,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        pointwise: bool = False,
        **kwargs,
    ):
        """
        Args:
            in_channels: Feature size of input (6 for XYZRGB)
            out_channels: Output feature dimension
            use_layernorm: Whether to use LayerNorm
            final_norm: Final normalization ("none" or "layernorm")
            use_projection: Whether to use final projection
            pointwise: If True, return per-point features; otherwise max-pool
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        logging.info(f"[PointNetEncoderXYZRGB] use_layernorm: {use_layernorm}")
        logging.info(f"[PointNetEncoderXYZRGB] final_norm: {final_norm}")

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels), nn.LayerNorm(out_channels)
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.pointwise = pointwise

    def forward(self, x):
        """
        Args:
            x: (B, N, 6) point cloud with RGB

        Returns:
            (B, out_channels) if pointwise=False
            (B, N, out_channels) if pointwise=True
        """
        x = self.mlp(x)
        if not self.pointwise:
            x = torch.max(x, 1)[0]  # Max pooling over points
        x = self.final_projection(x)
        return x


class DP3Encoder(nn.Module):
    """Encoder that combines point cloud and proprioception features."""

    def __init__(self, model_params: DP3EncoderParams):
        """
        Args:
            model_params: DP3EncoderParams instance with all configuration
        """
        super().__init__()

        point_cloud_shape = model_params.point_cloud_shape
        proprioception_dim = model_params.proprioception_dim
        out_channel = model_params.out_channel
        use_pc_color = model_params.use_pc_color
        pointnet_type = model_params.pointnet_type
        pointcloud_encoder_cfg = model_params.pointcloud_encoder_cfg
        state_mlp_size = (64, 64)
        state_mlp_activation_fn = nn.ReLU

        self.n_output_channels = out_channel
        self.point_cloud_shape = point_cloud_shape
        self.proprioception_dim = proprioception_dim

        # Set state MLP size from config if provided
        if pointcloud_encoder_cfg.get("state_mlp_size") is not None:
            state_mlp_size = pointcloud_encoder_cfg["state_mlp_size"]

        logging.info(f"[DP3Encoder] State MLP size: {state_mlp_size}")
        logging.info(f"[DP3Encoder] Point cloud shape: {point_cloud_shape}")
        logging.info(f"[DP3Encoder] Proprioception dim: {proprioception_dim}")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        # Create pointnet encoder
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg["in_channels"] = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg["in_channels"] = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        # Create state MLP for proprioception
        if len(state_mlp_size) == 0:
            raise RuntimeError("State MLP size cannot be empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = list(state_mlp_size[:-1])
        output_dim = state_mlp_size[-1]

        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(proprioception_dim, output_dim, net_arch, state_mlp_activation_fn))

        self.pointwise = pointcloud_encoder_cfg.get("pointwise", False)

        logging.info(f"[DP3Encoder] Output dim: {self.n_output_channels}")
        logging.info(f"[DP3Encoder] Pointwise: {self.pointwise}")

    def forward(self, point_cloud: torch.Tensor, proprioception: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_cloud: (B, num_points, point_dim) point cloud tensor
            proprioception: (B, proprioception_dim) proprioception tensor

        Returns:
            (B, n_output_channels) if pointwise=False
            (B, num_points, n_output_channels) if pointwise=True
        """
        # Note: FPS sampling is done during preprocessing (or inference preprocessing)
        # Point clouds should already be sampled to the correct size

        # Slice point cloud to xyz only if not using color
        if not self.use_pc_color:
            point_cloud = point_cloud[..., :3]

        # Extract point cloud features
        pn_feat = self.extractor(point_cloud)  # (B, out_channel) or (B, N, out_channel)

        # Extract proprioception features
        state_feat = self.state_mlp(proprioception.float())  # (B, state_out_dim)

        # Expand state features if pointwise
        if len(pn_feat.shape) == 3:
            # Each point has a feature - expand state to match
            state_feat = state_feat.unsqueeze(1).expand(-1, pn_feat.shape[1], -1)

        # Concatenate features
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self):
        """Return output feature dimension."""
        return self.n_output_channels
