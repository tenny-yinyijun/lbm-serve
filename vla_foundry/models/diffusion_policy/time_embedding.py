import math

import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    """
    Learnable time embedding leveraging rotary-style rotations.

    Each forward pass takes scalar timestamps in [0, 1] and rotates every
    consecutive pair of the learnable vector by an angle in [0, π].
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("TimeEmbedding dimension must be even to form pairs.")

        self.dim = dim
        self.num_pairs = dim // 2
        self.time_vector = nn.Parameter(torch.randn(dim))

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: Tensor shaped (batch,) or (batch, 1) containing floats
                typically within [0, 1].
        Returns:
            Tensor shaped (batch, dim) produced by rotating the learnable vector
            according to each timestamp.
        """
        if timestamps.ndim == 0:
            timestamps = timestamps.unsqueeze(0)
        if timestamps.ndim > 1:
            timestamps = timestamps.view(-1)

        angles = timestamps.to(self.time_vector.dtype) * math.pi
        cos_vals = torch.cos(angles).view(-1, 1, 1)
        sin_vals = torch.sin(angles).view(-1, 1, 1)

        base = self.time_vector.view(1, self.num_pairs, 2)
        x = base[..., 0]
        y = base[..., 1]

        rotated_x = x * cos_vals - y * sin_vals
        rotated_y = x * sin_vals + y * cos_vals

        return torch.stack((rotated_x, rotated_y), dim=-1).view(-1, self.dim)
