import torch

from vla_foundry.models.model_outputs.base_output import BaseOutput


class VisionLanguageBackboneOutput(BaseOutput):
    """Unified output for vision-language backbones used in action policies.

    This provides a consistent interface regardless of backbone type (VLM or CLIP).
    """

    def __init__(self, embeddings: torch.Tensor, attention_mask: torch.Tensor | None = None):
        super().__init__()
        # Primary embeddings for conditioning [B, N, D]
        # VLM: [B, 1, hidden_dim * num_layers] - single embedding from action token
        # CLIP: [B, 1+N, projection_dim] - concatenated [text, images] sequence
        self.embeddings = embeddings
        # Attention mask for the embeddings sequence [B, seq_len]
        # None means all positions are visible
        self.attention_mask = attention_mask
