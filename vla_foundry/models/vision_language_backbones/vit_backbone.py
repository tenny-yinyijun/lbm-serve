"""ViT backbone wrapper for action policy conditioning.

Wraps a standalone ViT (no text encoder) and provides patch embeddings
as conditioning tokens for the diffusion transformer.
"""

import einops
import torch

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.models.vit import ViT
from vla_foundry.params.model_params import ViTBackboneParams


class ViTBackboneWrapper(BaseBackboneWrapper):
    """Wraps a ViT and provides patch embeddings for action policy conditioning."""

    def __init__(self, backbone_params: ViTBackboneParams, load_pretrained: bool = True):
        # Skip BaseBackboneWrapper.__init__ which calls create_model with the wrong type.
        # Instead, initialize BaseModel directly and create the ViT ourselves.
        BaseModel.__init__(self, backbone_params)
        self._model = ViT(backbone_params)

    def get_conditioning_embeddings_dim(self) -> int:
        return self._model.model_params.hidden_dim

    def _expand_image_mask_to_token_mask(self, embeddings, attention_mask_images):
        """Expand image-level masks to match ViT conditioning tokens."""
        if attention_mask_images is None:
            return None

        image_mask = attention_mask_images.to(device=embeddings.device, dtype=torch.bool)
        if embeddings.ndim == 4:
            expected_shape = embeddings.shape[:2]
            if image_mask.shape != expected_shape:
                raise ValueError(
                    f"attention_mask_images shape {image_mask.shape} must match ViT image dimensions {expected_shape}."
                )
            return image_mask[:, :, None].expand(*expected_shape, embeddings.shape[2]).contiguous()

        if embeddings.ndim == 3:
            if image_mask.ndim == 2:
                if image_mask.numel() == embeddings.shape[0]:
                    image_mask = image_mask.reshape(-1)
                else:
                    raise ValueError(
                        f"Cannot align attention_mask_images shape {image_mask.shape} with ViT embeddings shape "
                        f"{embeddings.shape}."
                    )
            elif image_mask.ndim != 1 or image_mask.shape[0] != embeddings.shape[0]:
                raise ValueError(
                    f"Cannot align attention_mask_images shape {image_mask.shape} with ViT embeddings shape "
                    f"{embeddings.shape}."
                )
            return image_mask[:, None].expand(embeddings.shape[0], embeddings.shape[1]).contiguous()

        return None

    def get_action_conditioning(
        self,
        input_ids: torch.Tensor | None,
        pixel_values: torch.Tensor | None,
        attention_mask: torch.Tensor | None = None,
        attention_mask_images: torch.Tensor | None = None,
        **kwargs,
    ) -> VisionLanguageBackboneOutput:
        # pixel_values may arrive as [B, N, C, H, W] (5D) or [B*N, C, H, W] (4D).
        # HF image processors flatten multi-image samples to [B*N, C, H, W], so
        # reshape back to 5D before ViT.forward when the batch/image layout can
        # be inferred from attention_mask_images or input_ids.
        if pixel_values.ndim == 4 and (input_ids is not None or attention_mask_images is not None):
            if attention_mask_images is not None:
                if attention_mask_images.ndim != 2:
                    raise ValueError(
                        f"Cannot align attention_mask_images shape {attention_mask_images.shape} with "
                        f"4D pixel_values batch size {pixel_values.shape[0]}."
                    )
                batch_size, num_images = attention_mask_images.shape
            else:
                batch_size = input_ids.shape[0]
                num_images = pixel_values.shape[0] // batch_size
            if num_images * batch_size != pixel_values.shape[0]:
                if attention_mask_images is not None:
                    raise ValueError(
                        f"Cannot align attention_mask_images shape {attention_mask_images.shape} with "
                        f"pixel_values batch size {pixel_values.shape[0]}."
                    )
                raise ValueError(
                    f"pixel_values batch size ({pixel_values.shape[0]}) is not divisible by "
                    f"input_ids batch size ({batch_size}); cannot infer images-per-sample."
                )
            pixel_values = pixel_values.reshape(batch_size, num_images, *pixel_values.shape[1:])

        embeddings = self._model(pixel_values)  # [B, N, num_patches, hidden_dim]
        attention_mask = self._expand_image_mask_to_token_mask(embeddings, attention_mask_images)
        if attention_mask is not None:
            embeddings = embeddings * attention_mask[..., None].to(dtype=embeddings.dtype)
        # Flatten camera and patch dims into a single sequence
        if embeddings.ndim == 4:
            embeddings = einops.rearrange(embeddings, "b n t d -> b (n t) d")
            if attention_mask is not None:
                attention_mask = einops.rearrange(attention_mask, "b n t -> b (n t)")
        return VisionLanguageBackboneOutput(embeddings=embeddings, attention_mask=attention_mask)

    def _extract_action_conditioning(self, outputs) -> VisionLanguageBackboneOutput:
        # Not used — we override get_action_conditioning directly
        raise NotImplementedError
