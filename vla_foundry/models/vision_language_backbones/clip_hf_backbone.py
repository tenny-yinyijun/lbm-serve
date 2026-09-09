"""CLIP backbone wrapper for action policy conditioning."""

import torch

from vla_foundry.models.model_outputs.backbone_output import VisionLanguageBackboneOutput
from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.params.model_params import CLIPBackboneParams


class CLIPBackboneWrapper(BaseBackboneWrapper):
    """Wraps CLIPHF and provides conditioning embeddings for action policies."""

    def __init__(self, backbone_params: CLIPBackboneParams, load_pretrained: bool = True):
        super().__init__(backbone_params, load_pretrained)
        self.disable_text = backbone_params.disable_text

    def get_conditioning_embeddings_dim(self) -> int:
        """Return CLIP projection dimension."""
        return self._model.get_projection_dim()

    def _concatenate_for_conditioning(self, text_embeds, image_embeds, attention_mask_images=None):
        """Helper: Concatenate text and image embeddings into sequence with attention mask."""
        embeddings_list = []
        mask_list = []

        if not self.disable_text and text_embeds is not None:
            # [B, D] -> [B, 1, D]
            embeddings_list.append(text_embeds.unsqueeze(1))
            mask_list.append(torch.ones(text_embeds.shape[0], 1, dtype=torch.bool, device=text_embeds.device))

        if image_embeds is not None:
            # [B, D] or [B, N, D] -> [B, N, D]
            if image_embeds.ndim == 2:
                image_embeds = image_embeds.unsqueeze(1)
            if attention_mask_images is not None:
                image_mask = attention_mask_images.to(device=image_embeds.device, dtype=torch.bool)
                if image_mask.ndim == 1 and image_embeds.shape[1] == 1:
                    image_mask = image_mask.unsqueeze(1)
                if image_mask.shape != image_embeds.shape[:2]:
                    raise ValueError(
                        f"attention_mask_images shape {image_mask.shape} must match CLIP image embeddings "
                        f"shape {image_embeds.shape[:2]}."
                    )
                image_embeds = image_embeds * image_mask.unsqueeze(-1).to(dtype=image_embeds.dtype)
                mask_list.append(image_mask)
            else:
                mask_list.append(
                    torch.ones(
                        image_embeds.shape[0], image_embeds.shape[1], dtype=torch.bool, device=image_embeds.device
                    )
                )
            embeddings_list.append(image_embeds)

        if not embeddings_list:
            return None, None

        embeddings = torch.cat(embeddings_list, dim=1)
        attention_mask = torch.cat(mask_list, dim=1)
        return embeddings, attention_mask

    def _prepare_inputs(self, input_ids, pixel_values, attention_mask=None, attention_mask_images=None, **kwargs):
        """Store attention_mask_images for use in _extract_action_conditioning."""
        self._attention_mask_images = attention_mask_images
        return input_ids, pixel_values, attention_mask, attention_mask_images, kwargs

    def _extract_action_conditioning(self, outputs) -> VisionLanguageBackboneOutput:
        embeddings, attention_mask = self._concatenate_for_conditioning(
            outputs.text_embeds, outputs.image_embeds, self._attention_mask_images
        )
        return VisionLanguageBackboneOutput(embeddings=embeddings, attention_mask=attention_mask)
