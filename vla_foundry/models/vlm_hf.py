import inspect
import warnings

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText

from vla_foundry.models.registry import register_model
from vla_foundry.models.transformer_base import TransformerBase
from vla_foundry.models.utils import get_hidden_dim_hf, get_num_hidden_layers_hf
from vla_foundry.params.model_params import VLMHFParams


class VLMHF(TransformerBase):
    def __init__(self, model_params: VLMHFParams, load_pretrained: bool = True):
        super().__init__(model_params)
        self.model_name = model_params.hf_pretrained
        if load_pretrained:
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_name, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModelForImageTextToText.from_config(config, trust_remote_code=True)
        self._limit_hidden_states_to_last_n = None
        self._supports_pixel_attention_mask = self._forward_supports_pixel_attention_mask()
        self._setup_model_info()

    def _forward_supports_pixel_attention_mask(self):
        """Return whether the HF model explicitly declares pixel_attention_mask."""
        try:
            return "pixel_attention_mask" in inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            return False

    def _setup_model_info(self):
        """Detect model-specific information like hidden dimensions and expected image size."""
        model = self.model
        config = model.config

        # Detect language model component
        language_model_ref = None
        for attr in ["language_model", "text_model", "llm", "model"]:
            if hasattr(model, attr):
                candidate = getattr(model, attr)
                if candidate is not model and hasattr(candidate, "forward"):
                    language_model_ref = candidate
                    break

        # Detect language model hidden dimension
        self._lm_hidden_dim = None
        if language_model_ref is not None:
            lm_config = getattr(language_model_ref, "config", None)
            if lm_config is not None:
                for attr in ["hidden_size", "d_model", "n_embd", "hidden_dim"]:
                    if hasattr(lm_config, attr):
                        self._lm_hidden_dim = getattr(lm_config, attr)
                        break

        # Detect patches per image from vision config
        self._patches_per_image = None
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            img_size = getattr(vision_config, "image_size", None)
            patch_size = getattr(vision_config, "patch_size", 14)
            if isinstance(img_size, int) and isinstance(patch_size, int):
                self._patches_per_image = (img_size // patch_size) ** 2
            elif isinstance(img_size, (list, tuple)) and isinstance(patch_size, int):
                self._patches_per_image = (img_size[0] // patch_size) * (img_size[1] // patch_size)

    @property
    def lm_hidden_dim(self):
        """Get the language model's hidden dimension (may differ from embedding dim)."""
        if self._lm_hidden_dim is not None:
            return self._lm_hidden_dim
        # Fallback to general hidden_dim
        return self.hidden_dim

    def _expand_image_mask_to_pixel_mask(self, pixel_values, attention_mask_images, image_grid_thw=None):
        """Convert [B, N] image masks into HF pixel_attention_mask shape."""
        if pixel_values is None or attention_mask_images is None:
            return None

        if pixel_values.ndim == 2:
            if image_grid_thw is None or image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
                raise ValueError("Packed 2-D pixel_values require image_grid_thw with shape [num_images, 3].")
            image_mask = attention_mask_images.to(device=pixel_values.device, dtype=torch.bool).reshape(-1)
            if image_mask.shape[0] != image_grid_thw.shape[0]:
                raise ValueError(
                    f"attention_mask_images contains {image_mask.shape[0]} images, but image_grid_thw "
                    f"contains {image_grid_thw.shape[0]}."
                )
            patches_per_image = image_grid_thw.to(device=pixel_values.device).prod(dim=-1).to(dtype=torch.long)
            if patches_per_image.sum().item() != pixel_values.shape[0]:
                raise ValueError(
                    f"image_grid_thw describes {patches_per_image.sum().item()} packed patches, but "
                    f"pixel_values contains {pixel_values.shape[0]}."
                )
            return torch.repeat_interleave(image_mask, patches_per_image)

        spatial_shape = pixel_values.shape[-2:]
        if pixel_values.ndim == 5:
            expected_shape = pixel_values.shape[:2]
            if attention_mask_images.shape != expected_shape:
                raise ValueError(
                    f"attention_mask_images shape {attention_mask_images.shape} must match pixel_values "
                    f"image dimensions {expected_shape}."
                )
            image_mask = attention_mask_images
        elif pixel_values.ndim == 4:
            if attention_mask_images.ndim == 2 and attention_mask_images.numel() == pixel_values.shape[0]:
                image_mask = attention_mask_images.reshape(pixel_values.shape[0])
            elif attention_mask_images.ndim == 1 and attention_mask_images.shape[0] == pixel_values.shape[0]:
                image_mask = attention_mask_images
            else:
                raise ValueError(
                    f"Cannot align attention_mask_images shape {attention_mask_images.shape} with flat "
                    f"pixel_values shape {pixel_values.shape}."
                )
        else:
            return None

        return (
            image_mask.to(device=pixel_values.device, dtype=torch.bool)[..., None, None]
            .expand(*image_mask.shape, *spatial_shape)
            .contiguous()
        )

    def _apply_attention_mask_images(
        self, pixel_values, attention_mask_images, pixel_attention_mask, image_grid_thw=None
    ):
        """Apply image-level masks to HF pixel attention masks."""
        if attention_mask_images is None:
            return pixel_attention_mask

        image_mask = self._expand_image_mask_to_pixel_mask(
            pixel_values, attention_mask_images, image_grid_thw=image_grid_thw
        )
        if pixel_attention_mask is None:
            return image_mask

        image_mask = image_mask.to(device=pixel_attention_mask.device, dtype=torch.bool)
        if image_mask.ndim > pixel_attention_mask.ndim:
            raise ValueError(
                f"attention_mask_images shape {attention_mask_images.shape} has more dimensions than "
                f"pixel_attention_mask shape {pixel_attention_mask.shape}."
            )
        while image_mask.ndim < pixel_attention_mask.ndim:
            image_mask = image_mask.unsqueeze(-1)

        for mask_dim, target_dim in zip(image_mask.shape, pixel_attention_mask.shape, strict=True):
            if mask_dim not in (1, target_dim):
                raise ValueError(
                    f"Cannot align attention_mask_images shape {attention_mask_images.shape} with "
                    f"pixel_attention_mask shape {pixel_attention_mask.shape}."
                )
        image_mask = image_mask.expand_as(pixel_attention_mask).contiguous()

        if pixel_attention_mask.dtype == torch.bool:
            return pixel_attention_mask & image_mask
        return pixel_attention_mask * image_mask.to(dtype=pixel_attention_mask.dtype)

    def _zero_masked_pixel_values(self, pixel_values, attention_mask_images, image_grid_thw=None):
        """Zero complete missing-image tensors as a model-independent fallback."""
        pixel_mask = self._expand_image_mask_to_pixel_mask(
            pixel_values, attention_mask_images, image_grid_thw=image_grid_thw
        )
        if pixel_mask is None:
            return pixel_values
        channel_dim = -1 if pixel_values.ndim == 2 else -3
        return pixel_values * pixel_mask.unsqueeze(channel_dim).to(dtype=pixel_values.dtype)

    def forward(
        self,
        input_ids,
        pixel_values,
        attention_mask=None,
        attention_mask_images=None,
        output_hidden_states=False,
        **kwargs,
    ):
        # Handle multi-image input [B, N, C, H, W]
        # Convert pixel_values to bfloat16 (handles both standard and Qwen formats)
        image_grid_thw = kwargs.get("image_grid_thw")
        pixel_attention_mask = self._apply_attention_mask_images(
            pixel_values=pixel_values,
            attention_mask_images=attention_mask_images,
            pixel_attention_mask=kwargs.get("pixel_attention_mask"),
            image_grid_thw=image_grid_thw,
        )
        if pixel_attention_mask is not None and self._supports_pixel_attention_mask:
            kwargs["pixel_attention_mask"] = pixel_attention_mask
        elif pixel_attention_mask is not None:
            kwargs.pop("pixel_attention_mask", None)
            # Fallback only for models that cannot consume pixel_attention_mask. It is not
            # applied when the mask is honored: Idefics3/SmolVLM discard all-zero images as
            # batch padding, so zeroing a masked camera leaves fewer image embeddings than
            # <image> placeholder tokens and the embedding scatter raises
            # "Number of elements of source < number of ones in mask".
            pixel_values = self._zero_masked_pixel_values(
                pixel_values, attention_mask_images, image_grid_thw=image_grid_thw
            )
            warnings.warn(
                f"{type(self.model).__name__}.forward does not explicitly support pixel_attention_mask; "
                "missing images are zeroed, but their visual tokens may still be attended to.",
                UserWarning,
                stacklevel=2,
            )

        out = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        if self._limit_hidden_states_to_last_n is not None and output_hidden_states:
            out.hidden_states = out.hidden_states[-self._limit_hidden_states_to_last_n :]

        return out

    def resize_token_embeddings(self, new_num_tokens: int = None) -> int:
        """Add a new token to the vocabulary and return its ID.

        Args:
            new_num_tokens: The new vocabulary size. If None, adds exactly one token.

        Returns:
            The ID (index) of the newly added token.
        """
        current_size = int(self.model.get_input_embeddings().num_embeddings)

        if new_num_tokens is None:
            new_num_tokens = current_size + 1

        if new_num_tokens > current_size:
            print(f"Resizing token embeddings from {current_size} to {new_num_tokens}")
            self.model.resize_token_embeddings(new_num_tokens, mean_resizing=False)

        # Return the ID of the last token (the newly added one)
        return new_num_tokens - 1

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        if hasattr(self.model, "gradient_checkpointing_enable"):
            if enable:
                self.model.gradient_checkpointing_enable()
            else:
                self.model.gradient_checkpointing_disable()

    @property
    def hidden_dim(self) -> int:
        return get_hidden_dim_hf(self.model.config)

    @property
    def num_hidden_layers(self) -> int:
        return get_num_hidden_layers_hf(self.model.config)

    def generate(self, input_ids, pixel_values, attention_mask, max_new_tokens=20, **kwargs):
        """Generate text tokens using the VLM HF model"""
        # Add batch dimension if needed
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            attention_mask = attention_mask.unsqueeze(0)

        generated = input_ids.clone()
        attn_mask = attention_mask.clone()

        for _ in range(max_new_tokens):
            outputs = self.forward(input_ids=generated, pixel_values=pixel_values, attention_mask=attn_mask, **kwargs)
            last_output = outputs.logits[:, -1, :]
            next_token = torch.argmax(last_output, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)

            # Update attention mask: 1 for non-padding tokens
            next_token_mask = torch.ones_like(next_token, dtype=attn_mask.dtype)
            attn_mask = torch.cat([attn_mask, next_token_mask], dim=-1)

        return generated

    def get_fsdp_block_types(self):
        """Return block types for FSDP wrapping."""
        block_types = set()

        # Find text/language model blocks
        for attr in ["language_model", "text_model"]:
            if hasattr(self.model.model, attr):
                for _name, module in getattr(self.model.model, attr).named_modules():
                    if isinstance(module, nn.ModuleList) and len(module) > 0:
                        block_types.add(type(module[0]))

        # Find vision model blocks
        if hasattr(self.model.model, "vision_model") and hasattr(self.model.model.vision_model, "encoder"):
            for _name, module in self.model.model.vision_model.encoder.named_modules():
                if isinstance(module, nn.ModuleList) and len(module) > 0:
                    block_types.add(type(module[0]))

        if not block_types:
            raise ValueError("Could not find any model block classes.")

        return tuple(block_types)


@register_model("vlm_hf")
def create_vlm_hf(model_params: VLMHFParams, load_pretrained: bool = True):
    return VLMHF(model_params, load_pretrained=load_pretrained)
