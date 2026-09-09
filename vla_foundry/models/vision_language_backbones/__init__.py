"""Vision-language backbone construction.

TRIMMED FOR SERVING (lbm-serve). Upstream imports all four wrappers eagerly --
`ViTBackboneWrapper` (pulls `timm` via `models/vit_hf.py`), `VLMHFBackboneWrapper` and
`VLMFoundryBackboneWrapper` -- and dispatches on `backbone_params.type`. The open-world LBM
checkpoints are all `type: clip_backbone`, so only that branch is kept; the others now raise
with an explicit message instead of being unreachable imports.
"""

from vla_foundry.models.vision_language_backbones.base import BaseBackboneWrapper
from vla_foundry.models.vision_language_backbones.clip_hf_backbone import CLIPBackboneWrapper
from vla_foundry.params.model_params import CLIPBackboneParams

_TRIMMED = ("vlm_backbone", "vlm_foundry_backbone", "vit_backbone")


def get_vision_language_backbone(backbone_params, load_pretrained: bool = True):
    # Note: We do this instead of isintance() to avoid issues with class overlaps
    t = backbone_params.type
    if t == "clip_backbone":
        return CLIPBackboneWrapper(backbone_params, load_pretrained)
    elif t in _TRIMMED:
        raise ValueError(
            f"backbone type '{t}' was removed from this serving-only extraction of "
            f"vla_foundry (see vla_foundry/models/vision_language_backbones/__init__.py). "
            f"Use vla_foundry_internal to serve this checkpoint, or restore the wrapper."
        )
    else:
        raise ValueError(f"Unsupported vision language backbone type: {t}")
