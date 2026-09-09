"""
Models module with registry-based model creation.

This module imports all model definitions, which register themselves via decorators.
New models should be registered in their respective files or subdirectories.

TRIMMED FOR SERVING (lbm-serve). Upstream also eagerly imports
`vla_foundry.models.maniflow` ("ditx", "dp3_encoder", "maniflow"),
`vla_foundry.models.transformer_hf`, `vla_foundry.models.vlm` and
`vla_foundry.models.vlm_hf`, purely so those types self-register. The open-world LBM
checkpoints are `type: diffusion_policy` over a `clip_backbone` and a `transformer`, so
those registrations are unreachable here -- and dropping them is what removes `timm` and
`open-clip-torch` from the dependency set. Serving a checkpoint whose `config_model.yaml`
names any of those types will now fail in `create_model` with an unknown-type error rather
than silently working; re-add the import if that is ever needed.
"""

# Import registry functions first
# Model subdirectories
# Import batch handlers module to trigger registration
import vla_foundry.models.batch_handlers
import vla_foundry.models.diffusion  # noqa: F401  provides create_noise_scheduler
import vla_foundry.models.diffusion_policy  # registers "diffusion_policy", "clip_hf"

# Import all model modules to trigger their registrations
# Individual model files
import vla_foundry.models.transformer  # registers "transformer"

# Import helper function
from vla_foundry.models.diffusion import create_noise_scheduler

# Import registry functions
from vla_foundry.models.registry import create_batch_handler, create_model, register_model
