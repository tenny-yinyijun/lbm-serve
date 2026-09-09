"""Diffusion Policy models and CLIP encoders."""

from vla_foundry.models.diffusion_policy.clip_hf import CLIPHF
from vla_foundry.models.diffusion_policy.clip_openclip import CLIP_OpenCLIP
from vla_foundry.models.diffusion_policy.diffusion_policy import DiffusionPolicy
from vla_foundry.models.registry import register_model
from vla_foundry.models.vlm_hf import VLMHF
from vla_foundry.params.model_params import ModelParams


@register_model("clip_openclip")
def create_clip_openclip(model_params: ModelParams, load_pretrained: bool = True):
    return CLIP_OpenCLIP(model_params)


@register_model("clip_hf")
@register_model("clip_backbone")
def create_clip_hf(model_params: ModelParams, load_pretrained: bool = True):
    return CLIPHF(model_params, load_pretrained=load_pretrained)


@register_model("vlm_backbone")
def create_vlm_backbone(model_params: ModelParams, load_pretrained: bool = True):
    return VLMHF(model_params, load_pretrained=load_pretrained)


@register_model("vlm_foundry_backbone")
def create_vlm_foundry_backbone(model_params: ModelParams, load_pretrained: bool = True):
    import os

    from vla_foundry.file_utils import load_model_checkpoint
    from vla_foundry.models.registry import create_model as _create_model
    from vla_foundry.params.model_params import VLMParams
    from vla_foundry.params.train_experiment_params import load_params_from_yaml

    # Prefer vlm_experiment_dir (FT configs); fall back to deriving from resume_from_checkpoint path
    ckpt_path = model_params.resume_from_checkpoint
    experiment_dir = getattr(model_params, "vlm_experiment_dir", None)
    if experiment_dir is None and ckpt_path is not None:
        # Derive experiment dir from checkpoint path: <experiment_dir>/checkpoints/checkpoint_N.pt
        experiment_dir = os.path.dirname(os.path.dirname(ckpt_path))
    if experiment_dir is None:
        raise ValueError(
            "vlm_foundry_backbone requires vlm_experiment_dir or resume_from_checkpoint to locate config_model.yaml"
        )
    vlm_params = load_params_from_yaml(VLMParams, os.path.join(experiment_dir, "config_model.yaml"))

    vlm = _create_model(vlm_params, load_pretrained=False)

    if load_pretrained and ckpt_path is not None:
        load_model_checkpoint(vlm, ckpt_path)

    return vlm


@register_model("diffusion_policy")
def create_diffusion_policy(model_params: ModelParams, load_pretrained: bool = True):
    from vla_foundry.models.diffusion import create_noise_scheduler
    from vla_foundry.models.registry import create_model
    from vla_foundry.models.vision_language_backbones import get_vision_language_backbone

    vision_language_backbone = get_vision_language_backbone(model_params.vision_language_backbone, load_pretrained)
    transformer = create_model(model_params.transformer, load_pretrained)
    noise_scheduler = create_noise_scheduler(model_params)
    return DiffusionPolicy(model_params, vision_language_backbone, transformer, noise_scheduler)


__all__ = [
    "CLIPHF",
    "CLIP_OpenCLIP",
    "DiffusionPolicy",
]
