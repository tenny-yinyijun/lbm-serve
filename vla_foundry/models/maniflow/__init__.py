"""ManiFlow models including DiTX, DP3Encoder, and ManiFlow."""

from vla_foundry.models.maniflow.ditx import DiTX
from vla_foundry.models.maniflow.maniflow import ManiFlow
from vla_foundry.models.maniflow.pointnet_encoder import DP3Encoder
from vla_foundry.models.registry import register_model
from vla_foundry.params.model_params import DiTXParams, DP3EncoderParams, ManiFlowParams


@register_model("ditx")
def create_ditx(model_params: DiTXParams, load_pretrained: bool = True):
    return DiTX(model_params)


@register_model("dp3_encoder")
def create_dp3_encoder(model_params: DP3EncoderParams, load_pretrained: bool = True):
    return DP3Encoder(model_params)


@register_model("maniflow")
def create_maniflow(model_params: ManiFlowParams, load_pretrained: bool = True):
    from vla_foundry.models.registry import create_model

    encoder = create_model(model_params.encoder, load_pretrained)
    action_predictor = create_model(model_params.action_predictor, load_pretrained)
    return ManiFlow(model_params, encoder, action_predictor)


__all__ = [
    "DiTX",
    "DP3Encoder",
    "ManiFlow",
]
