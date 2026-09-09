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

"""DiTX: Diffusion Transformer for Consistency Flow Training."""

import logging
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Mlp, RmsNorm

from vla_foundry.models.maniflow.ditx_block import AdaptiveLayerNorm, DiTXBlock
from vla_foundry.models.maniflow.positional_embedding import SinusoidalPosEmb
from vla_foundry.params.model_params import DiTXParams

logger = logging.getLogger(__name__)


class FinalLayer(nn.Module):
    """The final layer of DiTX."""

    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RmsNorm(hidden_size, eps=1e-6)

        def approx_gelu():
            return nn.GELU(approximate="tanh")

        self.ffn_final = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size,
            out_features=out_channels,
            act_layer=approx_gelu,
            drop=0,
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.ffn_final(x)
        return x


class DiTX(nn.Module):
    """Consistency Flow Training model with a Diffusion Transformer backbone."""

    def __init__(self, model_params: DiTXParams):
        """
        Args:
            model_params: DiTXParams instance with all configuration
        """
        super().__init__()

        # Extract parameters from model_params
        input_dim = model_params.action_dim
        output_dim = model_params.action_dim
        horizon = model_params.horizon
        lowdim_past_timesteps = model_params.lowdim_past_timesteps
        cond_dim = model_params.cond_dim
        visual_cond_len = model_params.visual_cond_len
        diffusion_timestep_embed_dim = model_params.diffusion_timestep_embed_dim
        diffusion_target_t_embed_dim = model_params.diffusion_target_t_embed_dim
        n_layer = model_params.n_layer
        n_head = model_params.n_head
        n_emb = model_params.n_emb
        qkv_bias = model_params.qkv_bias
        qk_norm = model_params.qk_norm
        pre_norm_modality = model_params.pre_norm_modality
        language_conditioned = model_params.language_conditioned

        # Constants not in params (use defaults)
        mlp_ratio = 4.0
        p_drop_attn = 0.1
        language_model = "t5-small"
        block_type = "DiTX"

        self.lowdim_past_timesteps = lowdim_past_timesteps
        self.visual_cond_len = visual_cond_len
        self.language_conditioned = language_conditioned
        self.pre_norm_modality = pre_norm_modality

        # constants
        T = horizon
        self.T = T
        self.horizon = horizon

        # input embedding stem
        self.hidden_dim = n_emb
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.vis_cond_obs_emb = nn.Linear(cond_dim, n_emb)  # visual condition observation embedding
        self.vis_cond_pos_embed = nn.Parameter(
            torch.zeros(1, visual_cond_len * lowdim_past_timesteps, n_emb)
        )  # learnable visual condition positional embedding

        # pre-norm visual modality
        if self.pre_norm_modality:
            # If pre-norm modality is used, apply adaLN modulation before the transformer blocks
            self.vis_norm = AdaptiveLayerNorm(
                dim=n_emb,
                dim_cond=n_emb,
            )

        # timestep and target_t cond encoder
        flow_timestep_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_timestep_embed_dim),
            nn.Linear(diffusion_timestep_embed_dim, diffusion_timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_timestep_embed_dim * 4, n_emb),
        )
        flow_target_t_encoder = nn.Sequential(
            SinusoidalPosEmb(diffusion_target_t_embed_dim),
            nn.Linear(diffusion_target_t_embed_dim, diffusion_target_t_embed_dim * 4),
            nn.Mish(),
            nn.Linear(diffusion_target_t_embed_dim * 4, self.hidden_dim),
        )
        self.flow_timestep_encoder = flow_timestep_encoder
        self.flow_target_t_encoder = flow_target_t_encoder
        self.timestep_target_t_adaptor = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        # Language conditioning, use T5-small as default
        if self.language_conditioned:
            self.load_T5_encoder(model_name=language_model, freeze=True)
            self.lang_adaptor = self.build_condition_adapter(
                "mlp2x_gelu",
                in_features=self.language_encoder_out_dim,
                out_features=n_emb,
            )
            # pre-norm language modality
            if self.pre_norm_modality:
                self.lang_norm = AdaptiveLayerNorm(
                    dim=n_emb,
                    dim_cond=n_emb,
                )

        # Building the transformer blocks
        self.block_type = block_type
        if block_type == "DiTX":
            self.blocks = nn.ModuleList(
                [
                    DiTXBlock(
                        n_emb,
                        n_head,
                        mlp_ratio=mlp_ratio,
                        p_drop_attn=p_drop_attn,
                        qkv_bias=qkv_bias,
                        qk_norm=qk_norm,
                    )
                    for _ in range(n_layer)
                ]
            )
            logging.info(
                f"[DiTX Transformer] Initialized {n_layer} DiTX blocks with hidden size {n_emb}, "
                f"num heads {n_head}, mlp ratio {mlp_ratio}, dropout {p_drop_attn}, "
                f"qkv_bias {qkv_bias}, qk_norm {qk_norm}"
            )

        # Final Layer
        self.final_layer = FinalLayer(n_emb, output_dim)

        self.initialize_weights()
        logging.info("[DiTX Transformer] Initialized weights for DiTX")

        num_params = sum(p.numel() for p in self.parameters())
        logging.info(f"[DiTX] Number of parameters: {num_params:,}")

    def build_condition_adapter(self, projector_type, in_features, out_features):
        """Build adapter for conditioning signals."""
        projector = None
        if projector_type == "linear":
            projector = nn.Linear(in_features, out_features)
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(in_features, out_features)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU(approximate="tanh"))
                    modules.append(nn.Linear(out_features, out_features))
                projector = nn.Sequential(*modules)

        if projector is None:
            raise ValueError(f"Unknown projector type: {projector_type}")

        return projector

    # language encoder
    def load_T5_encoder(self, model_name, freeze=True):
        """Load T5 encoder for language conditioning."""
        from transformers import AutoTokenizer, T5Config, T5EncoderModel

        T5_model_name = ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b"]
        assert model_name in T5_model_name, f"Model name {model_name} not in {T5_model_name}"
        encoder_name = model_name
        pretrained_model_id = f"google-t5/{encoder_name}"
        encoder_cfg = T5Config()
        self.language_encoder = T5EncoderModel(encoder_cfg).from_pretrained(pretrained_model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_id)
        if freeze:
            self.language_encoder.eval()
            # freeze the language encoder
            for param in self.language_encoder.parameters():
                param.requires_grad = False

        self.language_encoder_out_dim = 512
        logging.info(f"Loaded T5 encoder: {encoder_name}")

    def encode_text_input_T5(
        self,
        lang_cond,
        norm_lang_embedding=False,
        output_type="sentence",
        device="cuda",
    ):
        """Encode text input using T5."""
        language_inputs = self.tokenizer(
            lang_cond,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = language_inputs["input_ids"].to(device)
        attention_mask = language_inputs["attention_mask"].to(device)
        encoder_outputs = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        token_embeddings = encoder_outputs.last_hidden_state
        if output_type == "token":
            return token_embeddings
        # obtain sentence embedding by averaging the token embeddings
        sentence_embedding = torch.mean(token_embeddings, dim=1).squeeze(1)  # (B, 512)
        if norm_lang_embedding:
            sentence_embedding = F.normalize(sentence_embedding, p=2, dim=-1)

        return sentence_embedding

    def initialize_weights(self):
        """Initialize model weights."""
        for block in self.blocks:
            # Initialize self_attn's in_proj_weight and out_proj
            nn.init.xavier_uniform_(block.self_attn.in_proj_weight)
            if block.self_attn.in_proj_bias is not None:
                nn.init.zeros_(block.self_attn.in_proj_bias)

            nn.init.xavier_uniform_(block.self_attn.out_proj.weight)
            if block.self_attn.out_proj.bias is not None:
                nn.init.zeros_(block.self_attn.out_proj.bias)

        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize input emb by normal distribution:
        nn.init.normal_(self.input_emb.weight, std=0.02)
        nn.init.constant_(self.input_emb.bias, 0) if self.input_emb.bias is not None else None

        # Initialize pos emb by normal distribution:
        nn.init.normal_(self.pos_emb, std=0.02)

        # Initialize diffusion step encoder:
        for layer in self.flow_timestep_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize diffusion target_t encoder:
        for layer in self.flow_target_t_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

        # Initialize conditional observation embedding:
        nn.init.normal_(self.vis_cond_obs_emb.weight, std=0.02)
        nn.init.constant_(self.vis_cond_obs_emb.bias, 0) if self.vis_cond_obs_emb.bias is not None else None

        # Initialize the adapter for timestep and target_t
        nn.init.normal_(self.timestep_target_t_adaptor.weight, std=0.02)
        nn.init.constant_(self.timestep_target_t_adaptor.bias, 0)

        if self.language_conditioned:
            # Initialize the language condition adapter
            nn.init.normal_(self.lang_adaptor[0].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[0].bias, 0) if self.lang_adaptor[0].bias is not None else None
            nn.init.normal_(self.lang_adaptor[-1].weight, std=0.02)
            nn.init.constant_(self.lang_adaptor[-1].bias, 0) if self.lang_adaptor[-1].bias is not None else None

        if self.pre_norm_modality:
            # Initialize the adaptive layer norm for visual condition
            nn.init.zeros_(self.vis_norm.cond_linear.weight)
            nn.init.constant_(self.vis_norm.cond_linear.bias[: self.hidden_dim], 1.0)
            nn.init.zeros_(self.vis_norm.cond_linear.bias[self.hidden_dim :])
            if self.language_conditioned:
                # Initialize the adaptive layer norm for language condition
                nn.init.zeros_(self.lang_norm.cond_linear.weight)
                nn.init.constant_(self.lang_norm.cond_linear.bias[: self.hidden_dim], 1.0)
                nn.init.zeros_(self.lang_norm.cond_linear.bias[self.hidden_dim :])

        # Initialize the final layer: zero-out the final linear layer
        nn.init.constant_(self.final_layer.ffn_final.fc2.weight, 0)
        nn.init.constant_(self.final_layer.ffn_final.fc2.bias, 0)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        """
        Separate parameters into those that will experience weight decay
        for regularization and those that won't (biases, layernorm/embedding weights).
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, RmsNorm)
        for mn, m in self.named_modules():
            for pn, _p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        if self.vis_cond_pos_embed is not None:
            # this is a learnable parameter, so we don't want to decay it
            no_decay.add("vis_cond_pos_embed")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups

    def configure_optimizers(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
    ):
        """Configure AdamW optimizer with weight decay."""
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        target_t: torch.Tensor | float | int,
        vis_cond: torch.Tensor,
        lang_cond: torch.Tensor | list | str = None,
        **kwargs,
    ):
        """
        Forward pass of the DiTX model.

        Args:
            sample: (B, T, input_dim) noisy action input
            timestep: (B,) or scalar, flow time step t
            target_t: (B,) or scalar, target absolute or relative time
            vis_cond: (B, L, vis_cond_dim) visual condition
            lang_cond: (B,) or list of strings, language condition input

        Returns:
            (B, T, output_dim) predicted velocity
        """
        # process input
        input_emb = self.input_emb(sample)  # (B, T, n_emb)
        x = input_emb + self.pos_emb  # (B, T, n_emb)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension
        timesteps = timesteps.expand(sample.shape[0])
        timestep_embed = self.flow_timestep_encoder(timesteps)  # (B, n_emb)

        # 2. target_t
        target_ts = target_t
        if not torch.is_tensor(target_ts):
            target_ts = torch.tensor([target_ts], dtype=torch.float32, device=sample.device)
        elif torch.is_tensor(target_ts) and len(target_ts.shape) == 0:
            target_ts = target_ts[None].to(sample.device)
        target_ts = target_ts.expand(sample.shape[0])
        target_t_embed = self.flow_target_t_encoder(target_ts)  # (B, n_emb)

        time_c = torch.cat([timestep_embed, target_t_embed], dim=-1)  # (B, 2*n_emb)
        time_c = self.timestep_target_t_adaptor(time_c)  # (B, n_emb)

        # 3. visual condition
        vis_con_obs_emb = self.vis_cond_obs_emb(vis_cond)  # (B, L, n_emb)
        vis_cond_pos_embed = self.vis_cond_pos_embed[:, : vis_cond.shape[1]]
        context_c = vis_con_obs_emb + vis_cond_pos_embed  # (B, L, n_emb)
        if self.pre_norm_modality:
            context_c = self.vis_norm(context_c, time_c)

        # 4. language condition
        if self.language_conditioned:
            assert lang_cond is not None
            lang_c = self.encode_text_input_T5(lang_cond, output_type="token", device=sample.device)  # (B, L_lang, 512)
            lang_c = self.lang_adaptor(lang_c)  # (B, L, D)
            if self.pre_norm_modality:
                lang_c = self.lang_norm(lang_c, time_c)
            context_c = torch.cat([context_c, lang_c], dim=1)  # (B, L + L_lang, n_emb)

        # 5. transformer blocks
        for block in self.blocks:
            x = block(x, time_c, context_c)  # (B, T, n_emb)

        # 6. head
        x = self.final_layer(x)

        # (B, T, output_dim)
        x = x[:, -self.horizon :]  # (B, T, out_channels)

        return x
