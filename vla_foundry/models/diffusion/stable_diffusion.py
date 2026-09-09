import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTokenizer

from vla_foundry.models.base_model import BaseModel
from vla_foundry.models.diffusion.noise_scheduler import NoiseScheduler
from vla_foundry.models.diffusion.noise_scheduler_diffusers import FlowMatchingScheduler, NoiseSchedulerDDPMDiffusers
from vla_foundry.models.diffusion.unet import UNet
from vla_foundry.models.diffusion_policy.clip_hf import CLIPHF
from vla_foundry.params.model_params import StableDiffusionParams


class StableDiffusion(BaseModel):
    def __init__(self, model_params: StableDiffusionParams, clip: CLIPHF, unet: UNet, scheduler: NoiseScheduler):
        super().__init__(model_params)
        self.clip = clip
        self.scheduler = scheduler
        self.unet = unet
        if self.model_params.use_flow_matching_scheduler and not isinstance(self.scheduler, FlowMatchingScheduler):
            raise ValueError("use_flow_matching_scheduler=True requires a FlowMatchingScheduler")
        # Continuous time (t in [0, 1]) is an opt-in mode of flow matching; the default uses
        # discrete integer timesteps scaled by num_timesteps (no breaking change).
        self.continuous_time = model_params.noise_scheduler.continuous_time
        if self.continuous_time and not self.model_params.use_flow_matching_scheduler:
            raise ValueError("noise_scheduler.continuous_time=True requires use_flow_matching_scheduler=True")
        if self.clip is not None:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip.model_name)
        else:
            self.clip_tokenizer = None

    @property
    def _is_flow_matching(self):
        return isinstance(self.scheduler, FlowMatchingScheduler)

    def forward(self, input_ids, image, attention_mask, noise):
        batch_size = image.shape[0]
        # Sample random timesteps
        if self.continuous_time:
            timesteps = torch.rand((batch_size,), device=image.device)  # [bsz], continuous in [0, 1)
        else:
            timesteps = torch.randint(0, self.scheduler.num_timesteps, (batch_size,), device=image.device)  # [bsz]

        if self.clip is not None:
            if self.model_params.do_classifier_free_guidance:
                unconditional_mask = torch.rand(batch_size, device=image.device) < self.model_params.dropout_percent
                uncond_tokens_output = self.clip_tokenizer(
                    [""] * batch_size,
                    padding="max_length",
                    max_length=input_ids.shape[1],
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_tokens = uncond_tokens_output.input_ids.to(image.device)
                uncond_attention_mask = uncond_tokens_output.attention_mask.to(image.device)
                mixed_input_ids = torch.where(
                    unconditional_mask.unsqueeze(1), uncond_tokens, input_ids.to(image.device)
                )
                mixed_attention_mask = torch.where(
                    unconditional_mask.unsqueeze(1), uncond_attention_mask, attention_mask.to(image.device)
                )
            else:
                mixed_input_ids = input_ids.to(image.device)
                mixed_attention_mask = attention_mask.to(image.device)
            text_embeddings = self.clip.model.text_model(
                mixed_input_ids, attention_mask=mixed_attention_mask
            ).last_hidden_state
            noisy_images = self.scheduler.add_noise(image, noise, timesteps)
            predicted_noise = self.unet(noisy_images, timesteps, text_embeddings)
        else:
            noisy_images = self.scheduler.add_noise(image, noise, timesteps)  # [bsz, channels, h, w]
            predicted_noise = self.unet(noisy_images, timesteps)  # [bsz, channels, h, w]

        return predicted_noise

    @torch.no_grad()
    def generate(
        self, batch_size, device, prompt=None, uncond_prompt="", max_text_seq_len=64, num_inference_steps=None
    ):
        # Initialize random latents
        images = torch.randn(
            batch_size, self.unet.in_channels, self.model_params.image_size, self.model_params.image_size
        ).to(device)

        # Prepare text embeddings for generation
        if prompt is not None:
            cond_tokens = self.clip_tokenizer(
                [prompt] * batch_size,
                padding="max_length",
                max_length=max_text_seq_len,
                truncation=True,
                return_tensors="pt",
            )
            cond_embeddings = self.clip.model.text_model(cond_tokens.input_ids.to(device)).last_hidden_state

            if self.model_params.do_classifier_free_guidance:
                uncond_tokens = self.clip_tokenizer(
                    [uncond_prompt] * batch_size,
                    padding="max_length",
                    max_length=max_text_seq_len,
                    truncation=True,
                    return_tensors="pt",
                )
                uncond_embeddings = self.clip.model.text_model(uncond_tokens.input_ids.to(device)).last_hidden_state
                text_embeddings = torch.cat([uncond_embeddings, cond_embeddings], dim=0)
            else:
                text_embeddings = cond_embeddings
        else:
            text_embeddings = None

        if self.continuous_time:
            if num_inference_steps is None:
                raise ValueError("num_inference_steps is required when noise_scheduler.continuous_time=True")
            # Denoising loop with continuous timesteps from 1 to 0
            step_list = torch.linspace(0, 1, num_inference_steps + 1, device=device).flip(dims=[0])[:-1]
            for i, step in enumerate(tqdm(step_list)):
                timestep_tensor = step.repeat(batch_size)

                model_input = images
                if self.model_params.do_classifier_free_guidance and text_embeddings is not None:
                    model_input = model_input.repeat(2, 1, 1, 1)
                    timestep_tensor = timestep_tensor.repeat(2)

                if text_embeddings is not None:
                    model_output = self.unet(model_input, timestep_tensor, text_embeddings)
                else:
                    model_output = self.unet(model_input, timestep_tensor)

                if self.model_params.do_classifier_free_guidance and text_embeddings is not None:
                    output_uncond, output_cond = model_output.chunk(2)
                    model_output = output_uncond + self.model_params.guidance_scale * (output_cond - output_uncond)

                step_size = step_list[i] - step_list[i + 1] if i < len(step_list) - 1 else step_list[i]
                # Flow matching does not use a discrete timestep in step(), so pass the state explicitly.
                images = self.scheduler.step(model_output, sample=images, step_size=step_size)
        else:
            # Discrete denoising loop with integer timesteps (DDPM and discrete-time flow matching).
            # FlowMatchingScheduler.step advances by 1/num_timesteps per single step here.
            if isinstance(self.scheduler, NoiseSchedulerDDPMDiffusers):
                self.scheduler.scheduler.set_timesteps(self.scheduler.num_timesteps)
            for t in tqdm(range(self.scheduler.num_timesteps - 1, 0, -1)):
                timestep_tensor = torch.tensor([t] * batch_size, device=device, dtype=torch.long)

                model_input = images
                if self.model_params.do_classifier_free_guidance and text_embeddings is not None:
                    model_input = model_input.repeat(2, 1, 1, 1)
                    timestep_tensor = timestep_tensor.repeat(2)

                if text_embeddings is not None:
                    model_output = self.unet(model_input, timestep_tensor, text_embeddings)
                else:
                    model_output = self.unet(model_input, timestep_tensor)

                if self.model_params.do_classifier_free_guidance and text_embeddings is not None:
                    output_uncond, output_cond = model_output.chunk(2)
                    model_output = output_uncond + self.model_params.guidance_scale * (output_cond - output_uncond)

                images = self.scheduler.step(model_output, t, images)

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = [Image.fromarray(image) for image in images]
        return images

    def get_fsdp_block_types(self):
        """Return block types for FSDP wrapping."""
        if self.model_params.use_diffusers_unet:
            from diffusers.models.unets.unet_2d_blocks import (
                AttnDownBlock2D,
                AttnUpBlock2D,
                DownBlock2D,
                UNetMidBlock2D,
                UpBlock2D,
            )

            return (DownBlock2D, UpBlock2D, UNetMidBlock2D, AttnUpBlock2D, AttnDownBlock2D)
        else:
            from vla_foundry.models.diffusion.unet import CrossAttentionBlock, ResnetBlock, SelfAttentionBlock

            return (ResnetBlock, SelfAttentionBlock, CrossAttentionBlock)
