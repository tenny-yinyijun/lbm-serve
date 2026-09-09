import logging
import os

import cv2
import draccus
import numpy as np
import torch

from vla_foundry.data.constants import POINT_MAP_MM_TO_M_SCALE, POINT_MAP_UINT16_OFFSET
from vla_foundry.data.processor import apply_chat_template, get_processor
from vla_foundry.data.robotics.normalization import build_normalizer
from vla_foundry.file_utils import json_load
from vla_foundry.params.base_data_params import DataParams
from vla_foundry.params.data_params import RoboticsDataParams


class RoboticsProcessor:
    """
    This class handles tokenization and normalization of robotics data.
    It also handles image loading and processing.
    """

    def __init__(self, data_params: RoboticsDataParams, pretrained_path: str | None = None):
        self.data_params = data_params
        self.vlm_processor = get_processor(data_params)
        self.processor_kwargs = getattr(data_params, "processor_kwargs", {})

        if pretrained_path is not None:
            self.normalizer = (
                build_normalizer(data_params, pretrained_path=pretrained_path)
                if data_params.normalization.enabled
                else None
            )
        else:
            statistics_entries = [json_load(stats_path) for stats_path in data_params.dataset_statistics]
            if self.data_params.normalization.enabled and statistics_entries:
                self.normalizer = build_normalizer(data_params, statistics_data=statistics_entries)
            else:
                self.normalizer = None

    def save(self, experiment_path: str):
        with open(os.path.join(experiment_path, "config_processor.yaml"), "w") as f:
            draccus.dump(self.data_params, f)

    @classmethod
    def load(cls, config_path: str):
        return cls(DataParams.from_file(config_path))

    @classmethod
    def from_pretrained(cls, config_path: str):
        data_params = DataParams.from_file(os.path.join(config_path, "config_processor.yaml"))
        return cls(data_params, pretrained_path=config_path)

    def denormalize_first_sample_images(self, pixel_values, image_grid_thw=None, batch_size=1):
        """Denormalize pixel_values and return images for the first sample in the batch.

        Intended for visualization/logging during inference. Only processes the
        first sample to avoid unnecessary computation.

        Supports both standard processors ((..., C, H, W) tensors) and Qwen-style
        processors (flat (total_patches, patch_dim) tensors with image_grid_thw metadata).

        Args:
            pixel_values: Tensor of shape (B, N, C, H, W) for standard processors,
                          or (B*N, C, H, W) for processors that flatten the batch and image dims,
                          or (total_patches, patch_dim) for Qwen-style processors.
            image_grid_thw: Optional tensor of shape (num_images, 3) with [grid_t, grid_h, grid_w]
                            per image. Required for Qwen-style denormalization.
            batch_size: Number of samples in the batch. Used to extract the first sample's
                        images from 4D [B*N, C, H, W] pixel_values. Defaults to 1.

        Returns:
            List of (H, W, C) numpy arrays with uint8 values in [0, 255],
            one per image/frame in the first sample.
        """
        image_processor = self.vlm_processor.image_processor
        mean = torch.tensor(image_processor.image_mean, dtype=pixel_values.dtype, device=pixel_values.device)
        std = torch.tensor(image_processor.image_std, dtype=pixel_values.dtype, device=pixel_values.device)

        if image_grid_thw is not None:
            return self._denormalize_qwen_pixel_values(pixel_values, image_grid_thw, mean, std)

        # Standard pixel_values: either [B, N, C, H, W] (5D) or [B*N, C, H, W] (4D).
        # Extract images for the first sample only.
        if pixel_values.ndim == 5:
            imgs = pixel_values[0]  # [N, C, H, W]
        else:
            images_per_sample = pixel_values.shape[0] // batch_size
            imgs = pixel_values[:images_per_sample]  # [N, C, H, W]
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        imgs = (imgs * std + mean).clamp(0, 1).mul(255).byte()
        # (N, C, H, W) -> list of (H, W, C)
        return [img.permute(1, 2, 0).cpu().numpy() for img in imgs]

    def _denormalize_qwen_pixel_values(self, pixel_values, image_grid_thw, mean, std):
        """Reverse Qwen's patch flattening and normalization.

        Inverts the reshape+transpose from Qwen's image processor:
          (grid_t, tp, C, grid_h//ms, ms, ps, grid_w//ms, ms, ps)
          -> transpose(0,3,6,4,7,2,1,5,8)
          -> flatten to (grid_t*grid_h*grid_w, C*tp*ps*ps)
        """
        patch_size = self.vlm_processor.image_processor.patch_size
        temporal_patch_size = self.vlm_processor.image_processor.temporal_patch_size
        merge_size = self.vlm_processor.image_processor.merge_size
        channel = 3

        patches_per_image = (image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).tolist()

        frames = []
        offset = 0
        for i, (grid_t, grid_h, grid_w) in enumerate(image_grid_thw.tolist()):
            grid_t, grid_h, grid_w = int(grid_t), int(grid_h), int(grid_w)
            n_patches = int(patches_per_image[i])
            flat = pixel_values[offset : offset + n_patches]
            offset += n_patches

            # Reverse flatten -> (grid_t, grid_h//ms, grid_w//ms, ms, ms, C, tp, ps, ps)
            patches = flat.reshape(
                grid_t,
                grid_h // merge_size,
                grid_w // merge_size,
                merge_size,
                merge_size,
                channel,
                temporal_patch_size,
                patch_size,
                patch_size,
            )
            # Reverse transpose (0,3,6,4,7,2,1,5,8) -> inverse is (0,6,5,1,3,7,2,4,8)
            patches = patches.permute(0, 6, 5, 1, 3, 7, 2, 4, 8)
            # Now: (grid_t, tp, C, grid_h//ms, ms, ps, grid_w//ms, ms, ps)
            # Reshape to (grid_t * tp, C, grid_h * ps, grid_w * ps)
            patches = patches.reshape(grid_t * temporal_patch_size, channel, grid_h * patch_size, grid_w * patch_size)

            # Denormalize
            m = mean.view(1, 3, 1, 1)
            s = std.view(1, 3, 1, 1)
            img = patches * s + m
            img = img.clamp(0, 1).mul(255).byte()
            # (frames, C, H, W) -> (frames, H, W, C)
            img = img.permute(0, 2, 3, 1).cpu().numpy()
            for f in range(img.shape[0]):
                frames.append(img[f])

        return frames

    def add_action_and_proprioception_fields(self, batch, action_fields=None, proprioception_fields=None):
        # Pre-extract concatenated actions if action fields are provided
        if action_fields:
            action_data = []
            for key in action_fields:
                if key in batch["lowdim"]:
                    action_data.append(batch["lowdim"][key])
                else:
                    raise KeyError(f"Action field '{key}' missing from lowdim data")

            batch["actions"] = torch.cat(action_data, dim=-1)  # [B, T, D]

        if proprioception_fields:
            proprioception_data = []
            for key in proprioception_fields:
                proprioception_data.append(batch["lowdim"][key][:, : self.data_params.lowdim_past_timesteps + 1])
            batch["proprioception"] = torch.cat(proprioception_data, dim=-1)

        return batch

    def _resize_point_map(self, point_map: np.ndarray, target_size: int) -> np.ndarray:
        """
        Resize a point map (H, W, 3) to (target_size, target_size, 3) using scale + center crop.
        This matches the approach used for RGB and depth images to ensure consistent field of view.
        Uses nearest neighbor interpolation to preserve coordinate values without artifacts.

        Args:
            point_map: (H, W, 3) uint16 array with XYZ coordinates
            target_size: Target height and width

        Returns:
            Resized point map (target_size, target_size, 3) uint16 array
        """
        h, w, c = point_map.shape
        if h == target_size and w == target_size:
            return point_map

        # Calculate scale to cover target dimensions (no black bars)
        scale = max(target_size / w, target_size / h)
        new_width = int(w * scale)
        new_height = int(h * scale)

        # Resize each channel separately using nearest neighbor to preserve coordinate integrity
        resized_channels = []
        for channel_idx in range(c):
            channel = point_map[:, :, channel_idx]
            # Use INTER_NEAREST to avoid interpolation artifacts
            resized_channel = cv2.resize(channel, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            resized_channels.append(resized_channel)

        resized_pm = np.stack(resized_channels, axis=-1).astype(np.uint16)

        # Center crop to exact target size
        left = (new_width - target_size) // 2
        top = (new_height - target_size) // 2
        right = left + target_size
        bottom = top + target_size
        cropped_pm = resized_pm[top:bottom, left:right, :]

        return cropped_pm

    def process_inputs(self, batch, image_names, max_text_seq_len=None):
        """Tokenizes the text and converts the image to pixel_values
        Args:
            batch: Batch of samples to convert to tensors.
            image_names: Automatically generated from camera_names and image_indices in the data_params.
        """
        batch_text, batch_images, batch_attention_mask_images = [], [], []
        for sample_images, instruction in zip(batch["images"], batch["language_instruction"], strict=False):
            if image_names is None or len(image_names) == 0:
                image_names = list(sample_images.keys())
                logging.warning(
                    "WARNING: Using sample_images.keys() to detect camera names. No guarantee of consistent ordering."
                    f"Sample keys: {list(sample_images.keys())}"
                )
            if self.data_params.pad_missing_images:
                # Zero-pad missing camera images and create mask to mask out later on.
                sample_images = [sample_images.get(k, None) for k in image_names]
                zero_image_size = None
                for i in sample_images:
                    if i is not None:
                        zero_image_size = i.shape
                        break
                if self.data_params.mask_padded_images:
                    attention_mask_images = [1 if i is not None else 0 for i in sample_images]
                else:
                    # LBM1.0 does not mask padded images
                    attention_mask_images = [1 for i in sample_images]
                sample_images = [i if i is not None else np.zeros(zero_image_size) for i in sample_images]
            else:
                sample_images = [sample_images[k] for k in image_names if k in sample_images]
                attention_mask_images = [1 for i in sample_images]

            instruction = apply_chat_template(self.vlm_processor, len(sample_images), instruction)

            batch_text.append(instruction)
            if len(sample_images) > 0:
                batch_images.append(sample_images)
                batch_attention_mask_images.append(attention_mask_images)

        # If no images, set batch_images to None
        if len(batch_images) == 0:
            batch_images = None
            batch_attention_mask_images = None
        else:
            image_counts = [len(imgs) for imgs in batch_images]
            assert len(set(image_counts)) == 1, (
                f"All samples must have the same number of images, got {image_counts}. "
                f"Set data_params.pad_missing_images=True to pad missing camera images."
            )
            batch_attention_mask_images = torch.tensor(batch_attention_mask_images, dtype=torch.bool)  # [B, num_images]

        # Run processor on entire batch — start from its output so all VLM-specific
        # keys (pixel_values, input_ids, attention_mask, image_grid_thw, etc.) are
        # automatically carried forward without explicit per-key copying.
        processed_batch = self.vlm_processor(
            images=batch_images,
            text=batch_text,
            padding=True,
            truncation=max_text_seq_len is not None,
            max_length=max_text_seq_len,
            return_tensors="pt",
            **self.processor_kwargs,
        )

        # Copy over non-VLM fields from the original batch (past_mask, future_mask,
        # metadata, language_instruction, intrinsics, extrinsics, etc.)
        for key, value in batch.items():
            if key not in processed_batch:
                processed_batch[key] = value

        processed_batch["attention_mask_images"] = batch_attention_mask_images
        processed_batch["camera_names"] = self.data_params.camera_names
        processed_batch["images"] = batch_images
        processed_batch["lowdim"] = {}
        for k in batch["lowdim"][0]:
            if isinstance(batch["lowdim"][0][k][0], str):
                continue
            values = [sample_lowdim[k] for sample_lowdim in batch["lowdim"]]
            processed_batch["lowdim"][k] = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in values])

        if self.data_params.use_point_cloud:
            # Point clouds are FPS-sampled either:
            # - During preprocessing (training): pre-generated in tar files
            # - During inference: generated from depth images in PolicyDataAdapter
            point_cloud_list = [sample_pc for sample_pc in batch["point_cloud"]]
            processed_batch["point_cloud"] = torch.stack(
                [torch.as_tensor(pc, dtype=torch.float32) for pc in point_cloud_list]
            )

            # Apply CLIP normalization to RGB channels (channels 3-5) for consistency with images
            # This is applied in both training and inference
            if processed_batch["point_cloud"].shape[-1] == 6:  # Only if RGB channels exist
                pc = processed_batch["point_cloud"]  # (B, T, N, 6)
                xyz = pc[..., :3]  # XYZ coordinates
                rgb = pc[..., 3:]  # RGB colors in [0, 1]

                # Apply CLIP normalization to RGB
                # CLIP normalization: mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=rgb.device)
                clip_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=rgb.device)
                rgb = (rgb - clip_mean) / clip_std

                # Concatenate normalized RGB with XYZ
                processed_batch["point_cloud"] = torch.cat([xyz, rgb], dim=-1)
        else:
            processed_batch["point_cloud"] = None

        # Normalize each field individually
        if self.normalizer and self.data_params.normalization.enabled:
            anchor_timestep = self.data_params.lowdim_past_timesteps
            # Normalize each lowdim field
            for field_name, tensor in processed_batch["lowdim"].items():
                if isinstance(tensor, torch.Tensor) and field_name in self.normalizer.include_fields:
                    processed_batch["lowdim"][field_name] = self.normalizer.normalize_tensor(
                        tensor, field_name, anchor_timestep=anchor_timestep
                    )

            # Normalize point cloud if enabled
            # Note: RGB is already normalized with CLIP normalization above (line 168-182)
            # The normalizer will handle 6-channel (XYZRGB) tensors by only normalizing XYZ
            if processed_batch["point_cloud"] is not None and "point_cloud" in self.normalizer.include_fields:
                processed_batch["point_cloud"] = self.normalizer.normalize_tensor(
                    processed_batch["point_cloud"], "point_cloud", anchor_timestep=anchor_timestep
                )

        # Process point maps
        if self.data_params.use_point_cloud and batch.get("point_maps") is not None:
            # Stack point maps from all samples in batch
            # Each sample has dict: {camera_t_offset: (H, W, 3) uint16}
            batch_point_maps = []
            for sample_pms in batch["point_maps"]:
                if sample_pms is None:
                    continue
                # Convert to list ordered by camera_names and image_indices
                pm_list = []
                for camera_name in self.data_params.camera_names:
                    for img_offset in self.data_params.image_indices:
                        key = f"{camera_name}_t{img_offset}"
                        if key not in sample_pms:
                            continue
                        pm = sample_pms[key]
                        # Resize point map to match image size
                        pm_resized = self._resize_point_map(pm, self.data_params.image_size)
                        # Convert uint16 (offset) to float32 meters
                        # Subtract offset to get signed mm values, then convert to meters
                        pm_float = (pm_resized.astype(np.float32) - POINT_MAP_UINT16_OFFSET) / POINT_MAP_MM_TO_M_SCALE
                        pm_list.append(pm_float)

                # Stack into (T, H, W, 3) where T = num_cameras * len(image_indices)
                if pm_list:
                    batch_point_maps.append(np.stack(pm_list, axis=0))

            if batch_point_maps:
                processed_batch["point_maps"] = torch.stack(
                    [torch.as_tensor(pm, dtype=torch.float32) for pm in batch_point_maps]
                )
                # Shape: (B, T, H, W, 3) in meters

                # Normalize point maps using dataset statistics
                if self.normalizer and "point_maps" in self.normalizer.include_fields:
                    pm = processed_batch["point_maps"]
                    B, T, H, W, C = pm.shape
                    # Flatten spatial dimensions for normalization
                    pm_flat = pm.reshape(B, T, H * W, C)

                    # Apply normalization (uses min/max from stats)
                    pm_normalized = self.normalizer.normalize_tensor(pm_flat, "point_maps", anchor_timestep=None)

                    # Reshape back
                    processed_batch["point_maps"] = pm_normalized.reshape(B, T, H, W, C)
            else:
                processed_batch["point_maps"] = None
        else:
            processed_batch["point_maps"] = None

        return processed_batch
