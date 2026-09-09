"""
Normalization utilities for robotics data.

``RoboticsNormalizer`` handles WebDataset statistics, which are keyed by field
name. ``LakehouseNormalizer`` handles the embodiment-keyed format data-lakehouse
writes::

    {"global":          {field: {mean, std, min, max, count, percentile_*}},
     "<embodiment_id>": {field: {...}},
     "counts":          {"global"|"<embodiment_id>": {episode_count, sample_count}}}

Training reads the ``global`` section; the other sections ride along into the
checkpoint's ``stats.json`` so per-embodiment normalization can be added later
without retraining. Callers pick neither class directly - ``build_normalizer``
selects one from the data params.
"""

import json
import logging
import os
from typing import Any, ClassVar

import draccus
import torch

from vla_foundry.data.robotics.utils import (
    LAKEHOUSE_STATS_COUNTS_KEY,
    LAKEHOUSE_STATS_GLOBAL_KEY,
    StatisticsMergeWeights,
    crop_sequence,
    merge_nested_lakehouse_statistics,
    merge_statistics,
)
from vla_foundry.file_utils import json_load
from vla_foundry.params.data_params import LakehouseDataParams, LakehouseStreamParams, RoboticsDataParams
from vla_foundry.params.robotics.normalization_params import (
    FieldNormalizationParams,
    LakehouseNormalizationParams,
    NormalizationParams,
)


def _crop_estimator_state(state: dict, start: int, end: int) -> dict:
    """Crop a tdigest/psquared per-timestep state dict along the first (timestep) axis.

    These states are stored sparsely: ``indices`` holds ``[timestep, channel, ...]``
    coordinates and the data lists are parallel to it. Cropping therefore means
    dropping the entries outside ``[start, end)`` and rebasing the timestep index,
    not slicing an array.

    Args:
        state: A ``TDigestEstimator.get_state()`` / ``PSquaredEstimator.get_state()`` dict.
        start: First timestep to keep.
        end: One past the last timestep to keep.

    Returns:
        A new state dict covering ``end - start`` timesteps.
    """
    new_shape = list(state["shape"])
    new_shape[0] = end - start

    def _filter_sparse(indices_list, data_lists, key_names):
        new_indices = []
        new_data = {k: [] for k in key_names}
        for j, idx in enumerate(indices_list):
            timestep = idx[0]
            if start <= timestep < end:
                new_indices.append([timestep - start] + idx[1:])
                for k in key_names:
                    new_data[k].append(data_lists[k][j])
        return new_indices, new_data

    cropped = {
        "shape": new_shape,
        "counts": state["counts"][start:end],
        "max_buffer": state.get("max_buffer", 1000),
        "compression": state.get("compression", 100),
    }

    if "digests" in state:
        digests = state["digests"]
        new_idx, new_vals = _filter_sparse(
            digests["indices"], {"means": digests["means"], "weights": digests["weights"]}, ["means", "weights"]
        )
        cropped["digests"] = {"indices": new_idx, "means": new_vals["means"], "weights": new_vals["weights"]}

    if "buffers" in state:
        buffers = state["buffers"]
        new_idx, new_vals = _filter_sparse(buffers.get("indices", []), {"data": buffers.get("data", [])}, ["data"])
        cropped["buffers"] = {"indices": new_idx, "data": new_vals["data"]}

    return cropped


def _align_stats_to_common_window(
    stats_list: list[dict[str, Any]],
    per_source_past: list[int],
    per_source_future: list[int],
    target_past: int,
    target_future: int,
) -> list[dict[str, Any]]:
    """Crop each source's per-timestep stats to a common window before merging.

    Each source's per-timestep arrays have shape ``(source_past + 1 + source_future, ...)``
    with the anchor at index ``source_past``. This crops them all to
    ``(target_past + 1 + target_future, ...)`` with the anchor at ``target_past``, so
    ``merge_statistics`` combines matching timesteps. Without it, sources preprocessed
    with different windows cannot be merged at all - ``merge_statistics`` raises on the
    ragged shapes.

    Args:
        stats_list: One field-keyed statistics dict per source, in source order.
        per_source_past: Each source's ``past_lowdim_steps``, same order.
        per_source_future: Each source's ``future_lowdim_steps``, same order.
        target_past: Anchor index of the common window (the min of ``per_source_past``).
        target_future: Future length of the common window.

    Returns:
        The statistics dicts, each covering the common window.
    """
    aligned = []
    for i, source_stats in enumerate(stats_list):
        src_past = per_source_past[i]
        src_future = per_source_future[i]
        if src_past == target_past and src_future == target_future:
            aligned.append(source_stats)
            continue

        # The anchor sits at src_past; keep target_past before it and target_future after.
        start = src_past - target_past
        end = src_past + 1 + target_future
        cropped = {}
        for field_name, field_stats in source_stats.items():
            cropped[field_name] = {}
            for stat_name, stat_value in field_stats.items():
                # `count` and `percentile_sample_count` are per-timestep despite the name.
                is_per_timestep = stat_name.endswith("_per_timestep") or stat_name in (
                    "count",
                    "percentile_sample_count",
                )
                if is_per_timestep and isinstance(stat_value, list | tuple):
                    cropped[field_name][stat_name] = stat_value[start:end]
                elif is_per_timestep and isinstance(stat_value, dict) and "shape" in stat_value:
                    # tdigest/psquared states are sparse dicts, not sliceable sequences.
                    cropped[field_name][stat_name] = _crop_estimator_state(stat_value, start, end)
                else:
                    # Anything else - whole-sequence stats, or a None percentile such as
                    # point_cloud's - is window-independent and passes through untouched.
                    cropped[field_name][stat_name] = stat_value
        aligned.append(cropped)
    return aligned


class RoboticsNormalizer:
    """
    Normalizer for robotics data with configurable strategies.

    Supports:
    - Global normalization: normalize across all timesteps
    - Per-timestep normalization: normalize each timestep separately
    - Std-based normalization: use mean/std
    - Quantile-based normalization: use percentiles (e.g., 5th/95th)

    Handles WebDataset statistics, which are keyed by field name at the top
    level; ``LakehouseNormalizer`` below serves the embodiment-keyed format.
    """

    # Params class ``load`` / ``from_pretrained`` decode the saved normalizer config
    # into; subclasses point this at their own params so extra knobs survive a
    # checkpoint round-trip.
    normalization_params_cls: ClassVar[type[NormalizationParams]] = NormalizationParams

    def __init__(
        self,
        normalization_params: dict[str, Any] | NormalizationParams,
        statistics_data: dict[str, Any] | None = None,
        statistics_path: str | list[str] | None = None,
    ):
        """
        Initialize normalizer.

        Args:
            normalization_params: NormalizationParams instance with field definitions and normalization settings
            statistics_data: Pre-loaded statistics dict
            statistics_path: Path to statistics JSON file
        """
        self.normalization_params = normalization_params

        self.enabled = self.normalization_params.enabled
        self.lowdim_past_timesteps = self.normalization_params.lowdim_past_timesteps
        self.lowdim_future_timesteps = self.normalization_params.lowdim_future_timesteps

        # Always load statistics when available, regardless of whether normalization is enabled
        # This allows action dimension computation even when normalization is disabled
        if statistics_data is not None:
            self.stats = statistics_data
        elif statistics_path is not None:
            self.stats = self._load_statistics(statistics_path)
        else:
            logging.warning("No statistics provided - normalization will be disabled")
            self.enabled = False
            self.stats = None
            return

        self._norm_param_cache = {}
        if isinstance(self.stats, list):
            self.stats = self._align_statistics_list(self.stats)
            if len(self.stats) > 1:
                self.stats = self._merge_statistics_list(self.stats)
            else:
                self.stats = self.stats[0]

        # Parse configuration from dataclass
        self.method = self.normalization_params.method
        self.scope = self.normalization_params.scope
        self.epsilon = self.normalization_params.epsilon
        self.field_configs = self.normalization_params.field_configs
        self.include_fields = self.normalization_params.include_fields
        self.centered_norm = self.normalization_params.centered_norm

        logging.info(f"RoboticsNormalizer initialized: method={self.method}, scope={self.scope}")

    def _align_statistics_list(self, statistics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Crop every source's per-timestep statistics to the common lowdim window.

        Sources preprocessed with different ``past/future_lowdim_steps`` have their
        anchor at different indices, so merging them elementwise would combine
        different timesteps. ``resolve_normalization_timesteps`` records each source's
        window on the params; this crops them all to the common one first.

        Args:
            statistics: Loaded per-dataset statistics dicts, in dataset order.

        Returns:
            The statistics, cropped to the common window. Returned unchanged when the
            per-source windows are unknown (nothing to align against).

        Raises:
            ValueError: If only one of the two per-source window lists is usable, which
                means the params were not resolved as a pair and cropping would read
                past the end of the missing one.
        """
        per_source_past = getattr(self.normalization_params, "per_source_lowdim_past", [])
        per_source_future = getattr(self.normalization_params, "per_source_lowdim_future", [])
        has_past = bool(per_source_past) and len(per_source_past) == len(statistics)
        has_future = bool(per_source_future) and len(per_source_future) == len(statistics)
        if has_past != has_future:
            raise ValueError(
                "Inconsistent per-source lowdim window metadata on NormalizationParams: "
                f"per_source_lowdim_past={per_source_past}, per_source_lowdim_future={per_source_future}, "
                f"len(stats)={len(statistics)}. Both lists must be present and match the number of "
                "statistics sources. Re-run resolve_derived_fields to regenerate them."
            )
        if not (has_past and has_future):
            return statistics
        return _align_stats_to_common_window(
            statistics,
            per_source_past,
            per_source_future,
            self.lowdim_past_timesteps,
            self.lowdim_future_timesteps,
        )

    def _merge_statistics_list(self, statistics: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge one statistics dict per dataset into a single one.

        Args:
            statistics: Loaded per-dataset statistics dicts, in dataset order.

        Returns:
            The merged statistics, pooled by per-field sample counts.
        """
        return merge_statistics(statistics)

    def field_statistics(self, field_name: str, embodiment_id: str | None = None) -> dict[str, Any] | None:
        """Return the loaded statistics for one field.

        The single seam every field-keyed read of ``self.stats`` goes through, so
        subclasses can serve a different stats layout (see ``LakehouseNormalizer``).

        Args:
            field_name: Canonical field name to look up.
            embodiment_id: Not supported here - WebDataset statistics pool every
                embodiment into one set of fields.

        Returns:
            The field's stats dict, or ``None`` when the field (or the whole
            statistics file) is absent.

        Raises:
            ValueError: If an embodiment is requested.
        """
        if embodiment_id is not None:
            raise ValueError(
                f"WebDataset statistics have no per-embodiment sections; cannot read {field_name!r} for "
                f"embodiment {embodiment_id!r}."
            )
        return self.stats.get(field_name) if self.stats else None

    def save(self, experiment_path: str):
        with open(os.path.join(experiment_path, "config_normalizer.yaml"), "w") as f:
            draccus.dump(self.normalization_params, f)
        with open(os.path.join(experiment_path, "stats.json"), "w") as f:
            json.dump(self.stats, f)

    @classmethod
    def load(cls, config_path: str, statistics_path: str):
        return cls(cls.normalization_params_cls.from_file(config_path), statistics_path=statistics_path)

    @classmethod
    def from_pretrained(cls, config_path: str):
        return cls(
            cls.normalization_params_cls.from_file(os.path.join(config_path, "config_normalizer.yaml")),
            statistics_path=os.path.join(config_path, "stats.json"),
        )

    def get_field_dimension(self, field_name: str) -> int:
        """Get the dimension of a field."""
        field_stats = self.field_statistics(field_name)
        if field_stats is None:
            raise ValueError(f"Field {field_name} not found in dataset statistics")
        return len(field_stats["mean"])

    def _load_statistics(self, statistics_path: str) -> dict[str, Any]:
        """Load statistics from JSON file."""
        if isinstance(statistics_path, str):
            stats = json_load(statistics_path)
        elif isinstance(statistics_path, list):
            stats = []
            for path in statistics_path:
                stats.append(json_load(path))
        else:
            raise ValueError(f"Invalid statistics path: {statistics_path}")
        logging.info(f"Loaded statistics from {statistics_path}")
        return stats

    def _get_field_config(self, field_name: str) -> FieldNormalizationParams:
        """Get configuration for a specific field."""
        # Check for exact match first
        if field_name in self.field_configs:
            return self.field_configs[field_name]

        # Check for pattern matches (mainly applies the same config to relative fields as the main field)
        for pattern, config in self.field_configs.items():
            if pattern in field_name:
                return config

        # Return default config
        return FieldNormalizationParams(
            method=self.method, scope=self.scope, epsilon=self.epsilon, enabled=self.enabled
        )

    def _should_normalize_field(self, field_name: str) -> bool:
        """Check if a field should be normalized."""
        if not self.enabled or not self._get_field_config(field_name).enabled:
            return False

        # Only normalize fields that are in the include_fields
        if field_name not in self.include_fields:
            return False

        # Skip text and mask fields
        if any(keyword in field_name.lower() for keyword in ["text", "language", "instruction", "mask", "valid"]):
            return False

        # Skip if no statistics available
        if self.field_statistics(field_name) is None:
            logging.warning(f"No statistics available for field: {field_name}")
            return False

        return True

    def _get_normalization_params(self, field_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get normalization parameters (center, scale) for a field.
        Use a cache for efficiency.

        Args:
            field_name: Name of the field

        Returns:
            Tuple of (center, scale) tensors
        """
        if field_name in self._norm_param_cache:
            return self._norm_param_cache[field_name]
        center, scale = self._compute_normalization_params(field_name)
        self._norm_param_cache[field_name] = (center, scale)
        return center, scale

    def _compute_normalization_params(self, field_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        field_stats = self.field_statistics(field_name)
        field_config = self._get_field_config(field_name)

        method = field_config.method
        scope = field_config.scope
        epsilon = field_config.epsilon

        if not self.enabled or not field_config.enabled:
            return torch.zeros(1), torch.ones(1)

        if scope == "global":
            if method == "std":
                center = torch.tensor(field_stats["mean"], dtype=torch.float32)
                scale = torch.tensor(field_stats["std"], dtype=torch.float32)
            elif method == "percentile_5_95":
                center = torch.tensor(field_stats["percentile_5"], dtype=torch.float32)
                scale = torch.tensor(field_stats["percentile_95"], dtype=torch.float32) - torch.tensor(
                    field_stats["percentile_5"], dtype=torch.float32
                )
            elif method == "percentile_1_99":
                center = torch.tensor(field_stats["percentile_1"], dtype=torch.float32)
                scale = torch.tensor(field_stats["percentile_99"], dtype=torch.float32) - torch.tensor(
                    field_stats["percentile_1"], dtype=torch.float32
                )
            elif method == "min_max":
                center = torch.tensor(field_stats["min"], dtype=torch.float32)
                scale = torch.tensor(field_stats["max"], dtype=torch.float32) - torch.tensor(
                    field_stats["min"], dtype=torch.float32
                )
            else:
                raise ValueError(f"Invalid normalization method: {method}")
        else:
            if method == "std":
                center = torch.tensor(field_stats["mean_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(field_stats["std_per_timestep"], dtype=torch.float32)
            elif method == "percentile_5_95":
                center = torch.tensor(field_stats["percentile_5_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(
                    field_stats["percentile_95_per_timestep"],
                    dtype=torch.float32,
                ) - torch.tensor(
                    field_stats["percentile_5_per_timestep"],
                    dtype=torch.float32,
                )
            elif method == "percentile_1_99":
                center = torch.tensor(field_stats["percentile_1_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(
                    field_stats["percentile_99_per_timestep"],
                    dtype=torch.float32,
                ) - torch.tensor(
                    field_stats["percentile_1_per_timestep"],
                    dtype=torch.float32,
                )
            elif method == "min_max":
                center = torch.tensor(field_stats["min_per_timestep"], dtype=torch.float32)
                scale = torch.tensor(field_stats["max_per_timestep"], dtype=torch.float32) - torch.tensor(
                    field_stats["min_per_timestep"], dtype=torch.float32
                )
            else:
                raise ValueError(f"Invalid normalization method: {method}")

        if self.centered_norm and ("percentile" in method or "min_max" in method):
            center = center + 0.5 * scale
            scale = scale * 0.5

        # Avoid division by zero
        scale = torch.clamp(scale, min=epsilon)
        return center, scale

    def normalize_tensor(
        self, tensor: torch.Tensor, field_name: str, anchor_timestep: int | None = None
    ) -> torch.Tensor:
        """
        Normalize a tensor.

        Args:
            tensor: Input tensor of shape [batch_size, timesteps, features] or [batch_size, features]
            field_name: Name of the field being normalized
            anchor_timestep: The index of the anchor timestep in the input tensor
            (only usedfor per-timestep normalization with cropped sequences)

        Returns:
            Normalized tensor
        """
        if not self._should_normalize_field(field_name):
            return tensor

        field_config = self._get_field_config(field_name)
        scope = field_config.scope

        center, scale = self._get_normalization_params(field_name)
        center = center.to(tensor.device)
        scale = scale.to(tensor.device)

        # For point_cloud with 6 channels (XYZ + RGB), only normalize XYZ
        # RGB channels (last 3) are already CLIP-normalized and should not be normalized again
        is_point_cloud_6ch = field_name == "point_cloud" and tensor.shape[-1] == 6
        if is_point_cloud_6ch:
            xyz = tensor[..., :3]  # Extract XYZ channels
            rgb = tensor[..., 3:]  # Extract RGB channels (already CLIP-normalized)
            center = center[:3]  # Use only XYZ statistics
            scale = scale[:3]
            tensor = xyz  # Normalize only XYZ, will concatenate RGB back at the end

        if scope == "global" or len(tensor.shape) == 2:
            # Global normalization or no time dimension
            # Broadcast to match tensor dimensions - add singleton dims for all but last
            target_shape = [1] * (len(tensor.shape) - 1) + [-1]
            center = center.view(target_shape)
            scale = scale.view(target_shape)
            normalized = (tensor - center) / scale

        elif scope == "per_timestep" and len(tensor.shape) == 3:
            # Per-timestep normalization
            # If anchor_timestep is provided, we need to align the tensor with the statistics
            # The statistics were computed with lowdim_past_timesteps past steps
            # The tensor has anchor_timestep as the index of the current timestep
            _batch_size, num_timesteps, _feature_dim = tensor.shape

            if anchor_timestep is not None and num_timesteps != center.shape[0]:
                if self.lowdim_past_timesteps is None:
                    raise ValueError(
                        "The normalizer is asked to normalize a tensor with a different number"
                        f"of timesteps {num_timesteps} than the statistics ({center.shape[0]})"
                        "but lowdim_past_timesteps must be set to align the statistics with the tensor."
                        "This is likely because the data preprocessing metadata is not available."
                    )
                # The statistics were computed with lowdim_past_timesteps past steps
                # We need to crop them to align with the tensor's anchor_timestep
                # This maps: tensor[anchor_timestep] -> stats[lowdim_past_timesteps]

                # Calculate how many past and future timesteps the tensor has relative to its anchor
                tensor_past = anchor_timestep
                tensor_future = num_timesteps - anchor_timestep - 1

                # Crop statistics around stats_anchor_idx to match tensor's time range
                stats_anchor_idx = self.lowdim_past_timesteps

                # Check if crop would be valid
                start_idx = stats_anchor_idx - tensor_past
                end_idx = stats_anchor_idx + tensor_future + 1

                if start_idx >= 0 and end_idx <= len(center):
                    # Normal case: statistics fully cover the tensor's time range
                    cropped_center = crop_sequence(center, stats_anchor_idx, tensor_past, tensor_future)
                    cropped_scale = crop_sequence(scale, stats_anchor_idx, tensor_past, tensor_future)
                else:
                    raise ValueError("The statistics do not cover the requested tensor's time range")

                # Add batch dimension and broadcast
                cropped_center = cropped_center.unsqueeze(0)  # [1, T', D]
                cropped_scale = cropped_scale.unsqueeze(0)  # [1, T', D]

                normalized = (tensor - cropped_center) / cropped_scale
            else:
                # No anchor provided, assume the tensor is already aligned
                center = center.unsqueeze(0)  # [1, T, D]
                scale = scale.unsqueeze(0)  # [1, T, D]
                normalized = (tensor - center) / scale
        else:
            # Unsupported tensor shape
            logging.warning(f"Unsupported tensor shape for normalization: {tensor.shape}")
            normalized = tensor

        # For 6-channel point clouds, concatenate RGB back (RGB was not normalized)
        if is_point_cloud_6ch:
            normalized = torch.cat([normalized, rgb], dim=-1)

        return normalized

    def denormalize_tensor(
        self, normalized_tensor: torch.Tensor, field_name: str, anchor_timestep: int = None
    ) -> torch.Tensor:
        """
        Denormalize a tensor (inverse of normalize_tensor).

        Args:
            normalized_tensor: Normalized tensor
            field_name: Name of the field being denormalized
            anchor_timestep: The index of the anchor timestep in the input tensor
                (only used for per-timestep denormalization with cropped sequences)

        Returns:
            Denormalized tensor
        """
        if not self._should_normalize_field(field_name):
            return normalized_tensor

        field_config = self._get_field_config(field_name)
        scope = field_config.scope

        center, scale = self._get_normalization_params(field_name)
        center = center.to(normalized_tensor.device)
        scale = scale.to(normalized_tensor.device)

        # For point_cloud with 6 channels (XYZ + RGB), only denormalize XYZ
        # RGB channels (last 3) are already CLIP-normalized and should not be denormalized
        is_point_cloud_6ch = field_name == "point_cloud" and normalized_tensor.shape[-1] == 6
        if is_point_cloud_6ch:
            xyz_normalized = normalized_tensor[..., :3]  # Extract normalized XYZ channels
            rgb = normalized_tensor[..., 3:]  # Extract RGB channels (already CLIP-normalized)
            center = center[:3]  # Use only XYZ statistics
            scale = scale[:3]

            # Denormalize only XYZ
            if scope == "global" or len(normalized_tensor.shape) == 2:
                # Global denormalization or no time dimension
                target_shape = [1] * (len(xyz_normalized.shape) - 1) + [-1]
                center = center.view(target_shape)
                scale = scale.view(target_shape)
                xyz_denormalized = xyz_normalized * scale + center
                # Concatenate denormalized XYZ with unchanged RGB and return early
                return torch.cat([xyz_denormalized, rgb], dim=-1)

            # For per-timestep, continue with the regular flow using xyz_normalized only, then concat rgb at the end
            normalized_tensor = xyz_normalized

        if scope == "global" or len(normalized_tensor.shape) == 2:
            # Global denormalization or no time dimension
            # Broadcast to match tensor dimensions - add singleton dims for all but last
            target_shape = [1] * (len(normalized_tensor.shape) - 1) + [-1]
            center = center.view(target_shape)
            scale = scale.view(target_shape)
            denormalized = normalized_tensor * scale + center

        elif scope == "per_timestep" and len(normalized_tensor.shape) == 3:
            # Per-timestep denormalization
            # If anchor_timestep is provided, we need to align the tensor with the statistics
            # The statistics were computed with lowdim_past_timesteps past steps
            # The tensor has anchor_timestep as the index of the current timestep
            _batch_size, num_timesteps, _feature_dim = normalized_tensor.shape

            if anchor_timestep is not None and num_timesteps != center.shape[0]:
                if self.lowdim_past_timesteps is None:
                    raise ValueError(
                        "The normalizer is asked to denormalize a tensor with a different number"
                        f"of timesteps {num_timesteps} than the statistics ({center.shape[0]})"
                        "but lowdim_past_timesteps must be set to align the statistics with the tensor."
                        "This is likely because the data preprocessing metadata is not available."
                    )
                # The statistics were computed with lowdim_past_timesteps past steps
                # We need to crop them to align with the tensor's anchor_timestep
                # This maps: tensor[anchor_timestep] -> stats[lowdim_past_timesteps]

                # Calculate how many past and future timesteps the tensor has relative to its anchor
                tensor_past = anchor_timestep
                tensor_future = num_timesteps - anchor_timestep - 1

                # Crop statistics around stats_anchor_idx to match tensor's time range
                stats_anchor_idx = self.lowdim_past_timesteps

                # Check if crop would be valid
                start_idx = stats_anchor_idx - tensor_past
                end_idx = stats_anchor_idx + tensor_future + 1

                if start_idx >= 0 and end_idx <= len(center):
                    # Normal case: statistics fully cover the tensor's time range
                    cropped_center = crop_sequence(center, stats_anchor_idx, tensor_past, tensor_future)
                    cropped_scale = crop_sequence(scale, stats_anchor_idx, tensor_past, tensor_future)
                else:
                    # Edge case: need to clamp
                    start_idx = max(0, start_idx)
                    end_idx = min(len(center), end_idx)
                    cropped_center = center[start_idx:end_idx]  # [T', D]
                    cropped_scale = scale[start_idx:end_idx]  # [T', D]

                # Add batch dimension and broadcast
                cropped_center = cropped_center.unsqueeze(0)  # [1, T', D]
                cropped_scale = cropped_scale.unsqueeze(0)  # [1, T', D]

                denormalized = normalized_tensor * cropped_scale + cropped_center
            else:
                # No anchor provided, assume the tensor is already aligned
                center = center.unsqueeze(0)  # [1, T, D]
                scale = scale.unsqueeze(0)  # [1, T, D]
                denormalized = normalized_tensor * scale + center
        else:
            # Unsupported tensor shape
            logging.warning(f"Unsupported tensor shape for denormalization: {normalized_tensor.shape}")
            denormalized = normalized_tensor

        # For 6-channel point clouds with per-timestep denormalization, concatenate RGB back
        if is_point_cloud_6ch and scope == "per_timestep":
            denormalized = torch.cat([denormalized, rgb], dim=-1)

        return denormalized


class LakehouseNormalizer(RoboticsNormalizer):
    """Normalizer for the nested (embodiment-keyed) lakehouse statistics format.

    Reads the ``global`` section, which pools every embodiment of the mixture.
    Multiple stats files are merged with the mixture's stream weights so the
    statistics describe the sample distribution training actually sees.
    """

    normalization_params_cls: ClassVar[type[NormalizationParams]] = LakehouseNormalizationParams

    def __init__(
        self,
        normalization_params: LakehouseNormalizationParams,
        statistics_data: dict[str, Any] | list[dict[str, Any]] | None = None,
        statistics_path: str | list[str] | None = None,
        statistics_weights: StatisticsMergeWeights | None = None,
    ):
        """Initialize the normalizer.

        Args:
            normalization_params: ``LakehouseNormalizationParams`` with the field
                definitions, normalization settings, and ``stats_scope``.
            statistics_data: Pre-loaded statistics; a list is merged.
            statistics_path: Path (or list of paths) to statistics JSON files.
            statistics_weights: Per-dataset mixture weights for the merge, from
                ``lakehouse_statistics_merge_weights``. ``None`` weights every
                sample equally. Ignored when a single file is given.

        Raises:
            NotImplementedError: If ``stats_scope="per_embodiment"``, which is a
                staged follow-up. Raised at construction rather than mid-training.
            ValueError: If normalization is enabled but the ``global`` section
                carries no statistics.
        """
        if normalization_params.stats_scope == "per_embodiment":
            raise NotImplementedError(
                f"normalization.stats_scope={normalization_params.stats_scope!r} is not implemented yet; "
                "training normalizes with the mixture-weighted 'global' statistics. Per-embodiment "
                "normalization is a staged follow-up - the sections are already persisted in the "
                "checkpoint's stats.json, so no retraining is needed to adopt it."
            )
        # Set before super(), which merges any loaded list through the hook below.
        self._statistics_weights = statistics_weights
        super().__init__(normalization_params, statistics_data, statistics_path)

        if self.enabled and not (self.stats or {}).get(LAKEHOUSE_STATS_GLOBAL_KEY):
            raise ValueError(
                f"Lakehouse statistics have no usable {LAKEHOUSE_STATS_GLOBAL_KEY!r} section "
                f"(loaded from {statistics_path!r}), but normalization is enabled. Re-assemble the dataset so its "
                "aggregated stats file is embodiment-keyed and carries normalizable fields, or disable normalization."
            )

    def _align_statistics_list(self, statistics: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return the statistics unchanged - the nested format is not cropped here.

        Lakehouse statistics are keyed by section (``global`` / embodiment id / ``counts``),
        not by field, so the field-keyed crop the base class performs would walk the wrong
        level. Lakehouse also takes its lowdim window from ``conversion_past/future_low_dim_steps``
        rather than a per-source ``preprocessing_config.yaml``, so there are no per-source
        windows to align against.

        Args:
            statistics: Loaded per-dataset nested statistics dicts, in stream order.

        Returns:
            ``statistics`` unchanged.
        """
        return statistics

    def _merge_statistics_list(self, statistics: list[dict[str, Any]]) -> dict[str, Any]:
        """Weight-merge one statistics file per stream, section by section.

        Args:
            statistics: Loaded per-dataset statistics dicts, in stream order.

        Returns:
            The merged nested statistics: the ``global`` section and every
            embodiment section are pooled separately, each weighted by how often
            training visits that dataset's samples.
        """
        return merge_nested_lakehouse_statistics(statistics, weights=self._statistics_weights)

    def field_statistics(self, field_name: str, embodiment_id: str | None = None) -> dict[str, Any] | None:
        """Return one field's statistics from an embodiment section.

        Args:
            field_name: Canonical field name to look up.
            embodiment_id: Embodiment whose section to read; ``None`` selects the
                mixture-wide ``global`` section, which is what training uses today.

        Returns:
            The field's stats dict, or ``None`` when the section does not carry
            that field - embodiments legitimately differ in which fields they have.

        Raises:
            ValueError: If a specific embodiment is requested and the merged
                statistics carry no section for it, which means the statistics do
                not describe the data being normalized.
        """
        sections = self.stats or {}
        section = sections.get(embodiment_id or LAKEHOUSE_STATS_GLOBAL_KEY)
        if section is None and embodiment_id is not None:
            covered = sorted(set(sections) - {LAKEHOUSE_STATS_COUNTS_KEY, LAKEHOUSE_STATS_GLOBAL_KEY})
            raise ValueError(
                f"Lakehouse statistics have no {embodiment_id!r} section; the merged mixture covers {covered}."
            )
        return (section or {}).get(field_name)


def lakehouse_statistics_merge_weights(streams: list[LakehouseStreamParams]) -> StatisticsMergeWeights:
    """Derive statistics-merge weights from a lakehouse mixture's stream weighting knobs.

    Mirrors how MosaicML streaming samples the mixture, so the merged statistics
    describe the distribution the model actually trains on:

    - ``proportion`` set: dataset-level relative epoch share.
    - ``choose`` set: dataset-level absolute samples per epoch.
    - ``repeat`` set: per-sample visit rate (each dataset is repeated whole).
    - none set: per-sample visit rate of 1 (natural count weighting).

    Args:
        streams: The mixture's ``LakehouseStreamParams`` entries, in the same
            order as the paired ``dataset_statistics`` files.

    Returns:
        The ``StatisticsMergeWeights`` implementing the table above.

    Raises:
        ValueError: If streams mix weighting knobs or set a knob on only a
            subset of streams (MosaicML streaming enforces the same
            all-or-none contract at dataset construction time).
    """
    knob_values = {
        "proportion": [stream.proportion for stream in streams],
        "repeat": [stream.repeat for stream in streams],
        "choose": [stream.choose for stream in streams],
    }
    set_knobs = [knob for knob, values in knob_values.items() if any(value is not None for value in values)]
    if len(set_knobs) > 1:
        raise ValueError(
            f"lakehouse_streams mix weighting knobs {set_knobs}; use at most one of proportion / repeat / choose "
            "across the mixture."
        )
    if not set_knobs:
        return StatisticsMergeWeights(multipliers=[1.0] * len(streams), dataset_level=False)
    (knob,) = set_knobs
    values = knob_values[knob]
    if any(value is None for value in values):
        raise ValueError(
            f"lakehouse_streams set '{knob}' on only a subset of streams; stream weighting is all-or-none."
        )
    return StatisticsMergeWeights(
        multipliers=[float(value) for value in values], dataset_level=knob in ("proportion", "choose")
    )


def build_normalizer(
    data_params: RoboticsDataParams,
    *,
    statistics_data: dict[str, Any] | list[dict[str, Any]] | None = None,
    statistics_path: str | list[str] | None = None,
    pretrained_path: str | None = None,
) -> RoboticsNormalizer:
    """Build the normalizer matching a dataset's statistics format.

    The single place the normalizer class is chosen: lakehouse data params get
    ``LakehouseNormalizer`` (with the mixture's merge weights), everything else
    the WebDataset-format ``RoboticsNormalizer``.

    Args:
        data_params: Resolved data params for the run; its type selects the class
            and, for lakehouse mixtures, supplies the stream weighting knobs.
        statistics_data: Pre-loaded statistics to normalize with.
        statistics_path: Path (or list of paths) to statistics JSON files.
        pretrained_path: Checkpoint directory to restore from. Takes precedence:
            the saved config and ``stats.json`` fully determine the normalizer.

    Returns:
        The constructed normalizer.
    """
    is_lakehouse = isinstance(data_params, LakehouseDataParams)
    normalizer_cls = LakehouseNormalizer if is_lakehouse else RoboticsNormalizer
    if pretrained_path is not None:
        return normalizer_cls.from_pretrained(pretrained_path)

    kwargs = {}
    if is_lakehouse and len(data_params.dataset_statistics) > 1:
        kwargs["statistics_weights"] = lakehouse_statistics_merge_weights(data_params.lakehouse_streams)
    return normalizer_cls(
        data_params.normalization,
        statistics_data=statistics_data,
        statistics_path=statistics_path,
        **kwargs,
    )
