"""Derive config fields from the referenced datasets on disk.

This module is the only place in `vla_foundry/params/` that performs
dataset-derived filesystem I/O. The dataclasses themselves stay pure — they
never read a manifest, stats file, or preprocessing_config.yaml. A caller
(trainer, inference, tooling) that holds a partially-specified config invokes
`cfg.resolve_derived_fields()` explicitly when the datasets referenced by the
config are accessible.

Saved configs coming off a training checkpoint already have every derived
field populated, so loading them via `load_experiment_params_from_yaml` does
not require this module.

Per-field semantics inside `resolve_robotics_data_fields`:
  - `camera_names`, `image_indices`, `image_names`: filled from the dataset-side
    preprocessing_config.yaml only when unset (or empty). User-set values are
    left alone (no re-validation against the dataset).
  - `point_cloud_num_points`: when `use_point_cloud=True`, filled from the
    preprocessing_config.yaml when unset (None). A user-set value that
    disagrees with the dataset raises `AssertionError`.
  - `action_dim`, `proprioception_dim`: filled from the stats file when unset.
    A user-set value that disagrees with the stats raises `AssertionError`.

After per-field resolution, `resolve_derived_fields` re-runs
`cfg.init_shared_attributes(cfg)` so that values resolved on the dataset side
(notably `normalization.lowdim_past/future_timesteps`) propagate into
`data.lowdim_*` and `model.lowdim_*`. It then sets `data._resolved = True` so
callers can distinguish configs resolved in this process from already
materialized saved configs.
"""

import os

from vla_foundry.file_utils import yaml_load
from vla_foundry.params.base_params import BaseParams
from vla_foundry.params.data_params import RoboticsDataParams
from vla_foundry.params.robotics.normalization_params import NormalizationParams


def resolve_derived_fields(cfg: BaseParams) -> None:
    """Fill every dataset-derived field on `cfg` that isn't already set.

    Idempotent: calling twice is safe and a no-op on the second call. Per-field
    semantics are documented in this module's docstring.
    """
    data = getattr(cfg, "data", None)
    if isinstance(data, RoboticsDataParams):
        if getattr(data, "_resolved", False):
            return
        # Local import to avoid a circular dependency with data_params.
        from vla_foundry.params.data_params import LakehouseDataParams

        if isinstance(data, LakehouseDataParams):
            # Lakehouse configs are mostly self-describing (camera_names,
            # image_indices, lowdim window come from YAML or conversion_*).
            # Only action_dim / proprioception_dim still need to be derived
            # from the stats file.
            resolve_action_proprio_dims(data)
        else:
            resolve_robotics_data_fields(data)
            resolve_normalization_timesteps(data.normalization, data)
        # Re-run the propagation chain now that dataset-derived values
        # (norm.lowdim_*, data.action_dim, data.proprioception_dim, ...) are
        # filled in. Without this, data.lowdim_* and model.lowdim_* would stay
        # None whenever the YAML omits them.
        if hasattr(cfg, "init_shared_attributes"):
            cfg.init_shared_attributes(cfg)
        object.__setattr__(data, "_resolved", True)


def resolve_robotics_data_fields(data: RoboticsDataParams) -> None:
    """Fill camera_names, image_indices, image_names, point_cloud_num_points,
    action_dim, proprioception_dim from dataset manifests and statistics."""
    try:
        _resolve_robotics_data_fields_impl(data)
    except (TypeError, ValueError, KeyError) as e:
        raise RuntimeError(
            f"resolve_robotics_data_fields failed: {type(e).__name__}: {e}\n"
            "Check that dataset_statistics paths are correct and all proprioception_fields "
            "and action_fields are present in the stats file."
        ) from e


def _resolve_robotics_data_fields_impl(data: RoboticsDataParams) -> None:
    needs_processing_configs = data.use_point_cloud or any(
        x is None or len(x) == 0 for x in [data.camera_names, data.image_indices]
    )
    if needs_processing_configs:
        processing_configs = [_load_preprocessing_config(m) for m in data.dataset_manifest]
    else:
        processing_configs = []

    if data.camera_names is None or len(data.camera_names) == 0:
        camera_names = processing_configs[0]["camera_names"]
        for processing_config in processing_configs:
            if processing_config["camera_names"] != camera_names:
                raise ValueError(
                    f"Camera names mismatch between preprocessing configs: {processing_config['camera_names']} "
                    f"and {camera_names}. Please provide camera names explicitly or use coherent data sources."
                )
        object.__setattr__(data, "camera_names", camera_names)

    if data.image_indices is None or len(data.image_indices) == 0:
        image_indices = processing_configs[0]["image_indices"]
        for processing_config in processing_configs:
            if processing_config["image_indices"] != image_indices:
                raise ValueError(
                    f"Image indices mismatch between preprocessing configs: {processing_config['image_indices']} "
                    f"and {image_indices}. Please provide image indices explicitly or use coherent data sources."
                )
        object.__setattr__(data, "image_indices", image_indices)

    if data.image_names is None or len(data.image_names) == 0:
        image_names = [f"{cname}_t{idx}" for idx in data.image_indices for cname in data.camera_names]
        object.__setattr__(data, "image_names", image_names)

    if data.use_point_cloud and processing_configs and "point_cloud_num_points" in processing_configs[0]:
        dataset_num_points = processing_configs[0]["point_cloud_num_points"]
        if data.point_cloud_num_points is None:
            object.__setattr__(data, "point_cloud_num_points", dataset_num_points)
        elif data.point_cloud_num_points != dataset_num_points:
            raise AssertionError(
                f"point_cloud_num_points mismatch: user-provided {data.point_cloud_num_points} does not match "
                f"dataset-side {dataset_num_points} from preprocessing_config.yaml. "
                "Provide the correct value or set it to None to auto-fill from the dataset."
            )

    resolve_action_proprio_dims(data)


def resolve_action_proprio_dims(data: RoboticsDataParams) -> None:
    """Fill action_dim and proprioception_dim from the stats file when unset."""
    if data.action_dim is not None and data.proprioception_dim is not None:
        return
    if not data.dataset_statistics:
        raise ValueError("Robotics datasets require dataset_statistics to be provided.")

    from vla_foundry.data.robotics.normalization import build_normalizer

    normalizer = build_normalizer(data, statistics_path=data.dataset_statistics)

    action_dim = _sum_field_dims(normalizer, data.action_fields, role="Action")
    if data.action_dim is None:
        object.__setattr__(data, "action_dim", action_dim)
    elif data.action_dim != action_dim:
        raise AssertionError(
            f"Action dimension mismatch: user-provided {data.action_dim} does not match "
            f"computed {action_dim}. Provide the correct action_dim or set it to None to "
            "auto-compute from action_fields. This may also indicate a discrepancy between "
            "action_fields and the normalization parameters."
        )

    proprioception_dim = _sum_field_dims(normalizer, data.proprioception_fields, role="Proprioception")
    if data.proprioception_dim is None:
        object.__setattr__(data, "proprioception_dim", proprioception_dim)
    elif data.proprioception_dim != proprioception_dim:
        raise AssertionError(
            f"Proprioception dimension mismatch: user-provided {data.proprioception_dim} does "
            f"not match computed {proprioception_dim}. Provide the correct proprioception_dim "
            "or set it to None to auto-compute from proprioception_fields."
        )


def resolve_normalization_timesteps(norm: NormalizationParams, data: RoboticsDataParams) -> None:
    """Fill the normalization lowdim window from the dataset-side preprocessing_config.yaml.

    ``norm.lowdim_past_timesteps`` is the index of the anchor timestep *inside the
    per-timestep statistics arrays* (see ``RoboticsNormalizer.normalize_tensor``,
    which uses it as ``stats_anchor_idx``). It therefore has to equal the
    ``past_lowdim_steps`` the statistics were computed with. Any smaller value is
    still a valid index, so a mismatch shifts every per-timestep mean/std by that
    many timesteps without raising: an explicitly configured value that disagrees
    with the dataset is rejected here instead.

    The per-source windows are recorded on ``norm`` so ``RoboticsNormalizer`` can
    crop each source's per-timestep statistics to the common window before merging
    them; without that, sources preprocessed with different windows cannot be
    merged at all.
    """
    dataset_statistics = data.dataset_statistics
    if not dataset_statistics:
        raise ValueError("Robotics normalization requires dataset_statistics.")

    statistics_paths = [dataset_statistics] if isinstance(dataset_statistics, str) else list(dataset_statistics)

    per_source_past = []
    per_source_future = []
    for statistics_path in statistics_paths:
        processing_config = _load_preprocessing_config(statistics_path)
        per_source_past.append(processing_config["past_lowdim_steps"])
        per_source_future.append(processing_config["future_lowdim_steps"])

    # The common window is the min across sources. Wider sources get their
    # per-timestep statistics cropped to it before the merge.
    available_past = min(per_source_past)
    available_future = min(per_source_future)

    if norm.lowdim_past_timesteps is not None and norm.lowdim_past_timesteps != available_past:
        raise ValueError(
            f"normalization.lowdim_past_timesteps={norm.lowdim_past_timesteps} does not match "
            f"past_lowdim_steps={available_past} from the preprocessing config(s) {per_source_past}. "
            "This value indexes the per-timestep statistics, so a different one shifts them silently. "
            "Remove lowdim_past_timesteps from the normalization config to take the dataset's value."
        )
    if norm.lowdim_future_timesteps is not None and norm.lowdim_future_timesteps != available_future:
        raise ValueError(
            f"normalization.lowdim_future_timesteps={norm.lowdim_future_timesteps} does not match "
            f"future_lowdim_steps={available_future} from the preprocessing config(s) {per_source_future}. "
            "This value sizes the per-timestep statistics window, so a different one shifts them silently. "
            "Remove lowdim_future_timesteps from the normalization config to take the dataset's value."
        )

    # The data-side loading window is independent of the anchor and only has to fit.
    if data.lowdim_past_timesteps is not None and data.lowdim_past_timesteps > available_past:
        raise ValueError(
            f"Requested lowdim_past_timesteps {data.lowdim_past_timesteps} exceeds available past timesteps "
            f"{available_past} from at least one data source."
        )
    if data.lowdim_future_timesteps is not None and data.lowdim_future_timesteps > available_future:
        raise ValueError(
            f"Requested lowdim_future_timesteps {data.lowdim_future_timesteps} exceeds available future timesteps "
            f"{available_future} from at least one data source."
        )

    object.__setattr__(norm, "per_source_lowdim_past", per_source_past)
    object.__setattr__(norm, "per_source_lowdim_future", per_source_future)
    object.__setattr__(norm, "lowdim_past_timesteps", available_past)
    object.__setattr__(norm, "lowdim_future_timesteps", available_future)


def _load_preprocessing_config(reference_path: str) -> dict:
    """Load preprocessing_config.yaml from the directory of `reference_path`.

    `reference_path` may be a manifest path or a stats path — both are expected
    to live next to the preprocessing_config.yaml that produced them.
    """
    path = os.path.dirname(reference_path)
    processing_config = yaml_load(os.path.join(path, "preprocessing_config.yaml"))
    # Handle indexed format from collect_preprocessing_configs (e.g. {0: {...}, 1: {...}})
    if processing_config and all(isinstance(k, int) for k in processing_config):
        processing_config = processing_config[0]
    return processing_config


def _sum_field_dims(normalizer, field_names: list[str], *, role: str) -> int:
    total = 0
    for field_name in field_names:
        field_stats = normalizer.field_statistics(field_name)
        if field_stats is None:
            raise ValueError(f"{role} field '{field_name}' missing from normalization statistics.")
        total += len(field_stats["mean"])
    return total
