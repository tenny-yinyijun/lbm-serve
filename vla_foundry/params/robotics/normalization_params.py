from dataclasses import dataclass, field

from vla_foundry.params.base_params import BaseParams

# Which section of the nested lakehouse statistics normalization reads.
STATS_SCOPES = ("global", "per_embodiment")


@dataclass(frozen=True)
class FieldNormalizationParams:
    """Configuration for a specific field's normalization.

    We create a dictionary of these objects in NormalizationParams which is not properly serialized by draccus.
    So we use dataclasses_json to make sure the result is still properly serializable.
    """

    method: str = field(default="std")  # "std", "percentile_5_95", "percentile_1_99" "min_max"
    scope: str = field(default="global")  # "global" or "per_timestep"
    epsilon: float = field(default=1e-8)
    enabled: bool = field(default=True)

    def to_dict(self):
        return {
            "method": self.method,
            "scope": self.scope,
            "epsilon": self.epsilon,
            "enabled": self.enabled,
        }

    def __reduce__(self):
        """Control how this object is pickled/serialized.

        This helps avoid the !!python/object tag in YAML output.
        """
        # Just return the data as a tuple of (class, args)
        # This will make it serialize as a plain mapping
        return (self.__class__, (self.method, self.scope, self.epsilon, self.enabled))

    # This is what PyYAML will use for representing the object
    def __repr__(self):
        return str(self.to_dict())


@dataclass(frozen=True)
class NormalizationParams(BaseParams):
    """
    Configuration for robotics data normalization that defines which fields to normalize and how.

    Note about per-timestep normalization:
    Some fields may be normalized "per-timestep". In such a case, the time sequences are normalized with respective
    time sequences in the statistics. The data-loading window (`data.lowdim_past_timesteps` /
    `data.lowdim_future_timesteps`) may be narrower than the statistics window, and is realigned per call via
    `anchor_timestep`. The normalization window here is different: it identifies *where the anchor sits inside
    the statistics*, so it is always taken from the dataset-side preprocessing config, and an explicitly
    configured value that disagrees with the dataset is rejected rather than silently shifting the statistics
    (see `vla_foundry.params.resolve.resolve_normalization_timesteps`).
    """

    enabled: bool = field(default=True)

    # Default parameters to be used for all fields if not specified in field_configs
    method: str = field(default="std")  # "std", "percentile_5_95", "percentile_1_99" "min_max"
    scope: str = field(default="global")  # "global" or "per_timestep"
    epsilon: float = field(default=1e-8)
    include_fields: list[str] = field(default_factory=list)
    centered_norm: bool = field(default=False)

    # Field-specific configurations (initialized in __post_init__)
    field_configs: dict[str, FieldNormalizationParams] = field(default_factory=dict)

    # Shared attributes. Overwritten in init_shared_attributes.
    # Low-dimensional trajectory window captured during preprocessing. With several
    # statistics sources this is the common window, i.e. the min across sources.
    lowdim_past_timesteps: int | None = field(default=None)
    lowdim_future_timesteps: int | None = field(default=None)
    # Each source's own window, ordered to match data.dataset_statistics. The
    # normalizer uses these to crop every source's per-timestep statistics to the
    # common window before merging them. Empty when the resolve step has not run
    # (e.g. Lakehouse datasets, which take their window from conversion_*).
    per_source_lowdim_past: list[int] = field(default_factory=list)
    per_source_lowdim_future: list[int] = field(default_factory=list)

    def to_dict(self):
        return {
            "enabled": self.enabled,
            "method": self.method,
            "scope": self.scope,
            "epsilon": self.epsilon,
            "include_fields": self.include_fields,
            "centered_norm": self.centered_norm,
            "field_configs": {k: v.to_dict() for k, v in self.field_configs.items()},
            "lowdim_past_timesteps": self.lowdim_past_timesteps,
            "lowdim_future_timesteps": self.lowdim_future_timesteps,
        }

    def __post_init__(self):
        self.check_asserts()
        if self.method == "std":
            # std is always centered so we force centered_norm to True
            object.__setattr__(self, "centered_norm", True)

    def check_asserts(self):
        if self.method not in ["std", "percentile_5_95", "percentile_1_99", "min_max"]:
            raise ValueError(f"Invalid normalization method: {self.method}")
        if self.scope not in ["global", "per_timestep"]:
            raise ValueError(f"Invalid normalization scope: {self.scope}")

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        # Deduplicate: if the same field name exists in both proprioception_fields
        # and action_fields, duplicates are removed while preserving order.
        include_fields = list(dict.fromkeys(cfg.data.proprioception_fields + cfg.data.action_fields))
        # Add point_cloud if enabled
        if cfg.data.use_point_cloud:
            include_fields.append("point_cloud")
        # Currently we don't support normalization of intrinsics and extrinsics fields
        object.__setattr__(self, "include_fields", include_fields)

        field_configs = dict(self.field_configs)
        for field_name in include_fields:
            if field_name not in field_configs:
                field_configs[field_name] = FieldNormalizationParams(
                    method=self.method,
                    scope=self.scope,
                    epsilon=self.epsilon,
                    enabled=self.enabled,
                )

        # Validate that point_cloud normalization uses min_max method
        if (
            cfg.data.use_point_cloud
            and "point_cloud" in field_configs
            and field_configs["point_cloud"].method != "min_max"
        ):
            pc_method = field_configs["point_cloud"].method
            raise ValueError(
                f"Point cloud normalization must use 'min_max' method, but got '{pc_method}'. "
                "Point clouds require min_max normalization to work correctly. "
                "Please set normalization.field_configs.point_cloud.method to 'min_max' in your config."
            )

        object.__setattr__(self, "field_configs", field_configs)


@dataclass(frozen=True)
class LakehouseNormalizationParams(NormalizationParams):
    """Normalization config for Lakehouse datasets.

    Lakehouse datasets do not write the WebDataset-side ``preprocessing_config.yaml``
    next to the statistics file, so the resolve.py path is skipped (see
    ``LakehouseDataParams._post_init_impl``). The available lowdim window is
    instead taken from ``data.conversion_past/future_low_dim_steps``.
    """

    # "global" pools across all embodiments; "per_embodiment" would select stats by
    # each sample's embodiment id (staged follow-up, not implemented).
    stats_scope: str = field(default="global")

    def to_dict(self):
        return {**super().to_dict(), "stats_scope": self.stats_scope}

    def check_asserts(self):
        super().check_asserts()
        if self.stats_scope not in STATS_SCOPES:
            raise ValueError(f"Invalid normalization stats_scope: {self.stats_scope}; expected one of {STATS_SCOPES}.")

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)

        if not cfg.data.dataset_statistics:
            raise ValueError("Robotics normalization requires dataset_statistics.")

        available_past = self.lowdim_past_timesteps
        if available_past is None:
            available_past = cfg.data.conversion_past_low_dim_steps
        available_future = self.lowdim_future_timesteps
        if available_future is None:
            available_future = cfg.data.conversion_future_low_dim_steps

        if available_past is None or available_future is None:
            raise ValueError(
                "Lakehouse normalization requires explicit conversion_past_low_dim_steps and "
                "conversion_future_low_dim_steps on data, or lowdim window values on data.normalization."
            )

        object.__setattr__(self, "lowdim_past_timesteps", available_past)
        object.__setattr__(self, "lowdim_future_timesteps", available_future)
