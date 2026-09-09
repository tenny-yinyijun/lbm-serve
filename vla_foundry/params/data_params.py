import logging
from dataclasses import dataclass, field
from typing import ClassVar

from vla_foundry.data.processor import get_processor
from vla_foundry.data.robotics.utils import LAKEHOUSE_AGGREGATED_STATS_RELPATH
from vla_foundry.params.base_data_params import DataParams
from vla_foundry.params.robotics.augmentation_params import DataAugmentationParams
from vla_foundry.params.robotics.normalization_params import (
    FieldNormalizationParams,
    LakehouseNormalizationParams,
    NormalizationParams,
)


def register_data_params(key: str):
    """
    Registers a DataParams subclass and sets its type attribute.
    Use decorator wrapper because draccus's model selection with --data.type doesn't
    automatically populate the attribute cfg.data.type
    """

    def decorator(cls):
        registered_cls = DataParams.register_subclass(key)(cls)
        registered_cls._type = key
        return registered_cls

    return decorator


@register_data_params("text")
@dataclass(frozen=True)
class TextDataParams(DataParams):
    pass


@register_data_params("text_untokenized")
@dataclass(frozen=True)
class TextUntokenizedDataParams(DataParams):
    tokenizer: str = field(default="EleutherAI/gpt-neox-20b")
    tokenizer_loaded = None

    @property
    def pad_token_id(self):
        if self.tokenizer_loaded is None:
            from vla_foundry.data.tokenizer import get_tokenizer

            tokenizer = get_tokenizer(self.tokenizer)
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            object.__setattr__(self, "tokenizer_loaded", tokenizer)
        return self.tokenizer_loaded.pad_token_id


@register_data_params("image_caption")
@dataclass(frozen=True)
class ImageCaptionDataParams(DataParams):
    processor: str = field(default="google/paligemma-3b-pt-224")
    processor_kwargs: dict = field(default_factory=dict)
    processor_loaded = None
    img_num_tokens: int = field(default=256)
    image_size: int = field(default=224)
    tokenizer: str = field(default=None)
    augmentation: DataAugmentationParams = field(default_factory=DataAugmentationParams)
    # video_caption-only: optional subset of frame indices to keep from each LAM clip.
    # E.g. [0] keeps only the temporally earliest frame (past-conditioned single-frame
    # training on data preprocessed as forward-time triplets); None keeps all frames.
    video_frames_to_use: list[int] | None = field(default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.tokenizer is None:
            object.__setattr__(self, "tokenizer", self.processor)

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        if hasattr(cfg.model, "image_size") and cfg.model.image_size is not None:
            object.__setattr__(self, "image_size", cfg.model.image_size)

    @property
    def image_token_id(self):
        if self.processor_loaded is None:
            object.__setattr__(self, "processor_loaded", get_processor(self))
        return self.processor_loaded.image_token_id

    @property
    def pad_token_id(self):
        if self.processor_loaded is None:
            object.__setattr__(self, "processor_loaded", get_processor(self))
        return self.processor_loaded.tokenizer.pad_token_id

    @property
    def eos_token_id(self):
        if self.processor_loaded is None:
            object.__setattr__(self, "processor_loaded", get_processor(self))
        return self.processor_loaded.tokenizer.eos_token_id


@register_data_params("robotics")
@dataclass(frozen=True)
class RoboticsDataParams(DataParams):
    """
    Configuration for robotics dataset field definitions and normalization.

    This dataclass defines which fields correspond to proprioception and actions,
    and how they should be normalized, replacing hardcoded field names in training scripts.
    """

    dataset_statistics: list[str] = field(default_factory=list)
    val_dataset_statistics: list[str] = field(default_factory=list)
    processor: str = field(default=None)
    processor_kwargs: dict = field(default_factory=dict)

    # Fraction of samples allowed to fail at any WebDataset pipeline stage before
    # training is aborted, i.e., 1.0 = tolerate all errors; 0.0 = raise on first error.
    wds_pipeline_error_tolerance: float = field(default=1.0)

    img_num_tokens: int = field(default=256)
    image_size: int = field(default=224)
    max_text_seq_len: int | None = field(default=None)

    # Language instruction types to use: "original", "randomized", "verbose", "alternative"
    language_instruction_types: list[str] = field(default_factory=lambda: ["original"])

    camera_names: list[str] = field(default_factory=list)
    image_indices: list[int] = field(default_factory=list)
    image_names: list[str] = field(default_factory=list)
    pad_missing_images: bool = field(default=False)
    mask_padded_images: bool = field(default=False)
    proprioception_fields: list[str] = field(default_factory=list)
    tactile_fields: list[str] = field(default_factory=list)
    action_fields: list[str] = field(default_factory=list)
    pose_groups: list[dict[str, str]] = field(default_factory=list)
    intrinsics_fields: list[str] = field(default_factory=list)
    extrinsics_fields: list[str] = field(default_factory=list)
    use_point_cloud: bool = field(default=False)
    # Total number of points for FPS sampling. If left as None, resolve_robotics_data_fields
    # fills it from the dataset-side preprocessing_config.yaml (when use_point_cloud=True).
    point_cloud_num_points: int | None = field(default=None)
    normalization: NormalizationParams = field(default_factory=NormalizationParams)
    augmentation: DataAugmentationParams = field(default_factory=DataAugmentationParams)

    lowdim_past_timesteps: int | None = field(default=None)
    lowdim_future_timesteps: int | None = field(default=None)
    action_dim: int = field(default=None)
    proprioception_dim: int | None = field(default=None)

    # Set to True by vla_foundry.params.resolve.resolve_derived_fields. This is
    # runtime state, not user config, so it is intentionally not a dataclass field.
    _resolved: ClassVar[bool] = False

    def __post_init__(self):
        try:
            self._post_init_impl()
        except (TypeError, ValueError, KeyError) as e:
            raise RuntimeError(f"RoboticsDataParams initialization failed: {type(e).__name__}: {e}") from e

    def _post_init_impl(self):
        super().__post_init__()

        if self.mask_padded_images and not self.pad_missing_images:
            raise ValueError("mask_padded_images requires pad_missing_images to be True")

        # Validate language instruction types
        valid_types = {"original", "randomized", "verbose", "alternative"}
        invalid_types = set(self.language_instruction_types) - valid_types
        if invalid_types:
            raise ValueError(f"Invalid language instruction types: {invalid_types}. Valid types are: {valid_types}")

        # If no pose groups are provided, they must be explicitly configured
        # Pose groups are now required to be explicitly specified in configuration
        if not self.pose_groups:
            logging.warning(
                "No pose groups specified. Relative coordinate transformations will not be available. "
                "Please add pose_groups to your data configuration if you need relative coordinates."
            )

        # Compute image_names from camera_names and image_indices
        if (self.image_names is None or len(self.image_names) == 0) and self.camera_names and self.image_indices:
            image_names = [f"{cname}_t{idx}" for idx in self.image_indices for cname in self.camera_names]
            object.__setattr__(self, "image_names", image_names)

        # For all used fields (proprioception and action), add default normalization parameters if not specified
        normalization_fields = self.normalization.field_configs
        for field_name in self.proprioception_fields + self.action_fields:
            if field_name not in normalization_fields:
                normalization_fields[field_name] = FieldNormalizationParams(
                    method=self.normalization.method, scope=self.normalization.scope, epsilon=self.normalization.epsilon
                )

        # Update normalization parameters with field-specific parameters
        object.__setattr__(self.normalization, "field_configs", normalization_fields)

        self._default_lowdim_window_from_normalization()

    def _default_lowdim_window_from_normalization(self):
        """Copy normalization.lowdim_past/future_timesteps into data.lowdim_* when unset.

        Idempotent and None-guarded, so it is safe to call from both __post_init__ and
        init_shared_attributes. The data-loading window (data.lowdim_*) and the statistics
        window (normalization.lowdim_*) are distinct concepts that may legitimately differ;
        this only fills data.* when the user left it unset.
        """
        if self.lowdim_past_timesteps is None and self.normalization.lowdim_past_timesteps is not None:
            object.__setattr__(self, "lowdim_past_timesteps", self.normalization.lowdim_past_timesteps)
        if self.lowdim_future_timesteps is None and self.normalization.lowdim_future_timesteps is not None:
            object.__setattr__(self, "lowdim_future_timesteps", self.normalization.lowdim_future_timesteps)

    def init_shared_attributes(self, cfg):
        super().init_shared_attributes(cfg)
        if cfg.data.processor:
            if hasattr(cfg.model, "hf_pretrained"):
                object.__setattr__(self, "processor", cfg.model.hf_pretrained)
            elif hasattr(cfg.model, "vlm_params") and hasattr(cfg.model.vlm_params, "hf_pretrained"):
                object.__setattr__(self, "processor", cfg.model.vlm_params.hf_pretrained)
        self._default_lowdim_window_from_normalization()


@dataclass(frozen=True)
class LakehouseStreamParams:
    """One stream in a weighted Lakehouse dataset mixture.

    Mirrors the stream-construction kwargs of
    ``lakehouse.dataset.streaming.LakehouseStream.from_lakehouse_root``. At most
    one weighting knob may be set, and the same knob must be set on every stream
    of the mixture: MosaicML streaming validates that natively at dataset
    construction time, and ``lakehouse_statistics_merge_weights`` re-derives it
    to weight the normalizer-statistics merge the same way.
    """

    local: str | None = field(default=None)
    remote: str | None = field(default=None)
    split: str | None = field(default=None)
    # Relative share of the combined epoch this stream contributes (e.g. 0.7 = 70%).
    proportion: float | None = field(default=None)
    # Up/downsample factor on the whole dataset (2.0 = see every sample twice per epoch).
    repeat: float | None = field(default=None)
    # Exact number of samples drawn from this stream per epoch.
    choose: int | None = field(default=None)


@register_data_params("lakehouse_robotics")
@dataclass(frozen=True)
class LakehouseDataParams(RoboticsDataParams):
    """Robotics data params for Lakehouse MDS datasets.

    Unlike the WebDataset robotics params, Lakehouse configs do not read
    ``preprocessing_config.yaml`` from manifest directories. The schema fields
    needed by the processor are explicit in the training config.

    Streams are the only convention: the scalar ``lakehouse_local/remote/split``
    (and ``val_lakehouse_*``) roots are single-dataset sugar, folded into one-entry
    ``lakehouse_streams`` / ``val_lakehouse_streams`` lists at init and then cleared,
    so downstream code reads the stream lists and a dumped config carries exactly
    one representation.

    ``dataset_statistics`` is always derived - one aggregated stats file per
    stream, in stream order - rather than configured, so it cannot go stale or
    pair with the streams in the wrong order. Multi-stream stats are merged at
    normalizer init, weighted by the mixture's ``proportion`` / ``repeat`` /
    ``choose`` knobs (see ``merge_nested_lakehouse_statistics``).

    This is the single canonical definition shared by training (the lakehouse
    dataloader merged via PR #754) and inference (the eval server decodes
    lakehouse-trained checkpoints whose config declares
    ``data.type: lakehouse_robotics``). Inference does not exercise the lakehouse
    roots / conversion windows / image-resize fields, but they must remain declared
    so saved checkpoint configs decode cleanly and so training configs that set
    ``image_resize_size`` / ``image_resize_method`` continue to parse.
    """

    # Single-dataset sugar, consumed at init: folded into a one-entry
    # ``lakehouse_streams`` and reset to None (streams are the only representation
    # that survives, so a dumped config re-decodes unchanged).
    lakehouse_local: str | None = field(default=None)
    lakehouse_remote: str | None = field(default=None)
    lakehouse_split: str | None = field(default=None)
    lakehouse_streams: list[LakehouseStreamParams] = field(default_factory=list)
    lakehouse_shuffle: bool = field(default=True)
    # Validation dataset is optional; when no val roots are configured and total_val_samples
    # is set, the validation loader falls back to the train streams. This mirrors the pragmatic
    # case where users have a small held-out shard inside the same lakehouse root (via ``split``).
    val_lakehouse_local: str | None = field(default=None)
    val_lakehouse_remote: str | None = field(default=None)
    val_lakehouse_split: str | None = field(default=None)
    val_lakehouse_streams: list[LakehouseStreamParams] = field(default_factory=list)
    conversion_past_low_dim_steps: int = field(default=0)
    conversion_future_low_dim_steps: int = field(default=0)
    image_resize_size: tuple[int, int] | None = field(default=None)
    image_resize_method: str = field(default="resize_fit")
    # Zero-pad declared action/proprio fields missing from a shard (mixed-embodiment).
    pad_missing_lowdim_fields: bool = field(default=False)
    # Zero the diffusion loss on action dims zero-filled for a sample (mixed-embodiment: H1 lacks G1-only joints).
    mask_padded_action_dims: bool = field(default=False)
    # Drop samples the converter flagged invalid (SampleMetadata.sample_valid_flag
    # = False, e.g., snap-tolerance violations) at collate time, refilling the
    # batch from the valid samples so batch shapes stay constant.
    skip_invalid_samples: bool = field(default=False)
    normalization: LakehouseNormalizationParams = field(default_factory=LakehouseNormalizationParams)

    def _post_init_impl(self):
        # The scalar roots are consumed, not kept: leaving both forms populated would
        # trip the mutual-exclusivity check when a dumped config is re-decoded, which
        # is how every inference entry point reads a checkpoint's config.yaml.
        object.__setattr__(
            self,
            "lakehouse_streams",
            self._resolve_streams(
                "lakehouse", self.lakehouse_streams, self.lakehouse_local, self.lakehouse_remote, self.lakehouse_split
            ),
        )
        object.__setattr__(self, "lakehouse_local", None)
        object.__setattr__(self, "lakehouse_remote", None)
        object.__setattr__(self, "lakehouse_split", None)
        if not self.lakehouse_streams:
            raise ValueError(
                "lakehouse_robotics requires at least one of lakehouse_local or lakehouse_remote "
                "(or a non-empty lakehouse_streams mixture)."
            )
        object.__setattr__(
            self,
            "val_lakehouse_streams",
            self._resolve_streams(
                "val_lakehouse",
                self.val_lakehouse_streams,
                self.val_lakehouse_local,
                self.val_lakehouse_remote,
                self.val_lakehouse_split,
            ),
        )
        object.__setattr__(self, "val_lakehouse_local", None)
        object.__setattr__(self, "val_lakehouse_remote", None)
        object.__setattr__(self, "val_lakehouse_split", None)
        self._derive_dataset_statistics()

        if not self.camera_names or not self.image_indices or not self.image_names:
            raise ValueError(
                "lakehouse_robotics requires explicit camera_names, image_indices, and image_names. "
                "No preprocessing_config.yaml is read in lakehouse mode."
            )

        super()._post_init_impl()

        if self.lowdim_past_timesteps is not None and self.lowdim_past_timesteps > self.conversion_past_low_dim_steps:
            raise ValueError(
                f"Requested lowdim_past_timesteps {self.lowdim_past_timesteps} exceeds Lakehouse conversion "
                f"past window {self.conversion_past_low_dim_steps}."
            )
        if (
            self.lowdim_future_timesteps is not None
            and self.lowdim_future_timesteps > self.conversion_future_low_dim_steps
        ):
            raise ValueError(
                f"Requested lowdim_future_timesteps {self.lowdim_future_timesteps} exceeds Lakehouse conversion "
                f"future window {self.conversion_future_low_dim_steps}."
            )

        if self.mask_padded_action_dims and not self.pad_missing_lowdim_fields:
            raise ValueError(
                "mask_padded_action_dims requires pad_missing_lowdim_fields=True "
                "(the per-dimension action mask only applies to zero-filled fields)."
            )

    @staticmethod
    def _resolve_streams(
        prefix: str,
        streams: list[LakehouseStreamParams],
        local: str | None,
        remote: str | None,
        split: str | None,
    ) -> list[LakehouseStreamParams]:
        """Fold the scalar single-root sugar into a stream list.

        Streams are the canonical source of truth after init, so a single-dataset
        config becomes a one-entry mixture (the caller then clears the scalars).

        Args:
            prefix: Field-name prefix used in error messages, ``"lakehouse"``
                (train) or ``"val_lakehouse"``.
            streams: The configured stream list, empty when unset.
            local: Configured scalar local root, if any.
            remote: Configured scalar remote root, if any.
            split: Configured scalar split, if any.

        Returns:
            The stream list to store: ``streams`` unchanged when it was set, a
            one-entry list built from the scalar roots, or an empty list when
            neither form is configured (how the optional validation dataset stays
            unconfigured; the train mixture is separately required to be non-empty).
            Every returned entry is guaranteed to carry a local or remote root.

        Raises:
            ValueError: If the stream list is set alongside any of the scalar
                fields, or if a configured stream carries neither a local nor a
                remote root.
        """
        if streams:
            if local is not None or remote is not None or split is not None:
                raise ValueError(
                    f"{prefix}_streams is mutually exclusive with {prefix}_local / {prefix}_remote / {prefix}_split; "
                    f"configure the mixture entirely via {prefix}_streams or use the scalar fields for a single "
                    "dataset."
                )
            for i, stream in enumerate(streams):
                if stream.local is None and stream.remote is None:
                    raise ValueError(
                        f"{prefix}_streams[{i}] has neither a local nor a remote root; every stream needs at "
                        "least one to be loadable."
                    )
            return streams
        if local is None and remote is None:
            return []
        return [LakehouseStreamParams(local=local, remote=remote, split=split)]

    def _derive_dataset_statistics(self):
        """Point ``dataset_statistics`` at one aggregated stats file per train stream.

        Lakehouse datasets always carry their statistics at a fixed path inside the
        dataset root, so these paths are derived rather than configured: a
        hand-written list can go stale when a root changes, or pair with the streams
        in the wrong order (the merge weights pair positionally). Every stream is
        known to carry a root by now (``_resolve_streams``), with the remote one
        preferred over the local cache.

        A configured value is tolerated only when it already matches, which is what
        lets a saved checkpoint config - dumped with the derived paths - decode again.

        Raises:
            ValueError: If a configured ``dataset_statistics`` differs from the
                derived paths.
        """
        derived = [
            f"{(stream.remote or stream.local).rstrip('/')}/{LAKEHOUSE_AGGREGATED_STATS_RELPATH}"
            for stream in self.lakehouse_streams
        ]
        if self.dataset_statistics and list(self.dataset_statistics) != derived:
            raise ValueError(
                "dataset_statistics is derived from the lakehouse stream roots and cannot be overridden: "
                f"configured {list(self.dataset_statistics)!r}, derived {derived!r}. Drop dataset_statistics "
                "from the config; to normalize with another run's statistics, resume from that checkpoint "
                "(its stats.json wins) instead of pointing at a different stats file."
            )
        object.__setattr__(self, "dataset_statistics", derived)
