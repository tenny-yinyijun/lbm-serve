"""Dataset helpers.

TRIMMED FOR SERVING (lbm-serve). Upstream this module is ~384 lines of WebDataset
plumbing: a boto3-backed `s3://` gopen handler, broken-pipe/fd-leak patches applied to
`webdataset.gopen` at import time, `tarfile_to_samples_closing`, `deterministic_shuffle`,
shard-resume helpers. None of it is reachable when serving -- the only two functions any
importer on the serving path uses are the two kept below:

    vla_foundry/data/processor/__init__.py   -> text_to_seed
    vla_foundry/data/tokenizer/__init__.py   -> text_to_seed
    vla_foundry/params/train_experiment_params.py -> epochs_to_samples

The upstream module could not simply have its `import webdataset as wds` made lazy:
`tarfile_to_samples_closing` and `deterministic_shuffle` subclass `wds.PipelineStage`, so
`wds` is needed at class-definition time. Removing those classes is what drops
`webdataset` (and transitively `av`, `zstandard`, `braceexpand`) from the dependency set.

If you need the dataloading path, use vla_foundry_internal -- do not grow this file back.
"""

import hashlib
from collections.abc import Sequence

from vla_foundry.file_utils import load_dataset_manifest


def epochs_to_samples(manifest_paths: Sequence[str], num_epochs: int) -> int:
    """
    Compute total samples as `num_epochs * sum(num_sequences in manifests)`.

    Args:
        manifest_paths: Sequence of manifest file paths/URIs.
        num_epochs: Number of epochs to iterate over the combined dataset.
    """
    manifests = [load_dataset_manifest(path) for path in manifest_paths]
    num_samples = 0
    for m in manifests:
        num_samples += sum(i["num_sequences"] for i in m)
    return num_samples * num_epochs


def text_to_seed(text: str) -> int:
    """Convert an arbitrary string to a stable 32-bit seed via SHA-256."""
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32 - 1)
