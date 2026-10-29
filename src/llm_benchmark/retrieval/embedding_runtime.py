"""Collect runtime metadata for local embedding execution."""

import platform
from importlib.metadata import version
from typing import Any

RUNTIME_PACKAGES = (
    "torch",
    "sentence-transformers",
    "transformers",
    "huggingface-hub",
    "tokenizers",
    "numpy",
)


def _thread_counts() -> dict[str, int]:
    import torch

    return {
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def collect_embedding_runtime(model: Any) -> dict[str, object]:
    tokenizer_class = type(model.tokenizer)

    return {
        "schema_version": 1,
        "python_version": platform.python_version(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "device": str(model.device),
        "embedding_dimension": model.get_embedding_dimension(),
        "max_sequence_length": model.max_seq_length,
        "parameter_dtypes": sorted(
            {
                str(parameter.dtype)
                for parameter in model.parameters()
                if parameter.is_floating_point()
            }
        ),
        "tokenizer_class": (
            f"{tokenizer_class.__module__}."
            f"{tokenizer_class.__qualname__}"
        ),
        "tokenizer_is_fast": bool(
            getattr(model.tokenizer, "is_fast", False)
        ),
        **_thread_counts(),
        "runtime_versions": {
            package: version(package)
            for package in RUNTIME_PACKAGES
        },
    }