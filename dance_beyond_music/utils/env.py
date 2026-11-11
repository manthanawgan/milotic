from __future__ import annotations

import torch

from dance_beyond_music.utils.logging import get_logger

logger = get_logger(__name__)
_MIB = 1024 * 1024


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def get_available_gpus() -> list[int]:
    if not torch.cuda.is_available():
        return []
    return list(range(torch.cuda.device_count()))


def get_gpu_memory(device_idx: int) -> dict[str, int]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot read GPU memory.")
    try:
        properties = torch.cuda.get_device_properties(device_idx)
        free_bytes, _ = torch.cuda.mem_get_info(device_idx)
    except (AssertionError, RuntimeError) as error:
        raise RuntimeError(f"CUDA device {device_idx} is not available.") from error
    return {
        "total_mb": int(properties.total_memory / _MIB),
        "free_mb": int(free_bytes / _MIB),
    }


def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    return device


def assert_gpu_available() -> None:

    if torch.cuda.is_available():
        return
    raise RuntimeError(
        "This step requires a CUDA GPU, but none is visible. "
        "Run inside the Docker dev service with the NVIDIA Container Toolkit, "
        "or set CUDA_VISIBLE_DEVICES to a valid device index."
    )
