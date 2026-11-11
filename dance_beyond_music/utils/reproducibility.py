from __future__ import annotations

import os
import platform
import random
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dance_beyond_music.utils.io import ensure_dir, save_json
from dance_beyond_music.utils.logging import get_logger
from dance_beyond_music.utils.type_defs import PathLike

logger = get_logger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seed set to %d", seed)


def _git_commit_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def capture_run_metadata() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    gpu_names = [torch.cuda.get_device_name(index) for index in range(device_count)]
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit_hash(),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_devices": gpu_names,
        "wandb_run_url": os.getenv("WANDB_RUN_URL"),
    }


def log_run_metadata(metadata: dict[str, Any], output_dir: PathLike) -> None:
    destination = ensure_dir(output_dir) / "run_metadata.json"
    save_json(metadata, destination)
    logger.info("Wrote run metadata to %s", destination)
