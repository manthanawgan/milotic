from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np

from dance_beyond_music.utils.type_defs import FrameArray, PathLike, PoseResult


def ensure_dir(path: PathLike) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_video(path: PathLike) -> list[FrameArray]:
    from dance_beyond_music.utils.video import decode_video

    return decode_video(path)


def save_pose_sequence(poses: list[PoseResult], path: PathLike) -> None:
    if not poses:
        raise ValueError("Cannot save an empty pose sequence.")
    destination = Path(path)
    ensure_dir(destination.parent)
    np.savez_compressed(
        destination,
        keypoints=np.stack([pose.keypoints for pose in poses]),
        confidence=np.stack([pose.confidence for pose in poses]),
        frame_idx=np.asarray([pose.frame_idx for pose in poses], dtype=np.int64),
        timestamp_ms=np.asarray([pose.timestamp_ms for pose in poses], dtype=np.float64),
        backend_name=np.asarray([pose.backend_name for pose in poses], dtype=str),
    )


def load_pose_sequence(path: PathLike) -> list[PoseResult]:
    with np.load(Path(path)) as archive:
        keypoints = archive["keypoints"]
        confidence = archive["confidence"]
        frame_indices = cast(list[int], archive["frame_idx"].tolist())
        timestamps = cast(list[float], archive["timestamp_ms"].tolist())
        backend_names = cast(list[str], archive["backend_name"].tolist())
    return [
        PoseResult(
            keypoints=keypoints[index],
            confidence=confidence[index],
            frame_idx=int(frame_indices[index]),
            timestamp_ms=float(timestamps[index]),
            backend_name=str(backend_names[index]),
        )
        for index in range(len(frame_indices))
    ]


def load_json(path: PathLike) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object in {path!s}.")
    return cast(dict[str, Any], payload)


def save_json(data: dict[str, Any], path: PathLike, indent: int = 2) -> None:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=indent, sort_keys=True)
        handle.write("\n")
