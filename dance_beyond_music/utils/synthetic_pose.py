#vibeocded ts file for synthetic media generatn.


from __future__ import annotations

from pathlib import Path

import numpy as np

from dance_beyond_music.utils.io import ensure_dir, save_pose_sequence
from dance_beyond_music.utils.type_defs import PoseResult

NUM_KEYPOINTS = 33


def pose_sequence(
    motion_class: int, num_frames: int, seed: int, num_keypoints: int = NUM_KEYPOINTS
) -> list[PoseResult]:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 0.12, size=(num_keypoints, 3)).astype(np.float32)
    timeline = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    amplitude = 0.10 + 0.04 * motion_class
    direction = np.array([1.0, 0.3, 0.5], dtype=np.float32)
    poses: list[PoseResult] = []
    for index, phase in enumerate(timeline):
        wave = np.sin(2.0 * np.pi * (phase + 0.15 * motion_class))
        offset = amplitude * wave * direction
        if motion_class == 2:
            offset = offset + np.array([0.0, -0.12 * abs(wave), 0.0], dtype=np.float32)
        keypoints = (base + offset[None, :]).astype(np.float32)
        poses.append(
            PoseResult(
                keypoints=keypoints,
                confidence=np.full(num_keypoints, 0.9, dtype=np.float32),
                frame_idx=index,
                timestamp_ms=float(index * (1000.0 / 12.0)),
                backend_name="synthetic",
            )
        )
    return poses


def write_ground_truth(root: Path, sample_id: str, poses: list[PoseResult]) -> dict[str, str]:
    pose_path = root / "processed" / "poses" / f"{sample_id}.npz"
    cloud_path = root / "processed" / "point_clouds" / f"{sample_id}.npz"
    save_pose_sequence(poses, pose_path)
    points = np.stack([pose.keypoints for pose in poses], axis=0).astype(np.float32)
    ensure_dir(cloud_path.parent)
    np.savez_compressed(
        cloud_path,
        points=points,
        timestamps=np.asarray([pose.timestamp_ms for pose in poses], dtype=np.float64),
    )
    return {
        "pose": pose_path.relative_to(root).as_posix(),
        "point_cloud": cloud_path.relative_to(root).as_posix(),
    }


def spatial_descriptor(motion_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    descriptor = rng.normal(0.0, 0.05, size=16).astype(np.float32)
    descriptor[motion_class] += 1.0
    descriptor[3 + motion_class] += 0.6
    return descriptor
