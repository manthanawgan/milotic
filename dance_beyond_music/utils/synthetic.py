#vibeocded ts file for synthetic media generatn.

from __future__ import annotations

from typing import Any

import numpy as np

from dance_beyond_music.utils import synthetic_pose
from dance_beyond_music.utils.io import ensure_dir, save_json
from dance_beyond_music.utils.logging import get_logger
from dance_beyond_music.utils.synthetic_media import write_audio, write_image, write_video
from dance_beyond_music.utils.type_defs import PathLike

logger = get_logger(__name__)

_TEXT = (
    "slow swaying like a willow in still air",
    "arms rising like wings unfolding",
    "a bright bouncing jump with wide legs",
)
_FREQUENCY = (220.0, 330.0, 440.0)
_COLOR = ((50, 80, 200), (40, 190, 90), (220, 70, 60))


def _split_for(index: int, total: int) -> str:
    """Assign a deterministic split, giving larger sets two test samples."""
    if total >= 10:
        test_count = max(2, total // 10)
        val_count = max(1, total // 10)
        if index >= total - test_count:
            return "test"
        if index >= total - test_count - val_count:
            return "val"
        return "train"
    if total >= 3:
        if index == total - 1:
            return "test"
        if index == total - 2:
            return "val"
    return "train"


def generate_synthetic_dataset(
    root: PathLike,
    num_samples: int = 6,
    num_frames: int = 24,
    fps: int = 12,
    image_size: int = 96,
    seed: int = 0,
    render_videos: bool = True,
    write_ground_truth: bool = False,
) -> dict[str, Any]:
    """Create synthetic raw and (optionally) processed data under ``root``.

    Args:
        root: Dataset root containing ``raw/``, ``processed/``, and ``metadata/``.
        num_samples: Number of paired samples to create.
        num_frames: Frames per synthetic video and pose sequence.
        fps: Video frame rate.
        image_size: Square image size in pixels.
        seed: Deterministic seed for all generated values.
        render_videos: Write MP4 videos for pose-extraction demos.
        write_ground_truth: Also write pose/point-cloud files directly. Use this for
            fast tests; use ``render_videos`` plus ``extract_poses`` for the real path.

    Returns:
        The synthetic source manifest that was written to metadata.
    """
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    dataset_root = ensure_dir(root)
    for relative in (
        "raw/videos", "raw/text", "raw/audio", "raw/images", "raw/spatial",
        "processed/poses", "processed/point_clouds", "metadata",
    ):
        ensure_dir(dataset_root / relative)

    records: list[dict[str, Any]] = []
    for index in range(num_samples):
        motion_class = index % len(_TEXT)
        sample_id = f"synthetic-{index:03d}"
        paths: dict[str, str] = {}
        if render_videos:
            video = write_video(
                dataset_root / "raw" / "videos" / f"{sample_id}.mp4",
                motion_class, num_frames, fps,
            )
            paths["video"] = video.relative_to(dataset_root).as_posix()
        text_path = dataset_root / "raw" / "text" / f"{sample_id}.txt"
        text_path.write_text(_TEXT[motion_class] + "\n", encoding="utf-8")
        paths["text"] = text_path.relative_to(dataset_root).as_posix()
        audio = write_audio(
            dataset_root / "raw" / "audio" / f"{sample_id}.wav", _FREQUENCY[motion_class]
        )
        paths["audio"] = audio.relative_to(dataset_root).as_posix()
        image = write_image(
            dataset_root / "raw" / "images" / f"{sample_id}.png",
            _COLOR[motion_class], image_size,
        )
        paths["image"] = image.relative_to(dataset_root).as_posix()
        spatial_path = dataset_root / "raw" / "spatial" / f"{sample_id}.npy"
        np.save(spatial_path, synthetic_pose.spatial_descriptor(motion_class, seed + index))
        paths["spatial"] = spatial_path.relative_to(dataset_root).as_posix()
        if write_ground_truth:
            paths.update(synthetic_pose.write_ground_truth(
                dataset_root,
                sample_id,
                synthetic_pose.pose_sequence(motion_class, num_frames, seed + index),
            ))
        records.append(
            {
                "sample_id": sample_id,
                "split": _split_for(index, num_samples),
                "motion_class": motion_class,
                "paths": paths,
            }
        )

    manifest = {
        "version": 1,
        "synthetic_only": True,
        "description": "Tiny class-conditioned data for tests and smoke runs.",
        "modalities": ["text", "audio", "image", "spatial"],
        "samples": records,
    }
    save_json(manifest, dataset_root / "metadata" / "synthetic_sources.json")
    logger.info(
        "Generated %d synthetic samples under %s (videos=%s, ground_truth=%s)",
        num_samples, dataset_root, render_videos, write_ground_truth,
    )
    return manifest
