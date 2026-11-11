from __future__ import annotations

from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from dance_beyond_music.pipeline.cache import StageCache
from dance_beyond_music.pipeline.point_cloud_builder import PointCloudBuilder
from dance_beyond_music.pipeline.pose_backends.mediapipe_backend import MediaPipeBackend
from dance_beyond_music.pipeline.pose_extractor import PoseExtractor
from dance_beyond_music.pipeline.video_loader import VideoLoader
from dance_beyond_music.utils import env
from dance_beyond_music.utils.io import ensure_dir, save_pose_sequence
from dance_beyond_music.utils.logging import configure_logging, get_logger
from dance_beyond_music.utils.type_defs import PathLike
from scripts import CONFIG_PATH

logger = get_logger(__name__)


def _videos(raw_dir: PathLike) -> list[Path]:
    root = Path(raw_dir)
    if not root.is_dir():
        return []
    extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in extensions)


def _make_backend(cfg: DictConfig) -> MediaPipeBackend:
    name = str(cfg.data.get("pose_backend", "mediapipe")).lower()
    if name != "mediapipe":
        raise ValueError(
            f"Unsupported pose backend '{name}'. Phase 10 implements 'mediapipe'; "
            "ViTPose needs a checkpoint and is not wired yet."
        )
    model_path = cfg.data.get("pose_model_path")
    return MediaPipeBackend(
        model_path=str(model_path) if model_path else None,
        min_detection_confidence=float(cfg.data.get("min_detection_confidence", 0.3)),
    )


def extract_poses(cfg: DictConfig) -> dict[str, int]:
    OmegaConf.resolve(cfg)
    data = cfg.data
    poses_dir = ensure_dir(data.poses_dir)
    clouds_dir = ensure_dir(data.point_clouds_dir)
    cache = StageCache(data.cache_dir)
    loader = VideoLoader(target_fps=float(data.fps), max_frames=int(data.max_frames))
    extractor = PoseExtractor(_make_backend(cfg), cache=cache)
    builder = PointCloudBuilder()
    videos = _videos(data.raw_dir)
    if not videos:
        logger.warning("No videos found under %s; nothing to extract.", data.raw_dir)
        return {"processed": 0, "skipped": 0, "failed": 0}

    counts = {"processed": 0, "skipped": 0, "failed": 0}
    for video in videos:
        video_id = video.stem
        pose_path = poses_dir / f"{video_id}.npz"
        cloud_path = clouds_dir / f"{video_id}.npz"
        try:
            frames, timestamps = loader.load_with_timestamps(video)
            cache_key = extractor.cache_key(video_id, timestamps)
            cloud_cached = cache.is_cached("point_clouds", cache_key)
            if pose_path.is_file() and cloud_path.is_file() and cloud_cached:
                logger.info("Cache hit; skipping %s", video_id)
                counts["skipped"] += 1
                continue
            poses = extractor.extract(frames, video_id, timestamps_ms=timestamps)
            save_pose_sequence(poses, pose_path)
            if cloud_cached and cloud_path.is_file():
                logger.info("Reusing cached point cloud for %s", video_id)
            else:
                cloud = builder.build(poses)
                builder.save(cloud, cloud_path)
                cache.save("point_clouds", cache_key, np.asarray(cloud.points, dtype=np.float32))
            counts["processed"] += 1
            logger.info("Wrote %s and %s", pose_path.name, cloud_path.name)
        except Exception:
            counts["failed"] += 1
            logger.exception("Pose extraction failed for %s; continuing with other videos.", video)
    logger.info(
        "Pose extraction finished: %d processed, %d skipped, %d failed",
        counts["processed"], counts["skipped"], counts["failed"],
    )
    return counts


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging()
    device = env.get_device()
    logger.info("Pose extraction starting on %s", device)
    extract_poses(cfg)


if __name__ == "__main__":
    main()
