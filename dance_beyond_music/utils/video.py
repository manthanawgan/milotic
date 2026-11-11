#Low-level video decoding with OpenCV.


from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import cast

import cv2
import numpy as np

from dance_beyond_music.utils.logging import get_logger
from dance_beyond_music.utils.type_defs import FrameArray, PathLike

logger = get_logger(__name__)

_DEFAULT_FPS = 30.0


def _open_capture(path: PathLike) -> cv2.VideoCapture:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Video file not found: {source}")
    capture = cv2.VideoCapture(source.as_posix())
    if not capture.isOpened():
        capture.release()
        raise ValueError(
            f"OpenCV could not open video {source}. "
            "Check the codec and that FFmpeg support is enabled in your OpenCV build."
        )
    return capture


def video_info(path: PathLike) -> tuple[float, int]:
    capture = _open_capture(path)
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if not np.isfinite(fps) or fps <= 0.0:
        fps = _DEFAULT_FPS
    return fps, max(frame_count, 0)


def iter_frames_with_timestamps(path: PathLike) -> Iterator[tuple[FrameArray, float]]:
    capture = _open_capture(path)
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(source_fps) or source_fps <= 0.0:
        source_fps = _DEFAULT_FPS
    index = 0
    try:
        while True:
            read_ok, frame = capture.read()
            if not read_ok or frame is None:
                break
            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not np.isfinite(timestamp_ms) or timestamp_ms <= 0.0:
                timestamp_ms = index * (1000.0 / source_fps)
            rgb = cast(FrameArray, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not np.issubdtype(rgb.dtype, np.uint8):
                raise ValueError(f"Decoder returned dtype {rgb.dtype}; expected uint8.")
            yield rgb, timestamp_ms
            index += 1
    finally:
        capture.release()


def decode_video(path: PathLike) -> list[FrameArray]:
    frames = [frame for frame, _ in iter_frames_with_timestamps(path)]
    if not frames:
        raise ValueError(f"Video {path!s} decoded to zero frames.")
    logger.debug("Decoded %d frames from %s", len(frames), path)
    return frames


def decode_video_with_timestamps(path: PathLike) -> tuple[list[FrameArray], list[float]]:
    pairs = list(iter_frames_with_timestamps(path))
    if not pairs:
        raise ValueError(f"Video {path!s} decoded to zero frames.")
    frames = [frame for frame, _ in pairs]
    timestamps = [timestamp for _, timestamp in pairs]
    return frames, timestamps
