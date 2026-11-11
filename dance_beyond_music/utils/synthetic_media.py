#vibeocded ts file for synthetic media generatn.

from __future__ import annotations

import wave
from pathlib import Path

import cv2
import numpy as np

from dance_beyond_music.utils.io import ensure_dir
from dance_beyond_music.utils.type_defs import PathLike

_JOINTS = {
    "head": (0.500, 0.188),
    "neck": (0.500, 0.313),
    "lsh": (0.422, 0.344),
    "rsh": (0.578, 0.344),
    "lelb": (0.375, 0.521),
    "relb": (0.625, 0.521),
    "lwr": (0.344, 0.688),
    "rwr": (0.656, 0.688),
    "lhip": (0.453, 0.542),
    "rhip": (0.547, 0.542),
    "lkne": (0.438, 0.729),
    "rkne": (0.562, 0.729),
    "lank": (0.430, 0.917),
    "rank": (0.570, 0.917),
}
_BONES = (
    ("neck", "head"),
    ("lsh", "rsh"),
    ("lsh", "lelb"),
    ("lelb", "lwr"),
    ("rsh", "relb"),
    ("relb", "rwr"),
    ("lsh", "lhip"),
    ("rsh", "rhip"),
    ("lhip", "rhip"),
    ("lhip", "lkne"),
    ("lkne", "lank"),
    ("rhip", "rkne"),
    ("rkne", "rank"),
)


def _motion_joints(motion_class: int, phase: float) -> dict[str, tuple[float, float]]:
    joints = {name: list(point) for name, point in _JOINTS.items()}
    wave = np.sin(2.0 * np.pi * phase)
    if motion_class == 0:  # slow sideways sway, arms relaxed
        for point in joints.values():
            point[0] += 0.030 * wave
        joints["lwr"][0] -= 0.020 * wave
        joints["rwr"][0] -= 0.020 * wave
    elif motion_class == 1:  # arms rising like wings
        lift = 0.15 * (0.5 - 0.5 * np.cos(2.0 * np.pi * phase))
        for side, sign in (("l", -1.0), ("r", 1.0)):
            joints[f"{side}elb"][1] -= lift * 0.5
            joints[f"{side}elb"][0] += sign * lift * 0.4
            joints[f"{side}wr"][1] -= lift
            joints[f"{side}wr"][0] += sign * lift * 0.8
    else:  # bouncing jump with wide legs
        bounce = -0.040 * abs(wave)
        for point in joints.values():
            point[1] += bounce
        for side, sign in (("l", -1.0), ("r", 1.0)):
            joints[f"{side}kne"][0] += sign * 0.030 * abs(wave)
            joints[f"{side}ank"][0] += sign * 0.050 * abs(wave)
    return {name: (float(point[0]), float(point[1])) for name, point in joints.items()}


def render_frame(motion_class: int, phase: float, width: int, height: int) -> np.ndarray:
    frame = np.full((height, width, 3), 24, dtype=np.uint8)
    joints = _motion_joints(motion_class, phase)
    points = {name: (int(x * width), int(y * height)) for name, (x, y) in joints.items()}
    for start, end in _BONES:
        cv2.line(frame, points[start], points[end], (220, 220, 220), 12, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(frame, point, 14, (240, 240, 240), -1)
    cv2.circle(frame, points["head"], max(18, width // 16), (240, 240, 240), -1)
    return frame


def write_video(
    path: PathLike,
    motion_class: int,
    num_frames: int,
    fps: int,
    width: int = 640,
    height: int = 480,
) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(destination.as_posix(), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {destination}.")
    try:
        for index in range(num_frames):
            frame = render_frame(motion_class, index / max(num_frames - 1, 1), width, height)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return destination


def write_audio(
    path: PathLike, frequency: float, duration: float = 1.0, sample_rate: int = 16000
) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    samples = int(duration * sample_rate)
    time = np.arange(samples, dtype=np.float64) / sample_rate
    envelope = np.sin(np.pi * time / duration) ** 2
    signal = (0.4 * envelope * np.sin(2.0 * np.pi * frequency * time) * 32767).astype(np.int16)
    with wave.open(destination.as_posix(), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(signal.tobytes())
    return destination


def write_image(path: PathLike, color: tuple[int, int, int], size: int = 96) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    ramp = np.linspace(0.35, 1.0, size, dtype=np.float32)
    image = np.zeros((size, size, 3), dtype=np.float32)
    for channel, value in enumerate(color):
        image[:, :, channel] = value * ramp[None, :]
    image = image.astype(np.uint8)
    cv2.circle(image, (size // 2, size // 2), size // 4, (245, 245, 245), -1)
    cv2.imwrite(destination.as_posix(), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return destination
