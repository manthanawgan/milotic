#Shared type vocabulary for the Dance Beyond Music scaffold.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt
import torch

Tensor: TypeAlias = torch.Tensor

PathLike: TypeAlias = str | Path

FrameArray: TypeAlias = npt.NDArray[np.uint8]

KeypointArray: TypeAlias = npt.NDArray[np.floating[Any]]

DatasetSplit: TypeAlias = Literal["train", "val", "test"]


@dataclass(slots=True)
class PoseResult:
    """Pose estimate for one video frame.

    Attributes:
        keypoints: Coordinate array shaped ``(N, 3)``; one x/y/z triplet per
            keypoint. Units are backend-defined (pixels or normalized space).
        confidence: Per-keypoint visibility scores shaped ``(N,)``, normally in
            ``[0, 1]``. Low values still carry information; do not discard them.
        frame_idx: Zero-based index of the source frame in the video.
        timestamp_ms: Presentation timestamp of the frame in milliseconds.
        backend_name: Name of the pose backend that produced the estimate, so
            downstream stages can invalidate caches when the backend changes.
    """

    keypoints: KeypointArray
    confidence: npt.NDArray[np.floating[Any]]
    frame_idx: int
    timestamp_ms: float
    backend_name: str


@dataclass(slots=True)
class PointCloud:
    """A temporal point cloud built from a sequence of pose estimates.

    Attributes:
        points: Array shaped ``(T, N, 3)``: ``T`` frames, ``N`` keypoints, and
            x/y/z coordinates for each point.
        timestamps: One timestamp in milliseconds per frame, length ``T``.
    """

    points: npt.NDArray[np.floating[Any]]
    timestamps: list[float]


class ModalityInput(TypedDict):
    """One sample of any modality, ready for encoding.

    Attributes:
        modality_name: Registered name of the modality, for example ``"text"``.
        data: The raw or preprocessed payload. Its type is modality-specific.
        metadata: Extra provenance such as source path, language, or license.
    """

    modality_name: str
    data: Any
    metadata: dict[str, Any]


class PairedPayload(TypedDict, total=False):
    """Payloads for one paired dance sample, keyed by modality name.

    Attributes:
        movement: Point-cloud array shaped ``(T, N, 3)``.
        text: Text content itself, not a file path.
        audio: Path to a WAV file.
        image: Path to an image file.
        spatial: Path to a ``.npy`` descriptor or point array.
    """

    movement: npt.NDArray[np.floating[Any]]
    text: str
    audio: str
    image: str
    spatial: str


class PairedBatch(TypedDict, total=False):
    """One collated batch of paired multimodal samples.

    Attributes:
        sample_ids: Stable identifier per row.
        splits: Dataset split per row.
        movement: Padded point clouds shaped ``(B, T, N, 3)``.
        movement_mask: Boolean ``(B, T)`` mask; ``True`` means a real frame.
        text: Text content per row, or ``None`` when absent.
        audio: WAV paths per row, or ``None`` when absent.
        image: Image paths per row, or ``None`` when absent.
        spatial: Spatial descriptor paths per row, or ``None`` when absent.
        modality_names: Sorted modality names present in the batch.
    """

    sample_ids: list[str]
    splits: list[str]
    movement: Tensor
    movement_mask: Tensor
    text: list[str | None]
    audio: list[str | None]
    image: list[str | None]
    spatial: list[str | None]
    modality_names: list[str]


class EmbeddingBatch(TypedDict):
    """A batch of embeddings produced by one or more encoders.

    Attributes:
        embeddings: Float tensor shaped ``(B, D)`` where ``B`` is batch size and
            ``D`` is the shared embedding dimension.
        modality_names: Modality name for each row, length ``B``.
        sample_ids: Stable sample identifier for each row, length ``B``.
    """

    embeddings: Tensor
    modality_names: list[str]
    sample_ids: list[str]
