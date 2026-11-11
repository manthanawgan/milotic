from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from dance_beyond_music.pipeline.pairing import ModalityPairer
from dance_beyond_music.pipeline.sources import MODALITIES, load_sources
from dance_beyond_music.utils import env
from dance_beyond_music.utils.io import save_json
from dance_beyond_music.utils.logging import configure_logging, get_logger
from scripts import CONFIG_PATH

logger = get_logger(__name__)

def build_dataset(cfg: DictConfig) -> dict[str, int]:
    OmegaConf.resolve(cfg)
    root = Path(str(cfg.data.root))
    pairer = ModalityPairer()
    samples: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}

    for source in load_sources(root):
        sample_id = str(source.get("sample_id", "")).strip()
        if not sample_id:
            logger.warning("Skipping source without sample_id: %s", source)
            continue
        split = str(source.get("split", "train"))
        movement = root / "processed" / "point_clouds" / f"{sample_id}.npz"
        if not movement.is_file():
            logger.warning("Skipping '%s': missing point cloud %s", sample_id, movement)
            continue
        modalities: dict[str, str] = {}
        for name in MODALITIES:
            relative = source.get("paths", {}).get(name)
            if not relative:
                continue
            if not (root / relative).is_file():
                logger.warning("Skipping %s file for '%s': %s", name, sample_id, relative)
                continue
            modalities[name] = str(relative)
            paired = pairer.pair(
                sample_id,
                {
                    "modality_name": name,
                    "data": None,
                    "metadata": {
                        "split": split,
                        "source": source.get("source", "synthetic"),
                        "synthetic_only": bool(source.get("synthetic_only", True)),
                        "path": relative,
                    },
                },
            )
            pairer.validate_pair(paired)
            pairs.append(
                {
                    "movement_sample_id": sample_id,
                    "modality_name": name,
                    "path": str(relative),
                    "split": split,
                    "metadata": paired["metadata"],
                }
            )
        samples.append(
            {
                "sample_id": sample_id,
                "split": split,
                "motion_class": source.get("motion_class"),
                "movement": movement.relative_to(root).as_posix(),
                "modalities": modalities,
                "synthetic_only": bool(source.get("synthetic_only", True)),
            }
        )
        split_counts[split] = split_counts.get(split, 0) + 1

    if not samples:
        raise RuntimeError(
            f"No complete samples found under {root}. Run extract_poses (or the synthetic "
            "generator with write_ground_truth=true) first."
        )
    manifest = {
        "version": 1,
        "synthetic_only": all(sample["synthetic_only"] for sample in samples),
        "samples": samples,
        "split_counts": split_counts,
        "modalities": sorted({name for sample in samples for name in sample["modalities"]}),
    }
    save_json(manifest, str(cfg.data.manifest))
    save_json({"version": 1, "pairs": pairs}, str(cfg.data.pairs))
    logger.info(
        "Wrote %d samples and %d pairs; splits=%s, modalities=%s",
        len(samples),
        len(pairs),
        split_counts,
        manifest["modalities"],
    )
    return split_counts


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging()
    device = env.get_device()
    logger.info("Dataset build starting on %s", device)
    build_dataset(cfg)


if __name__ == "__main__":
    main()
