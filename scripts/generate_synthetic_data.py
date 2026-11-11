#Create a tiny synthetic multimodal dataset for testing and smoke runs.

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from dance_beyond_music.utils.logging import configure_logging, get_logger
from dance_beyond_music.utils.synthetic import generate_synthetic_dataset
from scripts import CONFIG_PATH

logger = get_logger(__name__)


def generate(cfg: DictConfig) -> dict[str, object]:
    synthetic = cfg.data.synthetic
    manifest = generate_synthetic_dataset(
        root=cfg.data.root,
        num_samples=int(synthetic.num_samples),
        num_frames=int(synthetic.num_frames),
        fps=int(synthetic.fps),
        image_size=int(cfg.data.image_size),
        render_videos=bool(synthetic.render_videos),
        write_ground_truth=bool(synthetic.write_ground_truth),
        seed=int(cfg.training.seed),
    )
    logger.info("Synthetic dataset ready at %s", cfg.data.root)
    return manifest


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging()
    generate(cfg)


if __name__ == "__main__":
    main()
