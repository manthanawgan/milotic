from __future__ import annotations

import hydra
from omegaconf import DictConfig

import dance_beyond_music.modalities  #noqa: F401  (registers every modality) ;>
from dance_beyond_music.evaluation.runner import evaluate_checkpoint
from dance_beyond_music.utils.logging import configure_logging, get_logger
from scripts import CONFIG_PATH

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging()
    logger.info("Evaluation starting on %s", cfg.evaluation.split)
    evaluate_checkpoint(cfg)


if __name__ == "__main__":
    main()
