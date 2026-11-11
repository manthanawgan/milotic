from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

import dance_beyond_music.modalities  #noqa: F401  (registers every modality)
from dance_beyond_music.training.callbacks import (
    ModelCheckpointCallback,
    WandbMetadataCallback,
)
from dance_beyond_music.training.factory import (
    build_logger,
    build_module,
    build_trainer,
)
from dance_beyond_music.training.lit_datamodule import LitDataModule
from dance_beyond_music.utils import env
from dance_beyond_music.utils.io import ensure_dir
from dance_beyond_music.utils.logging import configure_logging, get_logger
from dance_beyond_music.utils.reproducibility import log_run_metadata, set_seed
from scripts import CONFIG_PATH

logger = get_logger(__name__)


def train(cfg: DictConfig) -> Path | None:
    OmegaConf.resolve(cfg)
    set_seed(int(cfg.training.seed))
    output_dir = ensure_dir(cfg.output_dir)
    log_run_metadata(
        {"stage": "train", "config": OmegaConf.to_container(cfg, resolve=True)}, output_dir
    )

    module = build_module(cfg)
    datamodule = LitDataModule(
        data_root=str(cfg.data.root),
        batch_size=int(cfg.training.batch_size),
        num_workers=int(cfg.training.num_workers),
    )
    checkpoint = ModelCheckpointCallback(output_dir / "checkpoints")
    callbacks = [WandbMetadataCallback(output_dir), checkpoint]
    trainer = build_trainer(cfg, callbacks, build_logger(cfg))
    logger.info("Fitting %s for %d epoch(s)", cfg.experiment.name, trainer.max_epochs)
    trainer.fit(module, datamodule=datamodule)
    if checkpoint.best_path is not None:
        logger.info(
            "Best checkpoint: %s (%s=%.5f)",
            checkpoint.best_path,
            checkpoint.monitor,
            checkpoint.best_score,
        )
    return checkpoint.best_path


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="config")
def main(cfg: DictConfig) -> None:
    configure_logging()
    device = env.get_device()
    logger.info("Training starting on %s", device)
    train(cfg)


if __name__ == "__main__":
    main()
