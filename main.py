"""This is the starting point for the project."""

import logging

import hydra
from omegaconf import DictConfig

from src.exploration.exploration import test_qwen

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Launch the current main task for the project."""
    test_qwen(cfg)


if __name__ == "__main__":
    main()
