"""This is the starting point for the project."""

import logging

import hydra
from omegaconf import DictConfig

from src.exploration.exploration import test_qwen, test_fashion_gen

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Launch the current main task for the project."""
    test_fashion_gen(cfg)


if __name__ == "__main__":
    main()
