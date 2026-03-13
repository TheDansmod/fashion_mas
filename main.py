"""This is the starting point for the project."""

import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.exploration.mistral_exploration import check_image_inputs
from src.utils.common_utils import validate_hydra_config

# The .env file should contain `HYDRA_FULL_ERROR=1` to see a full stacktrace in case
# of error.
# The .env file should also have the HF_TOKEN value from huggingface for vision model
# access.
# The .env file should populate langsmith endpoints like `LANGSMITH_TRACING=true`,
# `LANGSMITH_PROJECT=<project_name>`, `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT=<eu/us>`.
# The .env file should have a key for google AI api calls: `GOOGLE_API_KEY=<key>`
load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Launch the current main task for the project."""
    validate_hydra_config(cfg)
    check_image_inputs(cfg)


if __name__ == "__main__":
    main()
