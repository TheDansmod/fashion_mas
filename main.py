"""This is the starting point for the project."""

import logging

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from src.data_manager.vector_db_writer import populate_vector_db
from src.utils.common_utils import validate_hydra_config

# The .env file should contain `HYDRA_FULL_ERROR=1` to see a full stacktrace in case
# of error.
load_dotenv()
log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    """Launch the current main task for the project."""
    validate_hydra_config(cfg)
    populate_vector_db(cfg)
    # test_fashion_gen(cfg)


if __name__ == "__main__":
    main()
