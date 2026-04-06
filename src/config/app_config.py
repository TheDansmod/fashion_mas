"""Contains the configuration for the whole application."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from src.config.envs import EnvSettings
from src.config.models import ModelsConfig
from src.config.exploration import ExplorationConfig
from src.config.eval import EvaluationConfig
from src.config.tracking import TrackingConfig
from src.config.prompts import PromptsSetup
from src.config.orchestration import AgentOrchestrationConfig
from src.config.data import DataConfig

class ApplicationConfig(BaseSettings):
    """Config for the full application."""
    model_config = SettingsConfigDict(frozen=True)

    env: EnvSettings = EnvSettings()
    data: DataConfig = DataConfig()
    models: ModelsConfig = ModelsConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    tracking: TrackingConfig = TrackingConfig()
    exploration: ExplorationConfig = ExplorationConfig()
    prompts: PromptsSetup = PromptsSetup()
    orchestration: AgentOrchestrationConfig = AgentOrchestrationConfig()

