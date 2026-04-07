"""Contains the configuration for the whole application."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from frag.config.envs import EnvSettings
from frag.config.models import ModelsConfig
from frag.config.exploration import ExplorationConfig
from frag.config.eval import EvaluationConfig
from frag.config.tracking import TrackingConfig
from frag.config.prompts import PromptsSetup
from frag.config.orchestration import AgentOrchestrationConfig
from frag.config.data import DataConfig

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

