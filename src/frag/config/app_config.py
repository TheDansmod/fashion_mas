"""Contains the configuration for the whole application."""

from functools import lru_cache

from pydantic import BaseModel, ConfigDict

from frag.config.envs import EnvSettings
from frag.config.models import ModelsConfig
from frag.config.exploration import ExplorationConfig
from frag.config.eval import EvaluationConfig
from frag.config.tracking import TrackingConfig
from frag.config.prompts import PromptsSetup
from frag.config.orchestration import AgentOrchestrationConfig
from frag.config.data import DataConfig
from frag.config.logs import LogConfig


class ApplicationConfig(BaseModel):
    """Config for the full application."""

    model_config = ConfigDict(frozen=True)

    env: EnvSettings = EnvSettings()
    data: DataConfig = DataConfig()
    models: ModelsConfig = ModelsConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    tracking: TrackingConfig = TrackingConfig()
    exploration: ExplorationConfig = ExplorationConfig()
    prompts: PromptsSetup = PromptsSetup()
    orchestration: AgentOrchestrationConfig = AgentOrchestrationConfig()
    logs: LogConfig = LogConfig()
