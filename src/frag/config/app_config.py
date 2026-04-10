"""Contains the configuration for the whole application."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from frag.config.envs import EnvSettings
from frag.config.models import ModelsConfig
from frag.config.exploration import ExplorationConfig
from frag.config.eval import EvaluationConfig
from frag.config.tracking import TrackingConfig
from frag.config.prompts import PromptsSetup
from frag.config.orchestration import AgentOrchestrationConfig
from frag.config.data import DataConfig
from frag.config.logs import LogConfig
from frag.config.aws_parameter_store import AWSParamStoreConfig


class ApplicationConfig(BaseSettings):
    """Config for the full application."""

    model_config = SettingsConfigDict(
        frozen=True,
        validate_default=True,
        case_sensitive=False,
        env_nested_delimiter='__',
    )

    env: EnvSettings = Field(default_factory=EnvSettings)
    data: DataConfig = Field(default_factory=DataConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    exploration: ExplorationConfig = Field(default_factory=ExplorationConfig)
    prompts: PromptsSetup = Field(default_factory=PromptsSetup)
    orchestration: AgentOrchestrationConfig = Field(default_factory=AgentOrchestrationConfig)
    logs: LogConfig = Field(default_factory=LogConfig)
    aws_param_store: AWSParamStoreConfig = Field(default_factory=AWSParamStoreConfig)

