"""Configuration for the Agent Orchestration."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, FilePath, PostgresDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

class PostgresCheckpointerConfig(BaseSettings):
    """Config for the postgres checkpointer."""
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra="ignore", frozen=True, env_ignore_empty=True)

    # max simultaneous connections - this is for the postgres connection pool
    max_pool_size: int = 20

    # from .env file
    postgres_user: str
    postgres_password: str
    postgres_db: str
    
    # constructing dsn from env vars
    @computed_field
    @property
    def dsn(self) -> PostgresDsn:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@localhost:5432/{self.postgres_db}"

class SqliteCheckpointerConfig(BaseModel):
    """Config for SQLite Checkpointer."""
    model_config = ConfigDict(frozen=True)

    db_path: FilePath = 'data/pipeline_checkpoints.db'

class CheckpointerConfig(BaseModel):
    """Config for the langgraph checkpointer."""
    model_config = ConfigDict(frozen=True)

    backend: Literal["postgres", "sqlite", "dynamodb"] = "postgres"
    sqlite: SqliteCheckpointerConfig = SqliteCheckpointerConfig()
    postgres: PostgresCheckpointerConfig = PostgresCheckpointerConfig()

class MCPConfig(BaseModel):
    """Config for MCP use in Agent Orchestration."""
    llm_tool_names: list[str] = ['semantic_search', 'get_product_categories']
    db_tool_name: str = 'get_datapoint_by_index'

class AgentOrchestrationConfig(BaseModel):
    """Config for Agent Orchestration."""
    model_config = ConfigDict(frozen=True)

    checkpointer: CheckpointerConfig = CheckpointerConfig()
    mcp: MCPConfig = MCPConfig()

    # this is to determine the number of recommendations to return to the user if the user does not specify how many recommendations they want
    default_num_recommendations: int = 3
    
    # this is how many times we will try to generate the correct number of recommendations - in the modifier node
    num_recommendation_attempts: int = 3
    
    # this is the maximum number of times the critique node is allowed to critique the output of the recommendation node
    max_num_critiques: int = 3

    # this is where the images fetched from the fashion gen database for evaluation are stored. the folder is deleted at the end when running the app, but not deleted when running the cli_main.py file
    temporary_images_folder: str = 'data/temp_images/'
    
    # this is where the agent's langgraph orchestration diagram is stored
    node_diagram_path: str = 'fashion_agent_diagram.png'
