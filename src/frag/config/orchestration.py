"""Configuration for the Agent Orchestration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, FilePath, PostgresDsn, computed_field, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DynamoDBCheckpointerConfig(BaseSettings):
    """Config for dynamo db checkpointer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
        env_ignore_empty=True,
        validate_default=True,
    )

    # this value must match whatever is used when setting up the cloudformation table, so please ensure that we use the .env value when deploying the cloudformation template for the dynamo db langgraph checkpointer
    table_name: str = Field(validation_alias="env__aws_dynamodb_checkpointer_table_name")

    region_name: str = "us-east-1"

    # it stays for a day
    ttl_seconds: int = 60 * 60 * 24 * 1

    # whether or not to compress the entries in the dynamo db table
    do_compression: bool = True

    max_pool_size: int = 20

    # legacy — The original boto3 behaviour. Uses a fixed exponential backoff with jitter. Has a narrower set of retryable error codes. This is the default if you set nothing at all.
    # standard — A modernised, consistent retry policy that AWS introduced to align all SDKs. It retries on a wider set of transient errors and throttles. Defaults to 3 max attempts unless overridden. This is the recommended baseline for most production workloads.
    # adaptive — Builds on standard but adds client-side rate limiting. The client tracks the rate of throttling responses from AWS and proactively slows down its own request rate before AWS starts rejecting calls — similar to a token bucket on the client side. This is what the DynamoDB checkpointer example uses because LangGraph agents can burst heavily, and adaptive mode prevents a thundering-herd of retries from making throttling worse.
    retry_mode: Literal["legacy", "standard", "adaptive"] = "adaptive"

    # max_attempts — The maximum number of retry attempts after the initial request.
    max_retry_attempts: int = 3


class PostgresCheckpointerConfig(BaseSettings):
    """Config for the postgres checkpointer."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
        env_ignore_empty=True,
        validate_default=True,
    )

    # max simultaneous connections - this is for the postgres connection pool
    max_pool_size: int = 20

    # from .env
    postgres_user: str = Field(validation_alias="env__postgres_user")
    postgres_password: str = Field(validation_alias="env__postgres_password")
    postgres_db: str = Field(validation_alias="env__postgres_db")

    # constructing dsn from env vars
    # we are returning actual PostgresDsn since the type is a url subclass not string
    @computed_field
    @property
    def dsn(self) -> PostgresDsn:
        return PostgresDsn(f"postgresql://{self.postgres_user}:{self.postgres_password}@localhost:5432/{self.postgres_db}")


class SqliteCheckpointerConfig(BaseModel):
    """Config for SQLite Checkpointer."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    db_path: FilePath = "data/pipeline_checkpoints.db"


class CheckpointerConfig(BaseModel):
    """Config for the langgraph checkpointer."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    backend: Literal["postgres", "sqlite", "dynamodb"] = "dynamodb"
    sqlite: SqliteCheckpointerConfig = SqliteCheckpointerConfig()
    postgres: PostgresCheckpointerConfig = PostgresCheckpointerConfig()
    dynamodb: DynamoDBCheckpointerConfig = DynamoDBCheckpointerConfig()


class MCPConfig(BaseModel):
    """Config for MCP use in Agent Orchestration."""
    model_config = ConfigDict(frozen=True, validate_default=True)

    llm_tool_names: list[str] = ["semantic_search", "get_product_categories"]
    db_tool_name: str = "get_datapoint_by_index"


class AgentOrchestrationConfig(BaseModel):
    """Config for Agent Orchestration."""

    model_config = ConfigDict(frozen=True, validate_default=True)

    checkpointer: CheckpointerConfig = CheckpointerConfig()
    mcp: MCPConfig = MCPConfig()

    # this is to determine the number of recommendations to return to the user if the user does not specify how many recommendations they want
    default_num_recommendations: int = 3

    # this is how many times we will try to generate the correct number of recommendations - in the modifier node
    num_recommendation_attempts: int = 3

    # this is the maximum number of times the critique node is allowed to critique the output of the recommendation node
    max_num_critiques: int = 3

    # this is where the images fetched from the fashion gen database for evaluation are stored. the folder is deleted at the end when running the app, but not deleted when running the cli_main.py file
    temporary_images_folder: str = "data/temp_images/"

    # this is where the agent's langgraph orchestration diagram is stored
    node_diagram_path: str = "fashion_agent_diagram.png"
