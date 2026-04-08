"Config variables derived from .env file."

from enum import StrEnum

from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppEnv(StrEnum):
    DEV = "development"
    STG = "staging"
    PROD = "production"


class EnvSettings(BaseSettings):
    """Captures variables from .env file.

    This also helps enforce that these values must be present in the env,
    and that no other values are present in the .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="forbid",
        env_ignore_empty=True,
    )

    # for fetching the embedding model
    hf_token: str

    # for langsmith traceability
    langsmith_tracing: bool
    langsmith_project: str
    langsmith_api_key: str
    langsmith_endpoint: HttpUrl

    # for using mistral models
    mistral_api_key: str

    # for postgres checkpointer
    postgres_user: str
    postgres_password: str
    postgres_db: str

    # for env mode
    app_env: AppEnv

    # for the aws tables and buckets etc
    aws_s3_chainlit_persistence_bucket_name: str
    aws_dynamodb_chainlit_persistence_table_name: str
    aws_dynamodb_checkpointer_table_name: str
