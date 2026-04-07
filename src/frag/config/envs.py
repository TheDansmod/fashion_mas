"Config variables derived from .env file."""
from pydantic import HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict

class EnvSettings(BaseSettings):
    """Captures variables from .env file.

    This also helps enforce that these values must be present in the env.
    """
    # TODO: do you want to do extra=forbid?
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', frozen=True, extra="ignore", env_ignore_empty=True)

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
