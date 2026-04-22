"""Config for loguru logging setup"""

from pydantic import computed_field, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from frag.config.envs import AppEnv


class LogConfig(BaseSettings):
    """Config for Loguru Logging"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="ignore",
        env_ignore_empty=True,
        validate_default=True,
    )

    # from .env
    app_env: AppEnv = Field(validation_alias="app_env")

    write_human_readable_logs: bool = True
    write_machine_readable_logs: bool = True
    write_console_logs: bool = True

    # these are for the main process - the rag agent
    human_readable_log_file: str = "logs/human_readable/frag.log"
    machine_readable_log_file: str = "logs/machine_readable/frag.log"

    # these are for the mcp server process when it is running independently
    mcp_human_readable_log_file: str = "logs/human_readable/frag_mcp.log"
    mcp_machine_readable_log_file: str = "logs/machine_readable/frag_mcp.log"

    # the rotation interval could be a lot of things like 100 MB, 1 month 2 weeks, 2 days, 10h, monthly, 18:00, sunday, monday at 12:00
    rotation_interval: str = "1 day"

    # the retention period can also be many things like number of log files to keep (int), datetime.timedelta for max age of files, str for max age of files, callable (see docs)
    retention_period: str = "30 days"

    # this could be zip, gz, xz, tar, tar.gz, tar.xz, lzma, etc (see docs for rest)
    compression_method: str = "zip"

    @computed_field
    @property
    def log_level(self) -> str:
        if self.app_env in [AppEnv.DEV, AppEnv.STG]:
            return "DEBUG"
        elif self.app_env == AppEnv.PROD:
            return "INFO"
        else:
            raise ValueError(
                "App Environment should be either development, staging, or production."
            )

    @computed_field
    @property
    def is_dev(self) -> bool:
        if self.app_env == AppEnv.DEV:
            return True
        return False
