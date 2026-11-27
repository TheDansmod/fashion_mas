"""Setup and confirmation shared by the scripts in this folder (cleanup.md §3.2).

The scripts run by path (`uv run python scripts/<name>.py`), which puts this folder
on `sys.path`, so they import this module as `_common`.
"""

from dotenv import load_dotenv

from frag.logging_setup import setup_logging
from frag.settings import Settings, get_settings


def start(service: str) -> Settings:
    """Load `.env`, read the settings and set up logging for a script."""
    # makes .env visible to boto3 and other libraries that read os.environ
    load_dotenv()
    settings = get_settings()
    setup_logging(
        settings.logging,
        level=settings.log_level,
        is_dev=settings.is_dev,
        service=service,
    )
    return settings


def confirm(question: str, *, assume_yes: bool = False) -> bool:
    """Ask before something destructive or billable; only a typed YES agrees.

    `assume_yes` (a script's `--yes` flag) skips the prompt, for when the user has
    already agreed to the cost.
    """
    if assume_yes:
        return True
    return input(f"{question}\nType YES to continue: ").strip() == "YES"
