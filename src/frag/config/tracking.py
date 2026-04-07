"""Configuration for tracking."""
from pydantic import BaseModel, ConfigDict, FilePath

class TrackingConfig(BaseModel):
    """Config for Tracking."""
    model_config = ConfigDict(frozen=True)

    # for local token usage tracking
    token_usage_tracker_path: FilePath = 'data/token_usage.csv'
