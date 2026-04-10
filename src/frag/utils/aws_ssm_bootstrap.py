# this file was written by claude sonnet 4.6 thinking
import os
from loguru import logger as log
import boto3
from botocore.config import Config
from botocore.exceptions import (
    NoCredentialsError,
    EndpointConnectionError,
    EndpointResolutionError,
    NoRegionError,
)

def _bootstrap_ssm(path: str = "/frag/") -> None:
    """
    Fetch all params under the SSM path prefix and inject as env vars.

    Precedence (highest → lowest):
      1. Existing env vars (Lambda/local .env vars always win)
      2. AWS SSM Parameter Store
      3. .env file / pydantic default values

    Silently skips when AWS is unreachable (local offline dev) — the
    fast_connect_timeout ensures this fails in ~2 s, not 60+ s.
    """
    log.success("IN BOOTSTRAP FUNCTION. RETURNING.")
    return
    try:
        ssm = boto3.client(
            "ssm",
            region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
            config=Config(
                connect_timeout=2,          # fail fast locally
                retries={"max_attempts": 1},
            ),
        )
        paginator = ssm.get_paginator("get_parameters_by_path")
        log.success("GOT PAGINATOR")
        loaded = 0
        for page in paginator.paginate(Path=path, WithDecryption=True):
            log.success("GOT SOME PARAMS")
            for param in page["Parameters"]:
                # /fashion-agent/database__host  →  DATABASE__HOST
                # Also handles accidental slash-nesting: /fashion-agent/db/host → DB__HOST
                key = (
                    param["Name"]
                    .removeprefix(path)   # strip leading prefix
                    .upper()
                    .replace("/", "__")   # slash-nesting → pydantic nested delimiter
                )
                if key not in os.environ:  # existing vars (Lambda / .env) always win
                    # we don't add an envvar prefix since that is just pollution and I need to actually write envvars in .env file with that prefix. if it was just through this setup then it might be worth it, but otherwise not really
                    os.environ[key] = param["Value"]
                    loaded += 1
        log.debug(f"SSM bootstrap: loaded {loaded} parameters from {path}")

    except (NoCredentialsError, EndpointConnectionError, EndpointResolutionError, NoRegionError):
        # Running locally without AWS access — .env / defaults will cover it
        log.debug("SSM bootstrap: skipped (no AWS access). Falling back to .env / defaults.")
    except Exception as exc:
        # Catch-all: permission errors, throttling, etc.
        # Log as warning so you notice it in prod without hard-crashing at import time.
        log.warning("SSM bootstrap: unexpected error (%s: %s). Falling back.", type(exc).__name__, exc)

