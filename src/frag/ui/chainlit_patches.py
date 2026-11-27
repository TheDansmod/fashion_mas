"""Workarounds for two Chainlit bugs; checked against Chainlit 2.12.0.

Both are still present in 2.12.0 (checked on 2026-11-27). Re-check when upgrading
and drop the patch once a release fixes it.
"""

import string

from chainlit import secret
from chainlit.data.storage_clients.s3 import S3StorageClient
from loguru import logger as log

# the OAuth `state` parameter is built from secret.chars, whose punctuation AWS
# Cognito rejects in redirects:
# https://github.com/Chainlit/chainlit/issues/2707
# https://github.com/Chainlit/chainlit/issues/972
_URL_SAFE_CHARS = string.ascii_letters + string.digits + "-_"

_applied = False


async def _close_s3_client(self: S3StorageClient) -> None:
    # Chainlit awaits boto3's close(), which is synchronous and returns None, so
    # shutting down raised a TypeError; call it without awaiting
    try:
        if getattr(self, "client", None):
            self.client.close()
    except Exception:  # noqa: BLE001  closing at shutdown must not raise
        log.opt(exception=True).warning("Failed to close the Chainlit S3 client.")


def apply() -> None:
    """Apply the patches; calling it again does nothing."""
    global _applied  # noqa: PLW0603  a once-only flag is the point
    if _applied:
        return
    secret.chars = _URL_SAFE_CHARS
    S3StorageClient.close = _close_s3_client
    _applied = True
