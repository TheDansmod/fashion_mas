# monkey patch since in the redirect state parameter chainlit uses characters that are invalid for aws cognito
# https://github.com/Chainlit/chainlit/issues/2707
# https://github.com/Chainlit/chainlit/issues/972
from chainlit import secret as _cl_secret  # the secret.py file in the .venv folder
import string

_cl_secret.chars = string.ascii_letters + string.digits + "-_"

from typing import Optional

import chainlit as cl
import chainlit.data as cl_data
from chainlit.data.dynamodb import DynamoDBDataLayer


@cl.data_layer
def get_data_layer():
    return DynamoDBDataLayer(table_name="fashion_mas_table")


@cl.oauth_callback
def oauth_callback(
    provider_id: str,
    token: str,
    raw_user_data: dict[str, str],
    default_user: cl.User,
) -> Optional[cl.User]:
    if provider_id == "aws-cognito":
        sub = raw_user_data.get("sub")  # this is mandatory
        email = raw_user_data.get("email", None)
        username = raw_user_data.get("username", None)
        return cl.User(
            identifier=sub,  # stable, unique per user even if they change email etc
            metadata={
                "email": email,
                "provider": provider_id,
                "username": username,
                **default_user.metadata,
            },
        )
    return default_user


@cl.on_chat_start
async def on_chat_start() -> None:
    user: cl.User = cl.user_session.get("user")

    # user_session is in-memory, per-connection — use it for transient state
    cl.user_session.set("message_count", 0)

    await cl.Message(
        content=(
            f"👋 Hello **{user.metadata['username']}** \n\n"
            "Every message is **persisted** to `AWS DynamoDB`. "
            "Refresh the page — your chat history will appear in the sidebar.\n\n"
            "Try sending a few messages, then reload!"
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Fires every time the user sends a message."""
    count = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", count)
    user: cl.User = cl.user_session.get("user")

    await cl.Message(
        content=f"**Echo #{count}** from `{user.metadata['username']}`: {message.content}"
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: cl.types.ThreadDict) -> None:
    """
    Fires when the user clicks a past conversation in the sidebar.
    Re-populate any in-memory session state you need.
    """
    prior = [s for s in thread.get("steps", []) if s.get("type") == "user_message"]
    cl.user_session.set("message_count", len(prior))

    await cl.Message(
        content=(
            f"🔁 Resumed thread `{thread['id'][:8]}…` "
            f"with {len(prior)} prior message(s)."
        )
    ).send()
