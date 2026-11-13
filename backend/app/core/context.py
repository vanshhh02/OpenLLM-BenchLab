"""Request and worker execution context for workspace and user scoping."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

DEFAULT_WORKSPACE_ID = "ws_default"

_current_workspace_id: ContextVar[str] = ContextVar(
    "benchlab_workspace_id", default=DEFAULT_WORKSPACE_ID
)
_current_user_id: ContextVar[Optional[str]] = ContextVar(
    "benchlab_user_id", default=None
)


def get_current_workspace_id() -> str:
    """Return the active workspace ID for the current request or worker."""
    return _current_workspace_id.get()


def get_current_user_id() -> Optional[str]:
    """Return the active user ID for the current request or worker, if any."""
    return _current_user_id.get()


@contextmanager
def workspace_scope(
    workspace_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Iterator[None]:
    """Temporarily bind workspace_id and user_id in the current thread/async context."""
    ws_token = _current_workspace_id.set(workspace_id or DEFAULT_WORKSPACE_ID)
    user_token = _current_user_id.set(user_id)
    try:
        yield
    finally:
        _current_workspace_id.reset(ws_token)
        _current_user_id.reset(user_token)

