"""Authentication, password hashing, HS256 JWT tokens, API keys, and workspace RBAC."""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import logging
import os
import secrets
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, Iterable, List, Optional, Tuple

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.context import DEFAULT_WORKSPACE_ID, workspace_scope
from backend.app.database.models import (
    ApiKeyRecord,
    UserRecord,
    WorkspaceMemberRecord,
    WorkspaceRecord,
)
from backend.app.database.session import ensure_default_workspace, get_db

logger = logging.getLogger("benchlab.security")

_DEFAULT_DEV_SECRET = "benchlab-dev-secret-change-in-production"
_PBKDF2_ITERATIONS = 310_000
_ROLE_HIERARCHY = {"viewer": 1, "member": 2, "admin": 3, "owner": 4}


def get_secret_key() -> str:
    """Return signing key for JWT tokens."""
    secret = os.getenv("BENCHLAB_SECRET_KEY", "").strip()
    if not secret:
        return _DEFAULT_DEV_SECRET
    return secret


def is_auth_required() -> bool:
    """Return True when strict authentication is enforced across all API routes."""
    return os.getenv("BENCHLAB_AUTH_REQUIRED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def get_access_token_ttl_minutes() -> int:
    return int(os.getenv("BENCHLAB_ACCESS_TOKEN_MINUTES", "60"))


def get_refresh_token_ttl_days() -> int:
    return int(os.getenv("BENCHLAB_REFRESH_TOKEN_DAYS", "7"))


# -------------------------------------------------------------------------
# Password Hashing (PBKDF2-HMAC-SHA256)
# -------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hash a plaintext password using PBKDF2-HMAC-SHA256 with a random 16-byte salt."""
    if not password or len(password) < 6:
        raise ValueError("Password must be at least 6 characters long.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        _PBKDF2_ITERATIONS,
    )
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2-HMAC-SHA256 hash."""
    try:
        scheme, iter_str, salt_hex, hash_hex = stored_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iter_str)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False

    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )
    return hmac.compare_digest(candidate, expected)


# -------------------------------------------------------------------------
# RFC 7519 HS256 JWT Utilities
# -------------------------------------------------------------------------
def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)


def create_jwt_token(
    claims: Dict[str, Any],
    expires_delta: datetime.timedelta,
    secret_key: Optional[str] = None,
) -> str:
    """Create a signed HS256 JWT token."""
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        **claims,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "jti": uuid.uuid4().hex,
    }
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = _b64url_encode(
        json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    payload_b64 = _b64url_encode(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    key = (secret_key or get_secret_key()).encode("utf-8")
    sig = hmac.new(key, signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"


def decode_jwt_token(
    token: str,
    secret_key: Optional[str] = None,
    expected_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Decode and verify an HS256 JWT token, checking signature and expiration."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed JWT token.")

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    key = (secret_key or get_secret_key()).encode("utf-8")
    expected_sig = hmac.new(key, signing_input, hashlib.sha256).digest()

    try:
        actual_sig = _b64url_decode(sig_b64)
        header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise ValueError("Invalid JWT encoding.") from exc

    if header.get("alg") != "HS256":
        raise ValueError("Unsupported JWT algorithm.")

    if not hmac.compare_digest(actual_sig, expected_sig):
        raise ValueError("Invalid JWT signature.")

    exp = payload.get("exp")
    now_ts = int(datetime.datetime.now(datetime.timezone.utc).timestamp())
    if exp is None or int(exp) < now_ts:
        raise ValueError("JWT token has expired.")

    if expected_type and payload.get("type") != expected_type:
        raise ValueError(f"Expected JWT type '{expected_type}'.")

    return payload


def create_access_token(
    user_id: str,
    email: str,
    workspace_id: str,
    role: str = "member",
) -> str:
    return create_jwt_token(
        claims={
            "sub": user_id,
            "email": email,
            "workspace_id": workspace_id,
            "role": role,
            "type": "access",
        },
        expires_delta=datetime.timedelta(minutes=get_access_token_ttl_minutes()),
    )


def create_refresh_token(user_id: str, email: str, workspace_id: str) -> str:
    return create_jwt_token(
        claims={
            "sub": user_id,
            "email": email,
            "workspace_id": workspace_id,
            "type": "refresh",
        },
        expires_delta=datetime.timedelta(days=get_refresh_token_ttl_days()),
    )


# -------------------------------------------------------------------------
# API Key Utilities
# -------------------------------------------------------------------------
def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> Tuple[str, str, str]:
    """Generate a prefixed API key, returning (raw_key, key_prefix, key_hash)."""
    prefix = secrets.token_hex(4)
    secret_part = secrets.token_urlsafe(24)
    raw_key = f"blab_{prefix}_{secret_part}"
    return raw_key, f"blab_{prefix}", hash_api_key(raw_key)


# -------------------------------------------------------------------------
# Request Context & RBAC Dependencies
# -------------------------------------------------------------------------
@dataclass
class RequestContext:
    """Resolved authentication and workspace context for the active request."""

    workspace_id: str = DEFAULT_WORKSPACE_ID
    user: Optional[UserRecord] = None
    user_id: Optional[str] = None
    workspace_role: str = "owner"
    auth_method: str = "anonymous"
    scopes: List[str] = field(default_factory=lambda: ["*"])

    def has_role(self, min_role: str) -> bool:
        if self.user and self.user.is_superuser:
            return True
        return _ROLE_HIERARCHY.get(self.workspace_role, 0) >= _ROLE_HIERARCHY.get(min_role, 99)


def _resolve_api_key_context(
    raw_key: str,
    requested_workspace_id: Optional[str],
    db: Session,
) -> RequestContext:
    key_record = (
        db.query(ApiKeyRecord)
        .filter(ApiKeyRecord.key_hash == hash_api_key(raw_key))
        .first()
    )
    if not key_record or not key_record.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or inactive API key.",
        )

    now = datetime.datetime.utcnow()
    if key_record.expires_at and key_record.expires_at < now:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key has expired.",
        )

    user = db.query(UserRecord).filter_by(id=key_record.user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key owner account is inactive.",
        )

    target_ws_id = requested_workspace_id or key_record.workspace_id
    if requested_workspace_id and requested_workspace_id != key_record.workspace_id and not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="API key is not authorized for the requested workspace.",
        )

    membership = (
        db.query(WorkspaceMemberRecord)
        .filter_by(workspace_id=target_ws_id, user_id=user.id)
        .first()
    )
    role = membership.role if membership else ("owner" if user.is_superuser else "member")

    key_record.last_used_at = now
    db.commit()

    return RequestContext(
        workspace_id=target_ws_id,
        user=user,
        user_id=user.id,
        workspace_role=role,
        auth_method="api_key",
        scopes=list(key_record.scopes or ["*"]),
    )


def _resolve_jwt_context(
    token: str,
    requested_workspace_id: Optional[str],
    db: Session,
) -> RequestContext:
    try:
        payload = decode_jwt_token(token, expected_type="access")
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    user_id = payload.get("sub")
    user = db.query(UserRecord).filter_by(id=user_id).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    target_ws_id = requested_workspace_id or payload.get("workspace_id") or DEFAULT_WORKSPACE_ID
    membership = (
        db.query(WorkspaceMemberRecord)
        .filter_by(workspace_id=target_ws_id, user_id=user.id)
        .first()
    )

    if not membership and not user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"User does not have access to workspace '{target_ws_id}'.",
        )

    role = membership.role if membership else "owner"
    return RequestContext(
        workspace_id=target_ws_id,
        user=user,
        user_id=user.id,
        workspace_role=role,
        auth_method="jwt",
        scopes=["*"],
    )


async def get_request_context(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    x_workspace_id: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> AsyncGenerator[RequestContext, None]:
    """Resolve authenticated user/API key and workspace context, binding ContextVars."""
    ensure_default_workspace(db)

    raw_bearer: Optional[str] = None
    if authorization:
        parts = authorization.split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            raw_bearer = parts[1].strip()
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authorization header format. Expected 'Bearer <token>'.",
            )

    if x_api_key:
        ctx = _resolve_api_key_context(x_api_key.strip(), x_workspace_id, db)
    elif raw_bearer:
        if raw_bearer.startswith("blab_"):
            ctx = _resolve_api_key_context(raw_bearer, x_workspace_id, db)
        else:
            ctx = _resolve_jwt_context(raw_bearer, x_workspace_id, db)
    else:
        if is_auth_required():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required. Provide Bearer JWT token or X-API-Key header.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        ws_id = DEFAULT_WORKSPACE_ID
        if x_workspace_id:
            ws = db.query(WorkspaceRecord).filter_by(id=x_workspace_id).first()
            if not ws:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Workspace '{x_workspace_id}' not found.",
                )
            ws_id = ws.id
        ctx = RequestContext(
            workspace_id=ws_id,
            user=None,
            user_id=None,
            workspace_role="owner",
            auth_method="anonymous",
            scopes=["*"],
        )

    with workspace_scope(ctx.workspace_id, ctx.user_id):
        yield ctx


def require_authenticated_user(
    ctx: RequestContext = Depends(get_request_context),
) -> RequestContext:
    """Require an authenticated user (JWT or API key), regardless of global auth mode."""
    if ctx.user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required for this endpoint.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return ctx


def require_min_role(ctx: RequestContext, min_role: str) -> None:
    """Enforce minimum workspace role (viewer < member < admin < owner)."""
    if not ctx.has_role(min_role):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient workspace permissions. Requires '{min_role}' role or higher.",
        )

