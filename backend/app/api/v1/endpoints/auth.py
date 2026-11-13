"""Authentication, JWT token lifecycle, and API key management endpoints."""

from __future__ import annotations

import datetime
import re
import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.security import (
    RequestContext,
    create_access_token,
    create_refresh_token,
    decode_jwt_token,
    generate_api_key,
    get_access_token_ttl_minutes,
    get_request_context,
    hash_password,
    require_authenticated_user,
    require_min_role,
    verify_password,
)
from backend.app.database.models import (
    ApiKeyRecord,
    UserRecord,
    WorkspaceMemberRecord,
    WorkspaceRecord,
)
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    ApiKeyCreateRequest,
    ApiKeyResponse,
    AuthMeResponse,
    AuthTokenResponse,
    RefreshTokenRequest,
    UserLoginRequest,
    UserRegisterRequest,
    UserSummary,
    WorkspaceSummary,
)

router = APIRouter()


def _slugify(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "workspace"
    return f"{base[:48]}-{uuid.uuid4().hex[:6]}"


def _to_user_summary(user: UserRecord) -> UserSummary:
    return UserSummary(
        id=user.id,
        email=user.email,
        full_name=user.full_name,
        is_active=user.is_active,
        is_superuser=user.is_superuser,
        created_at=user.created_at,
    )


def _to_workspace_summary(db: Session, ws: WorkspaceRecord, role: str) -> WorkspaceSummary:
    member_count = db.query(WorkspaceMemberRecord).filter_by(workspace_id=ws.id).count()
    return WorkspaceSummary(
        id=ws.id,
        name=ws.name,
        slug=ws.slug,
        description=ws.description,
        owner_id=ws.owner_id,
        role=role,
        member_count=member_count,
        created_at=ws.created_at,
    )


@router.post("/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
def register_user(req: UserRegisterRequest, db: Session = Depends(get_db)):
    """Register a new user, provision a personal workspace, and issue JWT tokens."""
    normalized_email = req.email.strip().lower()
    if "@" not in normalized_email:
        raise HTTPException(status_code=400, detail="Valid email address is required.")

    if db.query(UserRecord).filter_by(email=normalized_email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists.")

    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    user = UserRecord(
        id=user_id,
        email=normalized_email,
        full_name=req.full_name.strip() or normalized_email.split("@")[0],
        hashed_password=hash_password(req.password),
        is_active=True,
        is_superuser=False,
    )
    db.add(user)

    ws_name = (req.workspace_name or f"{user.full_name}'s Workspace").strip()
    ws_id = f"ws_{uuid.uuid4().hex[:12]}"
    ws = WorkspaceRecord(
        id=ws_id,
        name=ws_name,
        slug=_slugify(ws_name),
        description="Personal optimization & benchmarking workspace",
        owner_id=user_id,
        settings_json={},
    )
    db.add(ws)

    membership = WorkspaceMemberRecord(
        id=f"wsm_{uuid.uuid4().hex[:12]}",
        workspace_id=ws_id,
        user_id=user_id,
        role="owner",
    )
    db.add(membership)
    db.commit()
    db.refresh(user)
    db.refresh(ws)

    access_token = create_access_token(user.id, user.email, ws.id, role="owner")
    refresh_token = create_refresh_token(user.id, user.email, ws.id)

    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in_seconds=get_access_token_ttl_minutes() * 60,
        user=_to_user_summary(user),
        workspace=_to_workspace_summary(db, ws, role="owner"),
    )


@router.post("/login", response_model=AuthTokenResponse)
def login_user(req: UserLoginRequest, db: Session = Depends(get_db)):
    """Authenticate with email and password and return JWT access + refresh tokens."""
    normalized_email = req.email.strip().lower()
    user = db.query(UserRecord).filter_by(email=normalized_email).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
        )
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is disabled.")

    membership = None
    if req.workspace_id:
        membership = (
            db.query(WorkspaceMemberRecord)
            .filter_by(workspace_id=req.workspace_id, user_id=user.id)
            .first()
        )
        if not membership and not user.is_superuser:
            raise HTTPException(status_code=403, detail="Not a member of the requested workspace.")
        ws = db.query(WorkspaceRecord).filter_by(id=req.workspace_id).first()
        if not ws:
            raise HTTPException(status_code=404, detail="Requested workspace not found.")
    else:
        membership = (
            db.query(WorkspaceMemberRecord)
            .filter_by(user_id=user.id)
            .order_by(WorkspaceMemberRecord.created_at.asc())
            .first()
        )
        if membership:
            ws = db.query(WorkspaceRecord).filter_by(id=membership.workspace_id).first()
        else:
            ws = None

    if not ws:
        # Provision workspace if user had none
        ws_id = f"ws_{uuid.uuid4().hex[:12]}"
        ws = WorkspaceRecord(
            id=ws_id,
            name=f"{user.full_name or 'My'} Workspace",
            slug=_slugify(user.email.split("@")[0]),
            owner_id=user.id,
        )
        db.add(ws)
        membership = WorkspaceMemberRecord(
            id=f"wsm_{uuid.uuid4().hex[:12]}",
            workspace_id=ws_id,
            user_id=user.id,
            role="owner",
        )
        db.add(membership)
        db.commit()
        db.refresh(ws)

    role = membership.role if membership else "owner"
    access_token = create_access_token(user.id, user.email, ws.id, role=role)
    refresh_token = create_refresh_token(user.id, user.email, ws.id)

    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in_seconds=get_access_token_ttl_minutes() * 60,
        user=_to_user_summary(user),
        workspace=_to_workspace_summary(db, ws, role=role),
    )


@router.post("/refresh", response_model=AuthTokenResponse)
def refresh_tokens(req: RefreshTokenRequest, db: Session = Depends(get_db)):
    """Exchange a valid refresh token for a fresh access/refresh token pair."""
    try:
        payload = decode_jwt_token(req.refresh_token, expected_type="refresh")
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    user = db.query(UserRecord).filter_by(id=payload.get("sub")).first()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive.")

    ws_id = payload.get("workspace_id")
    ws = db.query(WorkspaceRecord).filter_by(id=ws_id).first() if ws_id else None
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace no longer exists.")

    membership = (
        db.query(WorkspaceMemberRecord)
        .filter_by(workspace_id=ws.id, user_id=user.id)
        .first()
    )
    if not membership and not user.is_superuser:
        raise HTTPException(status_code=403, detail="Workspace access revoked.")

    role = membership.role if membership else "owner"
    return AuthTokenResponse(
        access_token=create_access_token(user.id, user.email, ws.id, role=role),
        refresh_token=create_refresh_token(user.id, user.email, ws.id),
        expires_in_seconds=get_access_token_ttl_minutes() * 60,
        user=_to_user_summary(user),
        workspace=_to_workspace_summary(db, ws, role=role),
    )


@router.get("/me", response_model=AuthMeResponse)
def get_current_identity(ctx: RequestContext = Depends(get_request_context)):
    """Inspect current authenticated identity, workspace, and role."""
    return AuthMeResponse(
        authenticated=ctx.user is not None,
        auth_method=ctx.auth_method,
        user=_to_user_summary(ctx.user) if ctx.user else None,
        workspace_id=ctx.workspace_id,
        workspace_role=ctx.workspace_role,
        scopes=ctx.scopes,
    )


@router.get("/api-keys", response_model=List[ApiKeyResponse])
def list_api_keys(
    ctx: RequestContext = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """List active and revoked API keys for the active workspace."""
    keys = (
        db.query(ApiKeyRecord)
        .filter_by(workspace_id=ctx.workspace_id)
        .order_by(ApiKeyRecord.created_at.desc())
        .all()
    )
    return [
        ApiKeyResponse(
            id=k.id,
            name=k.name,
            key_prefix=k.key_prefix,
            workspace_id=k.workspace_id,
            scopes=list(k.scopes or []),
            is_active=k.is_active,
            last_used_at=k.last_used_at,
            expires_at=k.expires_at,
            created_at=k.created_at,
        )
        for k in keys
    ]


@router.post("/api-keys", response_model=ApiKeyResponse, status_code=status.HTTP_201_CREATED)
def create_api_key(
    req: ApiKeyCreateRequest,
    ctx: RequestContext = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Generate a scoped API key for the active workspace (raw key shown once)."""
    require_min_role(ctx, "member")
    raw_key, key_prefix, key_hash = generate_api_key()
    expires_at = None
    if req.expires_in_days is not None:
        expires_at = datetime.datetime.utcnow() + datetime.timedelta(days=req.expires_in_days)

    record = ApiKeyRecord(
        id=f"key_{uuid.uuid4().hex[:12]}",
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        name=req.name.strip(),
        key_prefix=key_prefix,
        key_hash=key_hash,
        scopes=req.scopes,
        is_active=True,
        expires_at=expires_at,
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return ApiKeyResponse(
        id=record.id,
        name=record.name,
        key_prefix=record.key_prefix,
        workspace_id=record.workspace_id,
        scopes=list(record.scopes or []),
        is_active=record.is_active,
        last_used_at=record.last_used_at,
        expires_at=record.expires_at,
        created_at=record.created_at,
        raw_key=raw_key,
    )


@router.delete("/api-keys/{key_id}", response_model=ApiKeyResponse)
def revoke_api_key(
    key_id: str,
    ctx: RequestContext = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Revoke an API key in the active workspace."""
    require_min_role(ctx, "member")
    record = (
        db.query(ApiKeyRecord)
        .filter_by(id=key_id, workspace_id=ctx.workspace_id)
        .first()
    )
    if not record:
        raise HTTPException(status_code=404, detail=f"API key '{key_id}' not found.")

    if record.user_id != ctx.user_id:
        require_min_role(ctx, "admin")

    record.is_active = False
    db.commit()
    db.refresh(record)

    return ApiKeyResponse(
        id=record.id,
        name=record.name,
        key_prefix=record.key_prefix,
        workspace_id=record.workspace_id,
        scopes=list(record.scopes or []),
        is_active=record.is_active,
        last_used_at=record.last_used_at,
        expires_at=record.expires_at,
        created_at=record.created_at,
    )

