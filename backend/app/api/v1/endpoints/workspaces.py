"""Multi-user workspace and RBAC membership endpoints."""

from __future__ import annotations

import re
import uuid
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.app.core.security import (
    RequestContext,
    get_request_context,
    require_authenticated_user,
    require_min_role,
)
from backend.app.database.models import (
    AsyncJobRecord,
    ExperimentRecord,
    TrainingJobRecord,
    UserRecord,
    WorkspaceMemberRecord,
    WorkspaceRecord,
)
from backend.app.database.session import get_db
from backend.app.schemas.api_schemas import (
    WorkspaceCreateRequest,
    WorkspaceMemberAddRequest,
    WorkspaceMemberResponse,
    WorkspaceMemberUpdateRequest,
    WorkspaceSummary,
    WorkspaceUpdateRequest,
)

router = APIRouter()


def _slugify(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "workspace"
    return f"{base[:48]}-{uuid.uuid4().hex[:6]}"


def _require_workspace_membership(
    db: Session,
    workspace_id: str,
    ctx: RequestContext,
    min_role: str = "viewer",
) -> WorkspaceRecord:
    ws = db.query(WorkspaceRecord).filter_by(id=workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail=f"Workspace '{workspace_id}' not found.")

    if ctx.user is None:
        # Local anonymous dev mode
        return ws

    if ctx.user.is_superuser:
        return ws

    membership = (
        db.query(WorkspaceMemberRecord)
        .filter_by(workspace_id=workspace_id, user_id=ctx.user_id)
        .first()
    )
    if not membership:
        raise HTTPException(status_code=403, detail="You are not a member of this workspace.")

    scoped_ctx = RequestContext(
        workspace_id=workspace_id,
        user=ctx.user,
        user_id=ctx.user_id,
        workspace_role=membership.role,
        auth_method=ctx.auth_method,
    )
    require_min_role(scoped_ctx, min_role)
    return ws


@router.get("", response_model=List[WorkspaceSummary])
def list_workspaces(
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
):
    """List workspaces accessible to the current user (or default local workspaces)."""
    if ctx.user is None:
        workspaces = db.query(WorkspaceRecord).order_by(WorkspaceRecord.created_at.asc()).all()
        return [
            WorkspaceSummary(
                id=ws.id,
                name=ws.name,
                slug=ws.slug,
                description=ws.description,
                owner_id=ws.owner_id,
                role="owner",
                member_count=db.query(WorkspaceMemberRecord).filter_by(workspace_id=ws.id).count(),
                created_at=ws.created_at,
            )
            for ws in workspaces
        ]

    if ctx.user.is_superuser:
        workspaces = db.query(WorkspaceRecord).all()
        return [
            WorkspaceSummary(
                id=ws.id,
                name=ws.name,
                slug=ws.slug,
                description=ws.description,
                owner_id=ws.owner_id,
                role="owner",
                member_count=db.query(WorkspaceMemberRecord).filter_by(workspace_id=ws.id).count(),
                created_at=ws.created_at,
            )
            for ws in workspaces
        ]

    memberships = (
        db.query(WorkspaceMemberRecord)
        .filter_by(user_id=ctx.user_id)
        .all()
    )
    role_by_ws = {m.workspace_id: m.role for m in memberships}
    if not role_by_ws:
        return []

    workspaces = (
        db.query(WorkspaceRecord)
        .filter(WorkspaceRecord.id.in_(list(role_by_ws.keys())))
        .order_by(WorkspaceRecord.created_at.asc())
        .all()
    )
    return [
        WorkspaceSummary(
            id=ws.id,
            name=ws.name,
            slug=ws.slug,
            description=ws.description,
            owner_id=ws.owner_id,
            role=role_by_ws.get(ws.id, "member"),
            member_count=db.query(WorkspaceMemberRecord).filter_by(workspace_id=ws.id).count(),
            created_at=ws.created_at,
        )
        for ws in workspaces
    ]


@router.post("", response_model=WorkspaceSummary, status_code=status.HTTP_201_CREATED)
def create_workspace(
    req: WorkspaceCreateRequest,
    ctx: RequestContext = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Create a new isolated workspace and assign the creator as owner."""
    ws_id = f"ws_{uuid.uuid4().hex[:12]}"
    ws = WorkspaceRecord(
        id=ws_id,
        name=req.name.strip(),
        slug=_slugify(req.name),
        description=req.description,
        owner_id=ctx.user_id,
        settings_json={},
    )
    db.add(ws)
    db.add(
        WorkspaceMemberRecord(
            id=f"wsm_{uuid.uuid4().hex[:12]}",
            workspace_id=ws_id,
            user_id=ctx.user_id,
            role="owner",
        )
    )
    db.commit()
    db.refresh(ws)

    return WorkspaceSummary(
        id=ws.id,
        name=ws.name,
        slug=ws.slug,
        description=ws.description,
        owner_id=ws.owner_id,
        role="owner",
        member_count=1,
        created_at=ws.created_at,
    )


@router.get("/{workspace_id}", response_model=Dict[str, Any])
def get_workspace_detail(
    workspace_id: str,
    ctx: RequestContext = Depends(get_request_context),
    db: Session = Depends(get_db),
):
    """Retrieve workspace details along with member and resource statistics."""
    ws = _require_workspace_membership(db, workspace_id, ctx, min_role="viewer")
    member_count = db.query(WorkspaceMemberRecord).filter_by(workspace_id=ws.id).count()
    experiment_count = db.query(ExperimentRecord).filter_by(workspace_id=ws.id).count()
    training_count = db.query(TrainingJobRecord).filter_by(workspace_id=ws.id).count()
    async_job_count = db.query(AsyncJobRecord).filter_by(workspace_id=ws.id).count()

    return {
        "id": ws.id,
        "name": ws.name,
        "slug": ws.slug,
        "description": ws.description,
        "owner_id": ws.owner_id,
        "settings_json": ws.settings_json or {},
        "member_count": member_count,
        "resource_counts": {
            "experiments": experiment_count,
            "training_jobs": training_count,
            "async_jobs": async_job_count,
        },
        "created_at": ws.created_at,
    }


@router.patch("/{workspace_id}", response_model=WorkspaceSummary)
def update_workspace(
    workspace_id: str,
    req: WorkspaceUpdateRequest,
    ctx: RequestContext = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Update workspace name, description, or settings (requires admin/owner)."""
    ws = _require_workspace_membership(db, workspace_id, ctx, min_role="admin")
    if req.name is not None:
        ws.name = req.name.strip()
    if req.description is not None:
        ws.description = req.description
    if req.settings_json is not None:
        ws.settings_json = req.settings_json

    db.commit()
    db.refresh(ws)

    membership = (
        db.query(WorkspaceMemberRecord)
        .filter_by(workspace_id=ws.id, user_id=ctx.user_id)
        .first()
    )
    return WorkspaceSummary(
        id=ws.id,
        name=ws.name,
        slug=ws.slug,
        description=ws.description,
        owner_id=ws.owner_id,
        role=membership.role if membership else "owner",
        member_count=db.query(WorkspaceMemberRecord).filter_by(workspace_id=ws.id).count(),
        created_at=ws.created_at,
    )


@router.get("/{workspace_id}/members", response_model=List[WorkspaceMemberResponse])
def list_workspace_members(
    workspace_id: str,
    ctx: RequestContext = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """List all members and roles in a workspace."""
    _require_workspace_membership(db, workspace_id, ctx, min_role="viewer")
    rows = (
        db.query(WorkspaceMemberRecord, UserRecord)
        .join(UserRecord, WorkspaceMemberRecord.user_id == UserRecord.id)
        .filter(WorkspaceMemberRecord.workspace_id == workspace_id)
        .all()
    )
    return [
        WorkspaceMemberResponse(
            id=m.id,
            workspace_id=m.workspace_id,
            user_id=m.user_id,
            email=u.email,
            full_name=u.full_name,
            role=m.role,
            created_at=m.created_at,
        )
        for m, u in rows
    ]


@router.post("/{workspace_id}/members", response_model=WorkspaceMemberResponse, status_code=status.HTTP_201_CREATED)
def add_workspace_member(
    workspace_id: str,
    req: WorkspaceMemberAddRequest,
    ctx: RequestContext = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Add an existing user to a workspace with a specified RBAC role."""
    _require_workspace_membership(db, workspace_id, ctx, min_role="admin")
    target_user = db.query(UserRecord).filter_by(email=req.email.strip().lower()).first()
    if not target_user:
        raise HTTPException(status_code=404, detail=f"User '{req.email}' not found.")

    existing = (
        db.query(WorkspaceMemberRecord)
        .filter_by(workspace_id=workspace_id, user_id=target_user.id)
        .first()
    )
    if existing:
        raise HTTPException(status_code=409, detail="User is already a member of this workspace.")

    member = WorkspaceMemberRecord(
        id=f"wsm_{uuid.uuid4().hex[:12]}",
        workspace_id=workspace_id,
        user_id=target_user.id,
        role=req.role,
    )
    db.add(member)
    db.commit()
    db.refresh(member)

    return WorkspaceMemberResponse(
        id=member.id,
        workspace_id=member.workspace_id,
        user_id=member.user_id,
        email=target_user.email,
        full_name=target_user.full_name,
        role=member.role,
        created_at=member.created_at,
    )


@router.patch("/{workspace_id}/members/{user_id}", response_model=WorkspaceMemberResponse)
def update_workspace_member_role(
    workspace_id: str,
    user_id: str,
    req: WorkspaceMemberUpdateRequest,
    ctx: RequestContext = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Update a workspace member's RBAC role."""
    _require_workspace_membership(db, workspace_id, ctx, min_role="admin")
    member = (
        db.query(WorkspaceMemberRecord)
        .filter_by(workspace_id=workspace_id, user_id=user_id)
        .first()
    )
    if not member:
        raise HTTPException(status_code=404, detail="Workspace member not found.")

    target_user = db.query(UserRecord).filter_by(id=user_id).first()
    member.role = req.role
    db.commit()
    db.refresh(member)

    return WorkspaceMemberResponse(
        id=member.id,
        workspace_id=member.workspace_id,
        user_id=member.user_id,
        email=target_user.email if target_user else "",
        full_name=target_user.full_name if target_user else "",
        role=member.role,
        created_at=member.created_at,
    )


@router.delete("/{workspace_id}/members/{user_id}", response_model=Dict[str, str])
def remove_workspace_member(
    workspace_id: str,
    user_id: str,
    ctx: RequestContext = Depends(require_authenticated_user),
    db: Session = Depends(get_db),
):
    """Remove a user from a workspace (prevents removing the sole owner)."""
    _require_workspace_membership(db, workspace_id, ctx, min_role="admin")
    member = (
        db.query(WorkspaceMemberRecord)
        .filter_by(workspace_id=workspace_id, user_id=user_id)
        .first()
    )
    if not member:
        raise HTTPException(status_code=404, detail="Workspace member not found.")

    if member.role == "owner":
        owner_count = (
            db.query(WorkspaceMemberRecord)
            .filter_by(workspace_id=workspace_id, role="owner")
            .count()
        )
        if owner_count <= 1:
            raise HTTPException(
                status_code=400,
                detail="Cannot remove the sole owner of a workspace.",
            )

    db.delete(member)
    db.commit()
    return {"status": "removed", "workspace_id": workspace_id, "user_id": user_id}

