"""Database session management, PostgreSQL/SQLite configuration, and schema initialization."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from backend.app.core.context import DEFAULT_WORKSPACE_ID
from backend.app.database.models import Base, WorkspaceRecord

logger = logging.getLogger("benchlab.database")

# Database path defaults to local SQLite file in project root
DEFAULT_DB_PATH = Path(__file__).resolve().parents[3] / "benchlab.db"


def normalize_database_url(raw_url: str | None = None) -> str:
    """Normalize DATABASE_URL (handling legacy postgres:// scheme)."""
    url = raw_url or os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH}")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def build_engine(database_url: str) -> Engine:
    """Build SQLAlchemy Engine tailored for SQLite or PostgreSQL."""
    kwargs: Dict[str, Any] = {"echo": False}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    else:
        kwargs["pool_pre_ping"] = True
        kwargs["pool_size"] = int(os.getenv("DB_POOL_SIZE", "10"))
        kwargs["max_overflow"] = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        kwargs["pool_recycle"] = int(os.getenv("DB_POOL_RECYCLE", "1800"))
    return create_engine(database_url, **kwargs)


DATABASE_URL = normalize_database_url()
engine = build_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

_WORKSPACE_SCOPED_TABLES = (
    "models",
    "datasets",
    "evaluations",
    "training_jobs",
    "quantizations",
    "benchmarks",
    "experiments",
)


def _ensure_workspace_columns(target_engine: Engine) -> None:
    """Add workspace_id / created_by_user_id columns to legacy SQLite/Postgres tables."""
    inspector = inspect(target_engine)
    existing_tables = set(inspector.get_table_names())

    with target_engine.begin() as conn:
        for table_name in _WORKSPACE_SCOPED_TABLES:
            if table_name not in existing_tables:
                continue
            cols = {col["name"] for col in inspector.get_columns(table_name)}
            if "workspace_id" not in cols:
                conn.execute(
                    text(
                        f"ALTER TABLE {table_name} "
                        f"ADD COLUMN workspace_id VARCHAR(128) NOT NULL DEFAULT '{DEFAULT_WORKSPACE_ID}'"
                    )
                )
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS ix_{table_name}_workspace_id "
                        f"ON {table_name} (workspace_id)"
                    )
                )
            if "created_by_user_id" not in cols:
                conn.execute(
                    text(
                        f"ALTER TABLE {table_name} "
                        f"ADD COLUMN created_by_user_id VARCHAR(128)"
                    )
                )


def ensure_default_workspace(db: Session) -> WorkspaceRecord:
    """Ensure the default local workspace exists for backwards-compatible single-user mode."""
    ws = db.query(WorkspaceRecord).filter_by(id=DEFAULT_WORKSPACE_ID).first()
    if not ws:
        ws = WorkspaceRecord(
            id=DEFAULT_WORKSPACE_ID,
            name="Default Workspace",
            slug="default",
            description="Default local workspace for OpenLLM BenchLab experiments.",
            owner_id=None,
            settings_json={"is_default": True},
        )
        db.add(ws)
        db.commit()
        db.refresh(ws)
    return ws


def init_db(target_engine: Engine | None = None) -> None:
    """Create all tables, migrate missing workspace columns, and seed default workspace."""
    eng = target_engine or engine
    Base.metadata.create_all(bind=eng)
    _ensure_workspace_columns(eng)

    SessionMaker = sessionmaker(autocommit=False, autoflush=False, bind=eng)
    db = SessionMaker()
    try:
        ensure_default_workspace(db)
    finally:
        db.close()


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for obtaining a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
