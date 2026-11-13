"""Unit tests for Phase 8 security primitives, Alembic migrations, and DB migration."""

from __future__ import annotations

import datetime
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy.orm import sessionmaker

from backend.app.core.security import (
    create_access_token,
    create_jwt_token,
    create_refresh_token,
    decode_jwt_token,
    generate_api_key,
    hash_api_key,
    hash_password,
    verify_password,
)
from backend.app.database.models import ExperimentRecord, UserRecord, WorkspaceRecord
from backend.app.database.session import build_engine, init_db
from scripts.migrate_sqlite_to_postgres import migrate_database


def test_password_hashing_and_verification():
    pw = "SuperSecret123!"
    hashed = hash_password(pw)
    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password(pw, hashed) is True
    assert verify_password("WrongPassword", hashed) is False
    assert verify_password(pw, "malformed_hash") is False


def test_jwt_create_decode_and_validation():
    token = create_access_token(
        user_id="usr_123",
        email="alice@example.com",
        workspace_id="ws_abc",
        role="admin",
    )
    claims = decode_jwt_token(token, expected_type="access")
    assert claims["sub"] == "usr_123"
    assert claims["email"] == "alice@example.com"
    assert claims["workspace_id"] == "ws_abc"
    assert claims["role"] == "admin"

    # Wrong token type rejected
    with pytest.raises(ValueError, match="Expected JWT type"):
        decode_jwt_token(token, expected_type="refresh")

    # Tampered token rejected
    parts = token.split(".")
    tampered = f"{parts[0]}.{parts[1]}.invalid_sig"
    with pytest.raises(ValueError, match="Invalid JWT signature"):
        decode_jwt_token(tampered)

    # Expired token rejected
    expired = create_jwt_token(
        claims={"sub": "usr_123", "type": "access"},
        expires_delta=datetime.timedelta(seconds=-10),
    )
    with pytest.raises(ValueError, match="expired"):
        decode_jwt_token(expired)

    refresh = create_refresh_token("usr_123", "alice@example.com", "ws_abc")
    refresh_claims = decode_jwt_token(refresh, expected_type="refresh")
    assert refresh_claims["sub"] == "usr_123"


def test_api_key_generation_format():
    raw_key, prefix, key_hash = generate_api_key()
    assert raw_key.startswith("blab_")
    assert raw_key.startswith(prefix + "_")
    assert key_hash == hash_api_key(raw_key)
    assert len(key_hash) == 64


def test_alembic_upgrade_head_on_fresh_sqlite(tmp_path: Path):
    db_file = tmp_path / "alembic_test.db"
    env = os.environ.copy()
    env["DATABASE_URL"] = f"sqlite:///{db_file}"

    project_root = Path(__file__).resolve().parents[1]
    res = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"Alembic failed: {res.stderr}\n{res.stdout}"

    eng = build_engine(f"sqlite:///{db_file}")
    from sqlalchemy import inspect

    tables = set(inspect(eng).get_table_names())
    assert {
        "users",
        "workspaces",
        "workspace_members",
        "api_keys",
        "async_jobs",
        "experiments",
        "training_jobs",
        "alembic_version",
    }.issubset(tables)


def test_sqlite_migration_script_idempotent(tmp_path: Path):
    src_url = f"sqlite:///{tmp_path / 'src.db'}"
    dst_url = f"sqlite:///{tmp_path / 'dst.db'}"

    src_engine = build_engine(src_url)
    init_db(src_engine)
    SrcSession = sessionmaker(bind=src_engine)
    db = SrcSession()
    try:
        uid = f"usr_{uuid.uuid4().hex[:8]}"
        wid = f"ws_{uuid.uuid4().hex[:8]}"
        db.add(
            UserRecord(
                id=uid,
                email="mig@example.com",
                full_name="Migrator",
                hashed_password=hash_password("Pass1234!"),
            )
        )
        db.add(WorkspaceRecord(id=wid, name="Mig WS", slug=f"mig-{uid}", owner_id=uid))
        db.add(
            ExperimentRecord(
                id=f"exp_{uuid.uuid4().hex[:8]}",
                workspace_id=wid,
                created_by_user_id=uid,
                name="Migration Exp",
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                rougeL=0.31,
            )
        )
        db.commit()
    finally:
        db.close()

    report1 = migrate_database(src_url, dst_url)
    assert report1["users"]["inserted"] == 1
    assert report1["experiments"]["inserted"] == 1

    # Re-run should be idempotent (0 new inserts, all skipped)
    report2 = migrate_database(src_url, dst_url)
    assert report2["users"]["inserted"] == 0
    assert report2["users"]["skipped"] == 1
    assert report2["experiments"]["inserted"] == 0
    assert report2["experiments"]["skipped"] == 1

