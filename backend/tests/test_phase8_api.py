"""Integration tests for Phase 8: JWT auth, API keys, workspace RBAC isolation & async job queue."""

from __future__ import annotations

import time
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.app.database.models import AsyncJobRecord, ExperimentRecord
from backend.app.database.session import SessionLocal, init_db
from backend.app.main import app
from backend.app.workers.queue import job_dispatcher

init_db()
client = TestClient(app)


def _unique_email(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}@example.com"


def test_register_login_refresh_and_me():
    email = _unique_email("alice")
    reg = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "password": "Password123!",
            "full_name": "Alice Engineer",
            "workspace_name": "Alice Lab",
        },
    )
    assert reg.status_code == 201, reg.text
    reg_data = reg.json()
    assert reg_data["access_token"]
    assert reg_data["refresh_token"]
    assert reg_data["user"]["email"] == email
    assert reg_data["workspace"]["role"] == "owner"

    # /api/auth/me with Bearer JWT
    me = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {reg_data['access_token']}"},
    )
    assert me.status_code == 200
    assert me.json()["authenticated"] is True
    assert me.json()["auth_method"] == "jwt"
    assert me.json()["workspace_id"] == reg_data["workspace"]["id"]

    # Login
    login = client.post("/api/auth/login", json={"email": email, "password": "Password123!"})
    assert login.status_code == 200
    assert login.json()["user"]["id"] == reg_data["user"]["id"]

    # Refresh
    refresh = client.post("/api/auth/refresh", json={"refresh_token": reg_data["refresh_token"]})
    assert refresh.status_code == 200
    assert refresh.json()["access_token"]


def test_api_key_lifecycle_and_auth():
    email = _unique_email("apikey")
    reg = client.post(
        "/api/auth/register",
        json={"email": email, "password": "Password123!", "full_name": "Key Owner"},
    ).json()
    headers = {"Authorization": f"Bearer {reg['access_token']}"}

    created = client.post(
        "/api/auth/api-keys",
        json={"name": "CI Pipeline Key", "scopes": ["read", "write"], "expires_in_days": 30},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    key_data = created.json()
    raw_key = key_data["raw_key"]
    assert raw_key.startswith("blab_")

    # Authenticate using X-API-Key
    me = client.get("/api/auth/me", headers={"X-API-Key": raw_key})
    assert me.status_code == 200
    assert me.json()["auth_method"] == "api_key"
    assert me.json()["user"]["email"] == email

    # List keys (raw_key must not be exposed on list)
    listed = client.get("/api/auth/api-keys", headers=headers)
    assert listed.status_code == 200
    assert any(k["id"] == key_data["id"] and k["raw_key"] is None for k in listed.json())

    # Revoke key
    revoked = client.delete(f"/api/auth/api-keys/{key_data['id']}", headers=headers)
    assert revoked.status_code == 200
    assert revoked.json()["is_active"] is False

    # Revoked key rejected
    denied = client.get("/api/auth/me", headers={"X-API-Key": raw_key})
    assert denied.status_code == 401


def test_workspace_isolation_and_rbac():
    alice = client.post(
        "/api/auth/register",
        json={"email": _unique_email("wsa"), "password": "Password123!", "workspace_name": "Team A"},
    ).json()
    bob = client.post(
        "/api/auth/register",
        json={"email": _unique_email("wsb"), "password": "Password123!", "workspace_name": "Team B"},
    ).json()

    ws_a = alice["workspace"]["id"]
    ws_b = bob["workspace"]["id"]

    # Seed an experiment strictly in ws_a
    db = SessionLocal()
    exp_id = f"exp_iso_{uuid.uuid4().hex[:8]}"
    try:
        db.add(
            ExperimentRecord(
                id=exp_id,
                workspace_id=ws_a,
                created_by_user_id=alice["user"]["id"],
                name="Team A Secret Experiment",
                model_id="Qwen/Qwen2.5-0.5B-Instruct",
                category="lora",
                rougeL=0.42,
            )
        )
        db.commit()
    finally:
        db.close()

    # Alice sees her experiment
    alice_exps = client.get(
        "/api/experiments",
        headers={"Authorization": f"Bearer {alice['access_token']}"},
    )
    assert alice_exps.status_code == 200
    assert any(e["id"] == exp_id for e in alice_exps.json())

    # Bob does NOT see Alice's experiment in ws_b
    bob_exps = client.get(
        "/api/experiments",
        headers={"Authorization": f"Bearer {bob['access_token']}"},
    )
    assert bob_exps.status_code == 200
    assert all(e["id"] != exp_id for e in bob_exps.json())

    # Bob cannot access ws_a via X-Workspace-ID
    forbidden = client.get(
        "/api/experiments",
        headers={
            "Authorization": f"Bearer {bob['access_token']}",
            "X-Workspace-ID": ws_a,
        },
    )
    assert forbidden.status_code == 403

    # Alice invites Bob as a 'viewer'
    add_res = client.post(
        f"/api/workspaces/{ws_a}/members",
        json={"email": bob["user"]["email"], "role": "viewer"},
        headers={"Authorization": f"Bearer {alice['access_token']}"},
    )
    assert add_res.status_code == 201

    # Bob can now read ws_a experiments...
    bob_read_a = client.get(
        "/api/experiments",
        headers={
            "Authorization": f"Bearer {bob['access_token']}",
            "X-Workspace-ID": ws_a,
        },
    )
    assert bob_read_a.status_code == 200
    assert any(e["id"] == exp_id for e in bob_read_a.json())

    # ...but as 'viewer', Bob is blocked from submitting write jobs in ws_a
    bob_write_a = client.post(
        "/api/jobs/submit",
        json={"job_type": "evaluation", "payload": {}},
        headers={
            "Authorization": f"Bearer {bob['access_token']}",
            "X-Workspace-ID": ws_a,
        },
    )
    assert bob_write_a.status_code == 403


def test_strict_auth_required_mode(monkeypatch):
    monkeypatch.setenv("BENCHLAB_AUTH_REQUIRED", "true")

    # Health check remains accessible
    assert client.get("/api/health").status_code == 200

    # Protected routes reject unauthenticated requests
    assert client.get("/api/experiments").status_code == 401
    assert client.get("/api/jobs").status_code == 401

    # Authenticated user succeeds
    reg = client.post(
        "/api/auth/register",
        json={"email": _unique_email("strict"), "password": "Password123!"},
    ).json()
    ok = client.get(
        "/api/experiments",
        headers={"Authorization": f"Bearer {reg['access_token']}"},
    )
    assert ok.status_code == 200


def test_async_job_queue_submit_poll_and_cancel():
    # Register lightweight deterministic handler for testing
    def fake_eval_handler(payload, progress_cb):
        progress_cb(50.0, "Halfway done")
        time.sleep(0.05)
        return {"evaluation_id": "eval_test_1", "rougeL": 0.35, "echo": payload.get("tag")}

    def slow_cancel_handler(payload, progress_cb):
        for i in range(10):
            progress_cb(float(i * 10), f"Step {i}")
            time.sleep(0.05)
        return {"done": True}

    original_eval = job_dispatcher._handlers["evaluation"]
    original_quant = job_dispatcher._handlers["quantization"]
    job_dispatcher.register_handler("evaluation", fake_eval_handler)
    job_dispatcher.register_handler("quantization", slow_cancel_handler)

    try:
        reg = client.post(
            "/api/auth/register",
            json={"email": _unique_email("jobs"), "password": "Password123!"},
        ).json()
        headers = {"Authorization": f"Bearer {reg['access_token']}"}

        submit = client.post(
            "/api/jobs/submit",
            json={"job_type": "evaluation", "payload": {"tag": "smoke"}},
            headers=headers,
        )
        assert submit.status_code == 202, submit.text
        job_id = submit.json()["id"]

        # Poll until completed
        final = None
        for _ in range(40):
            res = client.get(f"/api/jobs/{job_id}", headers=headers)
            assert res.status_code == 200
            final = res.json()
            if final["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)

        assert final is not None
        assert final["status"] == "completed"
        assert final["progress_pct"] == 100.0
        assert final["result_json"]["echo"] == "smoke"

        # Worker health check
        health = client.get("/api/jobs/worker/health", headers=headers)
        assert health.status_code == 200
        assert health.json()["total_jobs"] >= 1

        # Cancellation test
        slow = client.post(
            "/api/jobs/submit",
            json={"job_type": "quantization", "payload": {}},
            headers=headers,
        )
        assert slow.status_code == 202
        slow_id = slow.json()["id"]
        cancel = client.post(f"/api/jobs/{slow_id}/cancel", headers=headers)
        assert cancel.status_code == 200
        assert cancel.json()["status"] == "cancelled"
    finally:
        job_dispatcher.register_handler("evaluation", original_eval)
        job_dispatcher.register_handler("quantization", original_quant)

