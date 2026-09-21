"""S506: IDUNA-issued JWTs as an additional credential on the existing
shared-secret gate (api/main.py's own ``_shared_secret_gate``).

Each test builds its own app via ``create_app()`` (matching
``test_public_deployment_guard.py``'s own pattern) since the gate captures
``BACKEND_SHARED_SECRET``/``IDUNA_URL`` once at app-construction time, not
per-request — reusing the module-level ``openexecutive.api.main.app``
singleton (as ``test_architecture_endpoint.py`` does) would not see env vars
set after that singleton's own import-time construction.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from openexecutive.api.main import create_app


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BACKEND_SHARED_SECRET", raising=False)
    monkeypatch.delenv("IDUNA_URL", raising=False)


def test_iduna_only_deployment_accepts_a_valid_bearer_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No BACKEND_SHARED_SECRET at all — an IDUNA JWT alone must be enough to
    call a protected route, not just to satisfy the public-deployment guard
    at boot (test_public_deployment_guard.py's own coverage)."""
    monkeypatch.setenv("IDUNA_URL", "http://localhost:8080")

    with patch(
        "openexecutive.auth.iduna_auth.IDUNAJWTValidator.verify",
        new_callable=AsyncMock,
        return_value={"sub": "agent:openexec-prod", "permissions": ["openexec.read"]},
    ):
        client = TestClient(create_app())
        res = client.get(
            "/architecture/sections", headers={"Authorization": "Bearer fake-but-mocked-valid"}
        )
        assert res.status_code == 200
        assert "sections" in res.json()


def test_iduna_only_deployment_rejects_a_missing_or_bad_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IDUNA_URL", "http://localhost:8080")

    client = TestClient(create_app())
    # No Authorization header at all.
    res = client.get("/architecture/sections")
    assert res.status_code == 401

    with patch(
        "openexecutive.auth.iduna_auth.IDUNAJWTValidator.verify",
        new_callable=AsyncMock,
        side_effect=ValueError("Key ID x not found in JWKS"),
    ):
        res = client.get(
            "/architecture/sections", headers={"Authorization": "Bearer garbage"}
        )
        assert res.status_code == 401


def test_shared_secret_still_works_when_iduna_also_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both credentials configured: the pre-existing x-api-key path must keep
    working unchanged — IDUNA is additive, never a replacement."""
    monkeypatch.setenv("BACKEND_SHARED_SECRET", "s" * 32)
    monkeypatch.setenv("IDUNA_URL", "http://localhost:8080")

    client = TestClient(create_app())
    res = client.get("/architecture/sections", headers={"x-api-key": "s" * 32})
    assert res.status_code == 200


def test_shared_secret_only_deployment_is_unaffected_by_iduna_code_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No IDUNA_URL at all: behavior must be byte-for-byte the pre-S506
    shared-secret-only gate — a wrong/missing x-api-key still 401s, a bad
    Authorization header is never even consulted."""
    monkeypatch.setenv("BACKEND_SHARED_SECRET", "s" * 32)

    client = TestClient(create_app())
    with patch(
        "openexecutive.auth.iduna_auth.IDUNAJWTValidator.verify",
        new_callable=AsyncMock,
    ) as mock_verify:
        res = client.get(
            "/architecture/sections",
            headers={"Authorization": "Bearer whatever-nobody-validates-this"},
        )
        assert res.status_code == 401
        # The docstring's own claim, actually checked: with no IDUNA_URL
        # configured, no IDUNAJWTValidator even exists on this app instance,
        # so a bad Authorization header is never handed to it at all —
        # not "consulted and happened to fail".
        mock_verify.assert_not_called()
