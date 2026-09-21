from __future__ import annotations

import os

import pytest

# Required env vars for Settings() — set here so individual test modules
# don't each have to remember. Real values come from .env in dev/prod.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-not-used")
os.environ.setdefault("EXEC_EMAIL_ADDRESS", "ceo.test@example.com")

# Tests run as a local, non-public process. If this leaks in from the developer's
# shell, create_app() fails closed on the missing BACKEND_SHARED_SECRET and every
# full-app test errors at construction — the same trap BACKEND_SHARED_SECRET sets
# (see CLAUDE.md → Testing). Clear it so the suite matches CI either way.
os.environ.pop("OE_PUBLIC_DEPLOYMENT", None)

# Real trap, found live (2026-09-21): Settings reads .env by FILE PATH
# (pydantic-settings' own env_file source), independent of os.environ — so
# `monkeypatch.delenv("IDUNA_URL")` (test_iduna_auth_gate.py's own existing
# _clean_env fixture) does NOT stop a value from still being read off disk;
# it only stops an explicit os.environ override from winning. This was latent
# until a real .env now legitimately exists here (IDUNA_URL, GEMINI_API_KEY —
# see docs/IDUNA_INTEGRATION.md, a real documented deployment shape, not a
# hypothetical), at which point test_iduna_auth_gate.py's own
# "unaffected by the IDUNA code path" case silently started reading a real
# IDUNA_URL off disk and failed. Fix at the source: tests never read the real
# .env file at all, regardless of what a developer happens to have configured
# there — every value a test needs comes from conftest's own os.environ.setdefault
# calls above or an individual test's explicit monkeypatch.setenv, both of which
# already correctly take precedence over the env_file source either way.
from openexecutive.config import Settings  # noqa: E402

Settings.model_config["env_file"] = None
Settings.model_config["env_file_encoding"] = None


@pytest.fixture(autouse=True)
def reset_active_gateway():
    """Ensure the module-level MCP gateway singleton is cleared between tests."""
    from openexecutive.orchestrator.mcp_gateway import set_active_gateway
    set_active_gateway(None)
    yield
    set_active_gateway(None)


@pytest.fixture
def install_source_feed(monkeypatch: pytest.MonkeyPatch):
    """Point one monitoring source adapter's bounded fetch at a canned body.

    Every feed adapter (``vendor_status``, ``rss``, ``edgar``, …) reaches
    the network through its own module-level ``fetch_bounded`` +
    ``validate_target_url`` pair, so each test module used to carry its own
    near-identical monkeypatch helper. Yields a setter:

        captured = install_source_feed("edgar", b"<feed>…</feed>")
        ...
        assert captured["user_agent"] == ...

    The returned dict records the arguments of the LAST fetch — ``url``,
    ``max_bytes``, and any keyword the adapter passes (edgar sends
    ``user_agent``; keys an adapter doesn't send are absent).
    """
    def _install(module: str, body: bytes | str) -> dict:
        captured: dict = {}
        payload = body.encode() if isinstance(body, str) else body

        async def fake_fetch(url: str, max_bytes: int, **kwargs) -> bytes:
            captured.clear()
            captured.update({"url": url, "max_bytes": max_bytes, **kwargs})
            return payload

        base = f"openexecutive.monitoring.sources.{module}"
        monkeypatch.setattr(f"{base}.fetch_bounded", fake_fetch)
        monkeypatch.setattr(f"{base}.validate_target_url", lambda u: (True, ""))
        return captured

    return _install
