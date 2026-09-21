"""Real, direct coverage for IDUNAJWTValidator's signature AND authorization
checks (S506), using a real EC keypair and real ES256-signed tokens — no
mocking of `jwt`/`verify` itself, since that's exactly the surface a mock
would hide bugs in (this is how the pre-fix RSA/EC algorithm bug, and the
pre-fix missing-audience/permission-check bug, were actually found and
confirmed).
"""
from __future__ import annotations

import asyncio
import base64
import time
from unittest.mock import patch

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

from openexecutive.auth.iduna_auth import IDUNAJWTValidator, JWKSCache


def _b64url(n: int, length: int = 32) -> str:
    return base64.urlsafe_b64encode(n.to_bytes(length, "big")).rstrip(b"=").decode()


@pytest.fixture()
def ec_keypair():
    priv = ec.generate_private_key(ec.SECP256R1())
    return priv, priv.public_key()


@pytest.fixture()
def jwks_for(ec_keypair):
    _, pub = ec_keypair
    numbers = pub.public_numbers()
    kid = "test-key-1"

    def _jwks() -> dict:
        return {
            "keys": [
                {
                    "kty": "EC",
                    "crv": "P-256",
                    "x": _b64url(numbers.x),
                    "y": _b64url(numbers.y),
                    "kid": kid,
                    "use": "sig",
                    "alg": "ES256",
                }
            ]
        }

    return _jwks, kid


def _sign(ec_keypair, kid: str, claims: dict) -> str:
    from cryptography.hazmat.primitives import serialization

    priv, _ = ec_keypair
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return jwt.encode(claims, priv_pem, algorithm="ES256", headers={"kid": kid})


def _validator_with_cache(iduna_url: str, jwks: dict, **kwargs) -> IDUNAJWTValidator:
    v = IDUNAJWTValidator(iduna_url, **kwargs)
    v.jwks_cache._cache = jwks
    v.jwks_cache._fetched_at = time.monotonic()
    return v


@pytest.mark.asyncio
async def test_valid_m2m_agent_token_with_openexec_permission_verifies(
    ec_keypair, jwks_for
):
    jwks, kid = jwks_for
    token = _sign(
        ec_keypair,
        kid,
        {
            "sub": "agent:openexec-prod",
            "permissions": ["openexec.read", "openexec.admin"],
            "aud": "farthq-ecosystem",
            "exp": int(time.time()) + 3600,
        },
    )
    validator = _validator_with_cache("http://fake-iduna:8080", jwks())
    claims = await validator.verify(token)
    assert claims["sub"] == "agent:openexec-prod"


@pytest.mark.asyncio
async def test_guest_game_account_token_is_rejected_despite_valid_signature(
    ec_keypair, jwks_for
):
    """The exact real-world escalation adversarial review found: a public,
    unauthenticated guest game-account registration (IDUNA's own
    game_online.go guestToken()) mints a token with the SAME real audience
    every M2M agent token carries, but only a game-play permission — never
    openexec.*. A valid IDUNA signature alone must NOT be enough."""
    jwks, kid = jwks_for
    guest_token = _sign(
        ec_keypair,
        kid,
        {
            "sub": "guest:abc123",
            "permissions": ["shankpit.play"],
            "aud": "farthq-ecosystem",
            "exp": int(time.time()) + 3600,
        },
    )
    validator = _validator_with_cache("http://fake-iduna:8080", jwks())
    with pytest.raises(PermissionError):
        await validator.verify(guest_token)


@pytest.mark.asyncio
async def test_wrong_audience_token_is_rejected(ec_keypair, jwks_for):
    """A real end-user token from an unrelated game (aud="shankpit" per
    IDUNA's own shankpit_auth.go) must not verify against this service."""
    jwks, kid = jwks_for
    token = _sign(
        ec_keypair,
        kid,
        {
            "sub": "player:xyz",
            "permissions": ["openexec.read"],  # even if it somehow had this
            "aud": "shankpit",
            "exp": int(time.time()) + 3600,
        },
    )
    validator = _validator_with_cache("http://fake-iduna:8080", jwks())
    with pytest.raises(jwt.InvalidAudienceError):
        await validator.verify(token)


@pytest.mark.asyncio
async def test_token_with_no_exp_claim_is_rejected(ec_keypair, jwks_for):
    jwks, kid = jwks_for
    token = _sign(
        ec_keypair,
        kid,
        {
            "sub": "agent:openexec-prod",
            "permissions": ["openexec.read"],
            "aud": "farthq-ecosystem",
            # no "exp" at all
        },
    )
    validator = _validator_with_cache("http://fake-iduna:8080", jwks())
    with pytest.raises(jwt.MissingRequiredClaimError):
        await validator.verify(token)


@pytest.mark.asyncio
async def test_expired_token_is_rejected(ec_keypair, jwks_for):
    jwks, kid = jwks_for
    token = _sign(
        ec_keypair,
        kid,
        {
            "sub": "agent:openexec-prod",
            "permissions": ["openexec.read"],
            "aud": "farthq-ecosystem",
            "exp": int(time.time()) - 10,
        },
    )
    validator = _validator_with_cache("http://fake-iduna:8080", jwks())
    with pytest.raises(jwt.ExpiredSignatureError):
        await validator.verify(token)


@pytest.mark.asyncio
async def test_rsa_algorithm_would_have_rejected_this_same_real_jwk(ec_keypair, jwks_for):
    """Regression guard for the original bug: confirms the JWKS this test
    suite signs against is genuinely EC-shaped (not something the old,
    wrong RSAAlgorithm.from_jwk path could have coincidentally accepted)."""
    import json

    jwks, _ = jwks_for
    jwk = jwks()["keys"][0]
    with pytest.raises(Exception):  # noqa: B017 - InvalidKeyError from cryptography/pyjwt
        jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(jwk))


@pytest.mark.asyncio
async def test_concurrent_cold_start_fetches_collapse_to_one_real_network_call():
    """Adversarial-review-found gap: with no lock, N concurrent requests all
    hitting an empty cache would each independently decide "must fetch" and
    fan out to N simultaneous requests against IDUNA -- unauthenticated
    amplification against the shared IAM. The fetch lock must collapse them
    to exactly one real network call, with every waiter reusing its result."""
    cache = JWKSCache("http://localhost:8080")
    call_count = 0

    async def slow_get(url):
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.05)

        class _Resp:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict:
                return {"keys": []}

        return _Resp()

    with patch.object(cache._client, "get", side_effect=slow_get):
        results = await asyncio.gather(*[cache.fetch() for _ in range(10)])

    assert all(r == {"keys": []} for r in results)
    assert call_count == 1


@pytest.mark.asyncio
async def test_concurrent_fetches_during_an_outage_do_not_retry_storm():
    """Round-2 adversarial review found the round-1 cooldown only covered
    force=True -- the ordinary fetch() every request makes first had NO
    cooldown, so a flood of garbage bearer tokens during a real IDUNA
    outage turned into one real outbound fetch PER REQUEST. The cooldown
    must cover every real attempt, not just forced ones."""
    cache = JWKSCache("http://localhost:8080")
    call_count = 0

    async def failing_get(url):
        nonlocal call_count
        call_count += 1
        raise ConnectionError("IDUNA is down")

    with patch.object(cache._client, "get", side_effect=failing_get):
        results = await asyncio.gather(
            *[cache.fetch() for _ in range(50)], return_exceptions=True
        )

    assert all(isinstance(r, ConnectionError) for r in results)
    assert call_count == 1


@pytest.mark.asyncio
async def test_cooldown_short_circuit_still_fails_closed_past_staleness_ceiling():
    """CRITICAL regression, found in round-2 review, introduced by the fix
    above: the cooldown short-circuit returned the cached JWKS unconditionally
    once in cooldown, with no staleness check at all -- silently defeating
    _JWKS_MAX_STALE_S (a JWKS is this gate's trust anchor; it must eventually
    fail closed during a sustained outage, never trust a possibly-revoked key
    forever). This asserts a cache already past the ceiling is REJECTED even
    on the very next request inside the cooldown window, not just on the one
    request that triggers the real (failing) fetch attempt."""
    from openexecutive.auth.iduna_auth import _JWKS_MAX_STALE_S

    cache = JWKSCache("http://localhost:8080")
    cache._cache = {"keys": [{"kid": "old-key"}]}
    # Already well past the staleness ceiling.
    cache._fetched_at = time.monotonic() - (_JWKS_MAX_STALE_S + 3600)

    async def failing_get(url):
        raise ConnectionError("IDUNA is down")

    with patch.object(cache._client, "get", side_effect=failing_get):
        # First call: real fetch attempt, fails, past the ceiling -> raises.
        with pytest.raises(ConnectionError):
            await cache.fetch()
        # Second call, immediately after (inside the cooldown window): must
        # STILL raise -- not silently hand back the over-stale cache just
        # because a fresh attempt was rate-limited.
        with pytest.raises(ConnectionError):
            await cache.fetch()
