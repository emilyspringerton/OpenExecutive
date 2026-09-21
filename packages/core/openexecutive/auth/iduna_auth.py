"""IDUNA JWT validation for OpenExecutive.

Allows OpenExecutive to validate JWTs issued by IDUNA (central IAM service),
enabling M2M authentication via IDUNA-issued credentials.

IDUNA issues ES256 JWTs (EC/P-256 keys — see ``IDUNA/internal/auth/jwt/jwks.go``,
which emits ``"kty": "EC", "crv": "P-256"``) with the public key available at
/.well-known/jwks.json.

**Authorization, not just authentication** (real, adversarial-review-found gap,
fixed here, not left as a TODO): IDUNA is EINHORN_INDUSTRIAL's shared IAM —
its ES256 key signs tokens for every principal in that ecosystem, not just
this service's own M2M agent. Checked directly against IDUNA's own handlers:
a public, unauthenticated guest-game-account registration
(``IDUNA/internal/http/handlers/game_online.go``'s ``guestToken``) issues a
token with the SAME ``aud: "farthq-ecosystem"`` every M2M agent token carries
(``IDUNA/internal/http/handlers/auth.go``'s ``AgentAuthHandler`` — the path
this service's own agent uses) — so ``audience`` validation ALONE does not
distinguish "an OpenExecutive-provisioned agent" from "a random public game
guest account". What genuinely differs is ``permissions``: a guest token only
ever carries a game-play permission (``cfg.PlayPerm``, e.g. ``"shankpit.play"``
— verified directly in ``guestToken``'s own claims), never one of the
``openexec.*`` permissions ``POST /api/v1/openexecutive/provision`` actually
grants. ``verify()`` below requires BOTH the expected audience (defense in
depth — narrows to IDUNA's service/M2M audience) AND at least one
``openexec.*``-prefixed permission (the real, load-bearing check) before a
token is accepted — without the second check, holding ANY valid IDUNA token
at all (including a self-registered guest account) would have been treated
as equivalent to holding this service's own ``BACKEND_SHARED_SECRET``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# How long a fetched JWKS is trusted before a re-fetch is attempted. IDUNA
# does not rotate its signing key on any fixed schedule today, but caching
# forever would mean a future key rotation silently locks this service out
# until restart — a bounded TTL keeps that self-healing without hitting
# IDUNA on every single request.
_JWKS_CACHE_TTL_S = 600.0

# Absolute ceiling on how long a fail-soft stale JWKS may keep being served
# once fetches start failing. A JWKS is the trust anchor for every request
# this gate accepts — unlike an ordinary optional external dependency, it
# must eventually fail CLOSED rather than trust a possibly-revoked key
# forever through a sustained IDUNA outage or a blackholed egress path.
_JWKS_MAX_STALE_S = 30 * 60.0

# An unauthenticated caller controls the JWT header (including `kid`) before
# any signature check runs. Without a cooldown, a flood of tokens carrying
# random unknown `kid`s would force a real outbound fetch to IDUNA per
# request — amplifying an unauthenticated request into load on the shared
# IAM. This bounds how often an unknown-`kid` retry actually reaches the
# network, independent of the ordinary TTL above.
_JWKS_FORCE_REFRESH_COOLDOWN_S = 5.0

# IDUNA's own fixed audience for service/M2M-issued tokens (every
# AgentAuthHandler-issued token AND every guestToken()-issued token both
# carry this — see this module's own top doc comment). Real, but
# intentionally not the whole check; see IDUNAJWTValidator.verify.
_DEFAULT_EXPECTED_AUDIENCE = "farthq-ecosystem"

# Only a permission string starting with this prefix authorizes a caller.
# Matches POST /api/v1/openexecutive/provision's own real example grant
# (["openexec.read", "openexec.admin"]) — the provisioning endpoint this
# service's own operator uses to mint an agent for it.
_REQUIRED_PERMISSION_PREFIX = "openexec."

# `kid` is attacker-controlled (read from the JWT header before any signature
# check). Bounding it before it reaches a log line or an exception message
# blocks log-injection (newlines/ANSI escapes) and unbounded-length flooding.
_MAX_LOGGED_KID_LEN = 64


def _safe_kid(kid: Any) -> str:
    """A `kid` value made safe to place in a log line or error message."""
    s = str(kid)[:_MAX_LOGGED_KID_LEN]
    return "".join(ch if ch.isprintable() else "?" for ch in s)


class JWKSCache:
    """In-memory cache for JWKS with lazy loading, a bounded TTL, a hard
    staleness ceiling, and a shared HTTP client (reused across fetches
    instead of opened fresh per call)."""

    def __init__(self, iduna_url: str, timeout_s: float = 30.0):
        self.iduna_url = iduna_url
        self.timeout_s = timeout_s
        self._cache: dict[str, Any] | None = None
        self._fetched_at: float = 0.0
        # Tracks every real attempt (force or not, success or failure) — the
        # cooldown below gates ALL network fetches, not just forced ones.
        # Round 2 adversarial review found the first version of this only
        # cooled down force=True: the ordinary fetch() every request makes
        # first had NO cooldown at all, so an unauthenticated flood of
        # garbage bearer tokens during an IDUNA outage (or a rolling
        # restart) turned into one real outbound fetch PER REQUEST — a
        # retry storm aimed at the one dependency already failing.
        self._last_attempt_at: float = 0.0
        self._last_error: BaseException | None = None
        self._client = httpx.AsyncClient(timeout=timeout_s)
        # Serializes real network fetches: without this, N concurrent
        # requests all hitting a cold (empty) cache would each independently
        # decide "no cache, must fetch" and fan out to N simultaneous
        # requests against IDUNA (found in adversarial review). Any request
        # that arrives while another is already fetching waits for that one
        # to finish and reuses its result instead of starting its own.
        self._fetch_lock = asyncio.Lock()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def fetch(self, *, force: bool = False) -> dict[str, Any]:
        """Fetch JWKS from IDUNA, caching the result for ``_JWKS_CACHE_TTL_S``.

        ``force=True`` bypasses a fresh cache — used when a token's ``kid``
        isn't found, so a key rotated since the last fetch is picked up
        within the same request instead of only after the TTL expires.

        EVERY real network attempt (forced or not) is rate-limited by
        ``_JWKS_FORCE_REFRESH_COOLDOWN_S`` so an unauthenticated caller
        cannot turn a flood of garbage bearer tokens into a fetch-per-request
        amplification attack against IDUNA — including during an IDUNA
        outage, when a cold/expired cache with no successful fetch to fall
        back on would otherwise retry on literally every request.
        """
        if self._is_fresh() and not force:
            assert self._cache is not None
            return self._cache

        if self._in_cooldown():
            if self._within_stale_ceiling():
                # Too soon since the last attempt (successful or not) —
                # serve whatever we have rather than hit the network again.
                # Gated on the SAME staleness ceiling as the real fetch-
                # failure path below: a cooldown-driven short-circuit must
                # never bypass "fail closed past _JWKS_MAX_STALE_S" (a real
                # regression found in round-2 review — the first version of
                # this branch returned the cache unconditionally once in
                # cooldown, silently disabling the ceiling for every request
                # except the ~1-per-cooldown-window that happened to escape
                # it and re-raise).
                assert self._cache is not None
                return self._cache
            # Either no cache at all yet (cold start), or the cache is past
            # the staleness ceiling: re-raise the last real failure instead
            # of trying again — the point of the cooldown is that a fresh
            # attempt this soon is not expected to behave differently, and
            # past the ceiling we must fail closed, not serve a stale key.
            if self._last_error is not None:
                raise self._last_error

        async with self._fetch_lock:
            # Double-checked: another coroutine may have already refreshed
            # the cache (or hit the same cooldown/failure) while this one
            # waited for the lock — reuse that outcome instead of issuing a
            # second real network fetch.
            if self._is_fresh() and not force:
                assert self._cache is not None
                return self._cache
            if self._in_cooldown():
                if self._within_stale_ceiling():
                    assert self._cache is not None
                    return self._cache
                if self._last_error is not None:
                    raise self._last_error

            self._last_attempt_at = time.monotonic()
            jwks_url = f"{self.iduna_url}/.well-known/jwks.json"
            try:
                resp = await self._client.get(jwks_url)
                resp.raise_for_status()
                self._cache = resp.json()
                self._fetched_at = time.monotonic()
                self._last_error = None
                logger.info(f"Fetched JWKS from {jwks_url}")
                return self._cache
            except Exception as e:
                # Store a fresh exception, not the exception object itself:
                # re-raising the SAME object on every subsequent cooldown
                # short-circuit appends a frame to its __traceback__ each
                # time (found in review — unbounded growth under a
                # sustained outage). Preserving the original TYPE (not just
                # the message) is deliberate and tested — a caller
                # distinguishing e.g. ConnectionError from TimeoutError
                # should still be able to after this rewrap.
                #
                # CORRECTED (2026-09-21, real live boot found this):
                # `type(e)(str(e))` assumes every exception class accepts a
                # single positional string — false for
                # `httpx.HTTPStatusError` (requires keyword-only `request`/
                # `response`), which is exactly what `raise_for_status()`
                # raises. That assumption crashed with a SECOND, unrelated
                # TypeError ("HTTPStatusError.__init__() missing 2 required
                # keyword-only arguments") that masked the real underlying
                # error (a 404 on IDUNA's own JWKS endpoint) behind a
                # useless one. Fix: try the type-preserving reconstruction;
                # fall back to a generic RuntimeError (with the original
                # type name folded into the message, so nothing is lost)
                # only for the exception classes that can't be rebuilt this
                # way, instead of assuming every class can.
                try:
                    self._last_error = type(e)(str(e))
                except Exception:
                    self._last_error = RuntimeError(f"{type(e).__name__}: {e}")
                if self._within_stale_ceiling():
                    # A transient IDUNA outage should not lock out every
                    # caller holding an otherwise-valid token — serve the
                    # stale cache rather than raising, but only up to
                    # _JWKS_MAX_STALE_S: a JWKS is this gate's trust anchor,
                    # not an ordinary optional dependency, so a sustained
                    # outage must eventually fail closed instead of trusting
                    # a possibly-revoked key forever.
                    assert self._cache is not None
                    stale_for = time.monotonic() - self._fetched_at
                    logger.warning(
                        f"JWKS re-fetch from {jwks_url} failed ({e}); serving "
                        f"stale cache ({stale_for:.0f}s old)"
                    )
                    return self._cache
                logger.error(f"Failed to fetch JWKS from {jwks_url}: {e}")
                raise

    def _is_fresh(self) -> bool:
        return self._cache is not None and (time.monotonic() - self._fetched_at) < _JWKS_CACHE_TTL_S

    def _within_stale_ceiling(self) -> bool:
        """True if the cache exists and is still within ``_JWKS_MAX_STALE_S``
        of its last successful fetch — the one condition under which ANY
        fail-soft path (cooldown short-circuit or a real fetch failure) may
        serve it. False (cache is ``None``, or past the ceiling) means fail
        closed: raise rather than trust a possibly-revoked key."""
        return (
            self._cache is not None
            and (time.monotonic() - self._fetched_at) < _JWKS_MAX_STALE_S
        )

    def _in_cooldown(self) -> bool:
        return (time.monotonic() - self._last_attempt_at) < _JWKS_FORCE_REFRESH_COOLDOWN_S


class IDUNAJWTValidator:
    """Validates JWTs issued by IDUNA — signature AND authorization (see
    this module's own top doc comment for why both are required).

    Usage:
        validator = IDUNAJWTValidator(iduna_url="http://localhost:8080")
        claims = await validator.verify(token)
    """

    def __init__(
        self,
        iduna_url: str,
        *,
        expected_audience: str = _DEFAULT_EXPECTED_AUDIENCE,
        required_permission_prefix: str = _REQUIRED_PERMISSION_PREFIX,
    ):
        self.iduna_url = iduna_url.rstrip("/")
        self.jwks_cache = JWKSCache(self.iduna_url)
        self.expected_audience = expected_audience
        self.required_permission_prefix = required_permission_prefix

    async def verify(self, token: str) -> dict[str, Any]:
        """Verify an IDUNA-issued JWT: signature, expiry, audience, AND that
        the token actually carries an ``openexec.*`` permission.

        Returns the decoded claims if valid.
        Raises an exception if the token is invalid, expired, wrong-audience,
        or lacks the required permission.
        """
        try:
            import jwt
        except ImportError as exc:
            raise RuntimeError(
                "PyJWT required for IDUNA token validation. "
                "Install with: pip install 'pyjwt[crypto]'"
            ) from exc

        key = await self._resolve_signing_key(token, jwt)

        verified = jwt.decode(
            token,
            key,  # type: ignore[arg-type]  # PyJWT's stubs omit EllipticCurvePublicKey
            # here even though it's the documented, correct ES256 verification
            # key type (cryptography.EllipticCurvePublicKey) — verified directly
            # with a real EC keypair + a real ES256-signed token round trip.
            algorithms=["ES256"],
            audience=self.expected_audience,
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_aud": True,
                # verify_exp only checks `exp` WHEN PRESENT — a token minted
                # without an exp claim at all would otherwise verify as
                # valid forever. `require` makes the claim's presence itself
                # part of the check.
                "require": ["exp"],
            },
        )

        self._require_permission(verified)

        logger.debug(f"IDUNA token verified for subject: {verified.get('sub')}")
        return verified

    async def _resolve_signing_key(self, token: str, jwt: Any) -> Any:
        """The real key-lookup half of ``verify``: header -> ``kid`` -> JWKS
        entry (retried once, forced, if the key looks rotated) -> a real
        ``cryptography`` EC public key object. Split out of ``verify`` so
        each concern (resolve the key vs. decode+authorize) is independently
        readable and testable.

        IDUNA signs with ES256 (EC/P-256) keys, not RSA — its own JWKS
        endpoint emits ``{"kty": "EC", "crv": "P-256", ...}``
        (``IDUNA/internal/auth/jwt/jwks.go``). ``ECAlgorithm`` is the
        matching PyJWT key-parsing class for that key type; ``RSAAlgorithm``
        expects ``{"n", "e"}`` fields an EC JWK never has and raises on this
        input.
        """
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")

        jwks = await self.jwks_cache.fetch()
        jwk = self._find_key(jwks, kid)
        if jwk is None:
            # The key may have rotated since our last fetch — force one
            # retry (itself cooldown-limited, see JWKSCache) before giving up.
            jwks = await self.jwks_cache.fetch(force=True)
            jwk = self._find_key(jwks, kid)
        if jwk is None:
            raise ValueError(f"Key ID {_safe_kid(kid)} not found in JWKS")

        return jwt.algorithms.ECAlgorithm.from_jwk(json.dumps(jwk))

    def _require_permission(self, claims: dict[str, Any]) -> None:
        """The real authorization check: a valid IDUNA signature alone is
        NOT sufficient (see this module's own top doc comment) — the token
        must also carry a permission this service's own operator actually
        granted via ``POST /api/v1/openexecutive/provision``.

        **Known, deliberate v0 limitation (round-2 adversarial review):** this
        is all-or-nothing — ANY ``openexec.*``-prefixed permission (including
        the narrower ``openexec.read`` IDUNA's own provisioning endpoint
        documents as a real, separate grant from ``openexec.admin``) passes
        this check and gets full API access; nothing downstream reads
        ``request.state.iduna_claims`` to enforce read-only routes for a
        ``read``-only token. Real per-route/per-scope authorization (mapping
        ``openexec.read`` to safe methods only) is a genuinely larger change
        (auditing every route in ``api/routes/`` for a read/write
        classification) that risks introducing its own bugs if rushed under
        time pressure — named here rather than attempted half-built. An
        operator provisioning a token today should treat ``openexec.read``
        and ``openexec.admin`` as equivalent for THIS service until that
        work lands.
        """
        permissions = claims.get("permissions") or []
        if not isinstance(permissions, list) or not any(
            isinstance(p, str) and p.startswith(self.required_permission_prefix)
            for p in permissions
        ):
            sub = claims.get("sub", "?")
            raise PermissionError(
                f"IDUNA token for subject {sub!r} has no "
                f"{self.required_permission_prefix!r}-prefixed permission"
            )

    @staticmethod
    def _find_key(jwks: dict[str, Any], kid: str | None) -> dict[str, Any] | None:
        for k in jwks.get("keys", []):
            if k.get("kid") == kid:
                return k
        return None
