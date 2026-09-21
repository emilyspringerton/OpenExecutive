# OpenExecutive — IDUNA + Vertex integration NORTHSTAR (S506, 2026-09-21)

Founder real-time (2026-09-20/21, relayed via the IDUNA kanban thread): fork
[SenteLabsAI/OpenExecutive](https://github.com/SenteLabsAI/OpenExecutive) into this monorepo's
ecosystem (`emilyspringerton/OpenExecutive`), wire it to **IDUNA** for identity/auth the same way
every other agent in this monorepo does, and migrate its LLM backbone from calling Anthropic
directly to **Google Vertex AI (Gemini)**. Claude Haiku 4.5 did the first real pass on this
(2026-09-20) without going through the Emily Way (no BACKLOG.md entry, no golden-doc registration,
work sitting untracked on `main`, the repo itself cloned into `/tmp`) — this doc is the real,
checked-not-assumed scoping pass that was skipped: what exists, a critical read of what's already
written, and a phased next-steps list. Per Principle 19 this is exactly the kind of unscoped,
partially-built work that needs a real accounting before more code gets layered on top of it.

## 1. What this actually is

**Open Executive** is a real, substantial, external open-source project (Apache 2.0,
`sentelabs.ai`) — not something built in this monorepo. It's a multi-agent "virtual corporate
executive": one coherent persona backed by eight specialist sub-agents (CSO/CFO/CHRO/GC/COO/CMO/
CPO/Board), Python FastAPI + Next.js 15, ChromaDB for RAG over built-in MBA knowledge + uploaded
company docs, SQLite for episodic memory, a background scheduler, and real integrations (Slack,
email, Telegram, Google Chat, Discord, MCP server/gateway). It ships with an existing, mature
multi-provider abstraction (`packages/core/openexecutive/providers/`) — Anthropic direct,
OpenRouter (with a live model catalog fetch), and any OpenAI-compatible local/hosted endpoint are
already real and working, all behind one `LLMProvider` protocol
(`providers/provider.py`: `messages_create`/`messages_stream`, both Anthropic-shaped). This
monorepo's fork (`emilyspringerton/OpenExecutive`, 155+ merged upstream PRs of history) had two
real, new things added on top by Haiku's pass:

1. **IDUNA-side**: `POST /api/v1/openexecutive/provision` (admin-only, mints an M2M agent +
   one-time secret via IDUNA's existing IAM store) and `GET /api/v1/openexecutive/health` — real,
   committed, wired into `main.go` (IDUNA commit `49e7f85`, verified live in the route table).
2. **OpenExecutive-side**: two new, **untracked** Python files —
   `packages/core/openexecutive/providers/gemini_vertex_provider.py` (a fourth `LLMProvider`
   implementation, targeting Vertex AI's Gemini models) and
   `packages/core/openexecutive/auth/iduna_auth.py` (a JWT validator meant to let OpenExecutive
   trust IDUNA-issued tokens the same way every other service in this monorepo does).

## 2. Process fixes made in this pass (before any more code)

- **Relocated the repo off `/tmp`.** It was cloned to `/tmp/OpenExecutive` — on this box that's
  real disk (same filesystem as `/home`, confirmed via `findmnt`), not tmpfs, but `/tmp` is still
  routinely swept by `systemd-tmpfiles` (commonly anything untouched 10+ days) and is not where
  any other repo in this monorepo lives. Moved (same-filesystem rename, git history + the two
  untracked files all verified intact before/after) to `/home/fatbaby/OpenExecutive`, matching
  every other repo's convention.
- **Relocated the integration doc.** `OPENEXECUTIVE_INTEGRATION.md` was sitting untracked in
  `DEADWEIGHT/` (an unrelated card-game repo) with no reference to it from anywhere in that
  repo — clearly just wherever Haiku's shell happened to be. Moved to
  `OpenExecutive/docs/IDUNA_INTEGRATION.md` and its `/tmp/OpenExecutive` path reference fixed.
  Its own header claimed **"Status: ✅ Ready to deploy"** — false per the critical review below
  (SAGA reconciliation: a doc's claim must match reality), corrected to point here.
- **Root `CLAUDE.md` repo table**: added a row for `OpenExecutive` (see that file's own diff).
- **`EMILY/BACKLOG.md` SECTION 506** and **`EMILY/context/golden-docs-index.md`**: this doc
  registered, so Emily Prime's own RSI cycle can actually see it (it couldn't have, sitting in
  `/tmp` with no index entry).
- **Not yet done, named**: `OpenExecutive` doesn't have its own monorepo-flavored `CLAUDE.md`
  section here in NORTHSTAR's home repo the way e.g. `BIG_O/CLAUDE.md` does — the repo already
  ships its own upstream `CLAUDE.md` (contributor-facing, describes the codebase itself well) and
  it would be wrong to overwrite that; if this repo needs an *Emily Way* operating-agreement
  layer on top (Apple filing, session trailers, `emily observe` routing), that's a small, real,
  separate follow-up, not done here.

## 3. Critical review — what's real vs. what's currently broken (checked, not assumed)

Per this monorepo's own standing convention (BIG_O/DEADWEIGHT/LO's own NORTHSTAR docs), a critical
read before anything gets treated as done. Everything below was verified directly against source,
not inferred from the integration doc's own (optimistic) claims.

### 3a. Two concrete, confirmed bugs — either one alone would break the integration

1. **Wrong JWT key algorithm.** `iduna_auth.py` does
   `jwt.algorithms.RSAAlgorithm.from_jwk(json.dumps(k))` on keys fetched from IDUNA's
   `/.well-known/jwks.json`. IDUNA's own key material is **EC, not RSA** — confirmed directly in
   `IDUNA/internal/auth/jwt/jwks.go`: `"kty": "EC", "crv": "P-256", ...`, matching IDUNA's own
   documented ES256 signing (`IDUNA/CLAUDE.md`: "ES256 JWT... on `crypto/ecdsa`"). PyJWT's
   `RSAAlgorithm.from_jwk` expects `n`/`e` fields and will raise on an EC-shaped JWK (`crv`/`x`/
   `y`) — **every real IDUNA-issued token would fail validation**, not just untested-but-plausible.
   Fix: `jwt.algorithms.ECAlgorithm.from_jwk(...)`, and `algorithms=["ES256"]` in the final
   `jwt.decode` call is already correct, only the key-parsing class is wrong.

2. **Wrong tool schema shape.** `gemini_vertex_provider.py`'s `_translate_tools` checks
   `tool.get("type") == "function"` and reads `tool["function"]` — that's the **OpenAI/OpenRouter**
   tool shape. The real Anthropic-shaped `tools[]` this provider is contractually handed (per
   `provider.py`'s own doc comment: "kwargs the same kwargs
   `anthropic.AsyncAnthropic().messages.create` accepts") has **no `type`/`function` wrapper at
   all** — flat `{"name", "description", "input_schema"}`. Confirmed directly: the existing
   `translator.py` has a whole function, `_anthropic_tools_to_openai`, whose entire job is
   converting FROM this real flat Anthropic shape TO the OpenAI-wrapped shape for the OpenRouter/
   OpenAI-compatible providers — proving what real inbound tool defs look like. As written,
   `_translate_tools` will **always return an empty list** against real input, silently dropping
   every tool on every Gemini-backed call. Since the entire specialist-routing architecture
   (`consult_specialist`) is tool-use-based per this repo's own README, **a Gemini-backed
   Executive could never actually consult a specialist** — it would just talk to itself.

### 3b. Real, live-found design gaps (not fatal individually, but real)

3. **System prompt handling breaks this repo's own stated-critical caching design.** Rather than
   Vertex's native `GenerativeModel(model_name, system_instruction=...)`, `_build_contents`
   prepends the system prompt as a synthetic `role="user"` message. This repo's own README/
   CLAUDE.md calls prompt caching architecture "critical" ("breaking caching = 10x cost increase")
   and structures the whole system prompt specifically to be cached separately from dynamic
   content — none of that applies here at all; no Vertex context-caching equivalent is used
   either. Every Gemini-backed turn re-sends and re-processes the full persona/company-profile/
   knowledge-index text as ordinary conversation content.
4. **No `tool_result` → Gemini translation.** `_build_contents` only handles outbound `tool_use`
   blocks (serialized as JSON text, not even a real Gemini function-call `Part`); there is no
   handling of inbound `tool_result` content blocks at all (the thing that actually carries a
   specialist's answer back into the conversation after a call). Multi-turn tool use — this app's
   core interaction pattern — isn't functionally supported, on top of finding #2 above already
   preventing tools from being offered in the first place.
5. **`messages_stream` isn't real streaming.** It's a synthetic wrapper that runs one full
   non-streaming call and re-emits fabricated stream events afterward — the code says so itself
   ("Full streaming support is a follow-on enhancement"). Listed here for completeness, not urgent
   relative to #1/#2.

### 3c. Zero wiring — none of the above is reachable today even if it were bug-free

6. **`GeminiVertexProvider` is never registered.** `providers/registry.py`'s provider-selection
   logic has no reference to it anywhere — the only Gemini-related strings there are OpenRouter
   catalog model slugs (`google/gemini-3.8-flash`, a completely different code path: routing
   Gemini *through* OpenRouter, not the native Vertex adapter). `get_provider()` cannot currently
   return a `GeminiVertexProvider` under any configuration.
7. **No config surface.** `config.py`'s `Settings` class has zero `GEMINI_*`/`GCP_*`/`IDUNA_*`
   fields, and `.env.example` documents none of the variables the integration doc describes
   (`GEMINI_ENABLED`, `GCP_PROJECT_ID`, `GCP_LOCATION`, `GEMINI_DEFAULT_MODEL`,
   `GEMINI_REASONING_MODEL`, `IDUNA_URL`, `IDUNA_AGENT_NAME`, `IDUNA_AGENT_SECRET`) — they're
   real-sounding but currently read by nothing.
8. **`IDUNAAuthMiddleware` is applied to zero routes.** Grepped `packages/core/openexecutive/api/`
   directly — no reference to it anywhere. It's a real, freestanding, unused class today.
9. **Missing dependencies.** Neither `google-cloud-aiplatform` nor `pyjwt` appear in
   `packages/core/pyproject.toml`. The integration doc's own step 4 says `pip install -q
   google-cloud-aiplatform pyjwt` as a manual, undeclared step — not how this repo's own real,
   reproducible `uv sync` build works; a fresh `uv sync` today would not install either package.

### 3d. Process debt found in the IDUNA side of this same work

10. **A stray, empty `iduna.db` file is committed** in IDUNA (commit `49e7f85`, still tracked
    today — confirmed via `git ls-files`). Almost certainly an accidental `git add` from running
    IDUNA locally with a default DB path. IDUNA's `.gitignore` has no `*.db` rule at all. Should be
    `git rm`'d and a rule added — real, separate, small IDUNA cleanup, not blocking this repo.
11. **That same IDUNA commit bundled unrelated changes** — the openexecutive handler alongside
    `nock_animations.go`, `cmd/mktutoriallevel/main.go`, and an unrelated GFD SQL migration, all in
    one `feat(openexecutive)` commit. Violates this monorepo's own "one atomic thing per commit"
    rule (`CLAUDE.md` §6). Not worth rewriting IDUNA history over at this point; naming it so it
    isn't repeated.
12. **New OpenExecutive files sat untracked directly on `main`**, not a feature branch — this
    upstream project's own `.github/CONTRIBUTING.md` requires PRs to carry "working implementation
    (no stubs)", tests, and eval scenarios for exactly this kind of change. Given findings #1-#9
    above, none of that bar was met yet; going forward this work should live on a branch/PR like
    every other change to this repo, per its own contribution rules.

## 4. Security/scope note (real, worth naming, not yet a decision needed)

Open Executive is a broad system with real outbound integrations (Slack/email/Discord/Telegram/
Google Chat), document upload, and an MCP server exposing company context + a `consult_specialist`
tool to external MCP clients. Wiring its identity to IDUNA is exactly the right move (this
monorepo's own standing rule: "IDUNA is the central trust authority... never trust tokens from
other sources") — but it also means an OpenExecutive M2M credential now sits inside this
monorepo's real trust boundary. `openexecutive.go`'s `ProvisionRequest.Permissions` accepts
arbitrary caller-supplied permission strings with no whitelist check before granting them — this
matches IDUNA's existing general M2M-provisioning pattern elsewhere (not a regression introduced
here), but is worth a founder-level look given how much surface area this particular agent has,
whenever real provisioning (not the fast-shipping doc's illustrative example) actually happens.

## 5. Phased next steps

**Phase A — make the Gemini path actually correct (blocking, do first, in this order):**
1. Fix `iduna_auth.py`: `ECAlgorithm.from_jwk`, not `RSAAlgorithm.from_jwk` (finding #1).
2. Fix `gemini_vertex_provider.py`'s tool translation: read the real Anthropic `input_schema`
   shape (no `type`/`function` wrapper), and add real inbound `tool_result` → Gemini
   `function_response` `Part` handling for outbound `tool_use` too (finding #2 + #4).
3. Switch system-prompt handling to Vertex's native `system_instruction=` (finding #3).

**Phase B — wire it up (blocking, nothing above matters until this exists):**
4. Add `GEMINI_ENABLED`/`GCP_PROJECT_ID`/`GCP_LOCATION`/`GEMINI_DEFAULT_MODEL`/
   `GEMINI_REASONING_MODEL`/`IDUNA_URL`/`IDUNA_AGENT_NAME`/`IDUNA_AGENT_SECRET` to `config.py`'s
   `Settings` + document them in `.env.example` (finding #7).
5. Register `GeminiVertexProvider` in `providers/registry.py`'s `get_provider()` selection logic
   (finding #6).
6. Apply `IDUNAAuthMiddleware` to the routes that should require it — **open decision**: which
   routes (chat/API certainly; `/health` almost certainly not) — not yet decided anywhere.
7. Add `google-cloud-aiplatform` and `pyjwt` to `packages/core/pyproject.toml`, regenerate the
   lock file (finding #9).

**Phase C — prove it, per this repo's own real contribution bar:**
8. Live boot test end-to-end (the integration doc's own existing Testing Checklist is a reasonable
   starting script for this).
9. Add the eval scenarios/tests this upstream project's own `CONTRIBUTING.md` requires for a
   provider change — currently zero exist for the Gemini path.
10. Open a real PR against this fork's `main` (per finding #12), not a direct-to-`main` untracked
    diff.

**Phase D — cleanup + open decisions:**
11. Remove the stray `iduna.db` from IDUNA git, add a `*.db` `.gitignore` rule (finding #10).
12. **Open decision for the founder**: does this fork track upstream `SenteLabsAI/OpenExecutive`
    (regular merges/rebases — real cost given 155+ PRs of shared history) or intentionally diverge
    once IDUNA/Gemini-specific changes land? Worth deciding before drift makes it painful either way.
13. **Open decision**: the integration doc's own "if Gemini fails, fall back to Anthropic" idea
    has no implementation anywhere — worth deciding whether that's actually wanted (adds real
    provider-selection complexity) or whether Gemini-vs-Anthropic should just be one operator
    choice via `DEFAULT_MODEL`/`GEMINI_ENABLED`, no automatic fallback.

Nothing in Phase A-C has been started as part of writing this NORTHSTAR — this pass is scoping and
process only, per the founder's own explicit ask ("fill in process and write the northstar...
come up with the next steps").
