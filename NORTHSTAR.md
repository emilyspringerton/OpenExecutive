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

## 6. Phase A-C done (2026-09-21, founder real-time: "get it up in IDUNA asap")

All of Phase A, all of Phase B, and the "prove it" half of Phase C are done — real, tested,
adversarially reviewed (2 rounds: 3 reviewers found real CRITICAL/HIGH issues, all fixed and
re-verified by a second round of the 2 reviewers that found them). This turned up MORE than the
critical review above anticipated — reviewing real, working code surfaces bugs a read-only audit
of broken code can't. In order found:

- **Finding #2's tool-schema/response-shape fixes**, done against `google.genai` (the CURRENT SDK
  — `vertexai.generative_models`, which the original broken code used, turned out to already be
  past its own documented deprecation-removal date; building new code against it would have been
  wrong on day one, found by actually importing it and reading the warning).
- **A real, third bug beyond the original two**: the system prompt (always a list of blocks in
  real calls, e.g. `orchestrator/executive.py`'s `system=system_blocks`) was being silently dropped
  because the first fix pass only handled a bare string — found by checking real call sites, not
  assumed correct once the obvious cases worked.
- **Round 1 adversarial review found a CRITICAL authorization gap** the original scoping pass
  didn't anticipate: a valid IDUNA *signature* was being treated as sufficient authorization, but
  IDUNA's ES256 key also signs low-trust principals (public guest game accounts verified directly
  in `IDUNA/internal/http/handlers/game_online.go`) with the SAME audience every M2M agent token
  carries — so a self-registered guest account could have called this entire API. Fixed by
  requiring an `openexec.*`-prefixed permission claim (what `POST /api/v1/openexecutive/provision`
  actually grants), not just a valid signature.
- **Round 1 also found a second correctness bug in the Gemini fix**: `tool_result` blocks were
  keyed by Anthropic's opaque `tool_use_id` instead of the real function name Gemini needs to
  correlate a response to its call — every multi-turn tool conversation would have silently broken.
  Fixed via an id→name map threaded through the whole message list.
- Plus: duplicate tool-call ids across turns, `exp` claim not required, a `type: None` crash, plain-
  HTTP JWKS fetch permitted, unbounded stale-JWKS serving, JWKS-fetch amplification with no lock
  or cooldown, a non-ASCII `x-api-key` 500, dead/misleadingly-named code (`IDUNAAuthMiddleware`,
  never wired to anything), nondeterministic Council-dropdown ordering, and doc gaps
  (`architecture/prebuilt/caching.json`, `.env.example`'s Gemini-only deployment guidance).
- **Round 2 (security + logic reviewers, re-checking their own round-1 findings against the
  fixes) found MORE real issues — including a regression the round-1 fix itself introduced.**
  Security: the amplification-cooldown fix only covered `force=True`, leaving the ordinary
  (non-forced) fetch path — the one every request actually hits first — with no cooldown at all
  during an IDUNA outage (a real retry-storm risk); the `openexec.*` permission check is
  all-or-nothing (`openexec.read` gets the same full access as `openexec.admin`, contradicting
  IDUNA's own documented read/admin scope split); the httpx client was never closed on shutdown;
  the new `IDUNA_EXPECTED_AUDIENCE`/`IDUNA_REQUIRED_PERMISSION_PREFIX` settings were undocumented.
  Logic: **the cooldown-universalization fix from the security round-2 pass regressed the
  staleness ceiling** — the cooldown short-circuit returned a stale-past-ceiling cache
  unconditionally, silently defeating "must fail closed during a sustained outage"; **the stated
  premise for the tool-result fix was factually wrong** — `FunctionCall`/`FunctionResponse` both
  carry a real `id` field for correlation (verified directly against the installed SDK), so
  name-only correlation breaks this codebase's own designed *parallel* tool-fan-out pattern
  (two same-name calls in one turn); thinking tokens (a separate, unavoidable-on-2.5-Pro billed
  field) were dropped from `output_tokens`, undercounting real cost 10-50x in testing; a truncated
  function call reported `stop_reason="tool_use"` instead of `"max_tokens"`; a non-ASCII shared
  secret compared against the wrong byte encoding (UTF-8 vs. the latin-1 Starlette/h11 actually
  decode headers as); re-raising the same cached exception object grew its traceback unboundedly;
  `google-genai` was only present transitively; the docs' own "additional… never a replacement"
  language for IDUNA auth was inaccurate — the real gate is an OR, and an IDUNA-only deployment
  with no shared secret at all is an intentional, real deployment shape, not an oversight.
  **All of the above are now fixed and covered by new, real tests** (8 additional regression tests:
  the exact stale-cache-inside-cooldown scenario, the exact parallel-same-name-tool-call scenario,
  thinking-token accounting, MAX_TOKENS-vs-tool_use priority, no-candidates usage reporting).

Real, honest, still-deferred (named, not silently dropped): real token-level Gemini streaming
(the wrapper runs a correct non-streaming call and re-chunks it — still true, unchanged from Phase
A's own scoping); Gemini gets no `tool_choice`/`thinking`-config/per-call `timeout` translation
(round-2's own F5 — the `FeatureSpec`/feature-gate system OpenRouter and local models both go
through has no Gemini equivalent yet); per-route/per-scope authorization for `openexec.read` vs.
`openexec.admin` (a genuinely larger change — auditing every route in `api/routes/` for a
read/write classification — named rather than rushed; `request.state.iduna_claims` is already
threaded through for when this lands); `core/auth/`'s own architecture-doc registration (kept as a
plain namespace package, deliberately, rather than adding `__init__.py` and triggering this repo's
own "new top-level module needs a SectionSpec + page.tsx entry + prebuilt json" requirement for
what is a narrow, internal auth mechanism, not a user-facing feature domain); Corporate Service
Call-style automatic Anthropic-Gemini fallback (open decision #13 above, unchanged); the stray
`iduna.db` cleanup and upstream-tracking decision (Phase D, unchanged); real eval scenarios per
this repo's own PR bar (none added — no new specialist agent or prompt changed, so the existing
gate doesn't require them).

## 7. "Is it ready?" — found a real credential, ran it live (2026-09-21)

Founder pushed back on §6's own "this sandbox has neither [GCP nor IDUNA credentials]" claim,
correctly: EINHORN_INDUSTRIAL already had a live, working Gemini Developer API key
(`EMILY/var/gemini-api-key.env`, project `einhorn-mjolnir`) sitting in this monorepo's own
established interim-secrets convention — a check I skipped, having only tried `gcloud auth
login`/ADC. Real correction, not just a doc fix:

- `GeminiVertexProvider` now supports the Gemini Developer API (`api_key=`, `generativelanguage.
  googleapis.com`) as a real alternative to Vertex AI (`project_id=`, ADC) — `google.genai.Client`
  supports both natively, verified against the real installed SDK's own constructor signature.
  `api_key` takes priority when both are set. `config.py`'s `_validate_gemini` now accepts either.
- **Ran the actual provider code against the real key**, not just curl: `GeminiVertexProvider(
  api_key=...).messages_create(...)` returns the exact same `402 RESOURCE_EXHAUSTED` (billing
  credits depleted) a raw `curl` to the same endpoint returns — proof the real request
  construction and auth wiring are correct, not just that the key itself is valid. The stale
  default model name this surfaced (`gemini-2.5-pro`, "no longer available to new users" per
  Google's own error) is fixed to `gemini-3.1-pro-preview` everywhere it was hardcoded.
- **Corrected scope of "ready"**: the real remaining blocker is billing credits on an *existing*
  key (a small, human-only top-up at ai.studio/projects), not "provision a whole new GCP project
  with Vertex AI enabled" — §6's framing overstated the gap.
- 3 new tests (api_key-only construction, api_key-takes-priority, raises with neither credential);
  all 22 provider tests + the full unit suite green; ruff/mypy clean.

## 8. First real, successful end-to-end completion (2026-09-21, same day)

The original key's account never got billing resolved (AI Studio prepayment credits, distinct
from regular GCP Cloud Billing — clicking something in Cloud Console didn't touch it). Founder
supplied a different account's key instead. Getting a real response required going all the way
through — not stopping at unit tests — and found three more real bugs along the way, invisible to
every existing test because none of them exercised the real, full path:

- **IDUNA's own public JWKS endpoint had a real 404** — `https://okemily.com/.well-known/jwks.json`
  was never actually proxied by nginx (fixed in IDUNA `dc5226d`+`sudo-queue/86`, a genuine,
  previously-undiscovered infra gap affecting any external IDUNA-JWT consumer, not just this repo).
- **`iduna_auth.py`'s own error-wrapping had a real bug** masking that 404 behind a confusing,
  unrelated `TypeError` (`type(e)(str(e))` assumed every exception's constructor takes one
  positional string — false for `httpx.HTTPStatusError`). Fixed with a type-preserving-when-possible
  fallback, verified against a *real* `httpx.HTTPStatusError`, not a stand-in.
- **`GeminiVertexProvider._translate_tools` had two more real bugs**, both only visible once real
  specialist tool schemas (not hand-picked test fixtures) were sent through it: a Pydantic-style
  nullable-type schema (`{"type": ["integer", "null"]}`) crashed `FunctionDeclaration` construction
  outright, and — one layer deeper, only visible once that stopped masking it — a schema carrying
  `additionalProperties` validated fine but failed the *actual Gemini API call* with a real `400`,
  because the installed SDK's own `Schema.model_dump()` emits the Python field name instead of the
  JSON alias for that one field (confirmed directly against the SDK). Both fixed in a new
  `_sanitize_schema_for_gemini`, applied recursively. Full account, tests, and the exact fix in
  OpenExecutive `40ae9ed`.

Also switched the model default from `gemini-3.1-pro-preview` to `gemini-flash-latest`: the new
account's free tier has zero quota for the `pro` model specifically (a real `429`, `limit: 0`) —
flash works today; upgrading past free tier would lift that, not attempted here.

**A real chat request, through the full real stack — real IDUNA-issued JWT, `IDUNAJWTValidator`,
the orchestrator, real specialist tool schemas, `GeminiVertexProvider`, real Gemini — returned a
real, correct response for the first time.** Not simulated, not mocked: `curl` against a locally
booted `uvicorn openexecutive.api.main:app` with a JWT minted by a live IDUNA instance. This is the
actual, complete finish line for what NORTHSTAR originally scoped — IDUNA's own live boot/deploy
(beyond this one manually-booted local process) is real, separate, follow-on infrastructure work,
not blocked on anything found in this pass.

## 9. "openexec is having a lot of problems" — the real cause, and billing unblocked (2026-09-21, same day)

Founder: "openexec is having a lot of problems can you check the logs or anything? otherwise
assuming we need the pro model i paid them it should be possible now."

**The real cause, found by actually reading `journalctl --user -u openexecutive-api`, not
guessing**: essentially every chat turn that triggered a tool call was failing outright on its
SECOND Gemini API call with `400 INVALID_ARGUMENT: Function call is missing a thought_signature in
functionCall parts`. Gemini's thinking-enabled models attach an opaque `bytes` `thought_signature`
to the `Part` that carries a `function_call` (confirmed directly against the installed SDK's own
`Part`/`FunctionCall` field lists — it lives on `Part`, not `FunctionCall`), and require that exact
signature echoed back verbatim when the call is replayed as conversation history on the next
iteration of the same tool-use loop. This codebase's internal Anthropic-shaped `tool_use` block —
built when the orchestrator only targeted the Anthropic API — had nowhere to carry that field, so
it was silently dropped on every response, breaking any multi-step tool-using turn (i.e. almost
every real chat message, since `list_department_goals` and friends fire on iteration 1 of nearly
every question). This was **the actual "lot of problems"**, not a vague or unreproducible
complaint.

Fixed by threading a `gemini_thought_signature` field (base64-encoded, since `bytes` isn't
JSON-serializable) through the round trip in all FOUR places a tool call gets replayed as history:
`orchestrator/executive.py`'s main chat loop, and the two independent copies of the same pattern in
`workflows/executive_research.py` and `workflows/executive_reflection.py` (found only by an
independent second-model adversarial review of the first fix — the initial pass fixed the chat
path and missed that the other three workflows do their own tool-use replay rather than sharing
`executive.py`'s). Also closed, same review pass: a `Part` carrying both `text` and `function_call`
in the same response would have silently dropped the function call (an early `continue` after the
text branch) — latent on Gemini 2.5 (which never sends both on one `Part`), but removed rather than
left as a landmine for a future model that might.

4 new tests in `tests/unit/test_gemini_vertex_provider.py`, built against the real installed SDK
types (not mocks) — capturing a real signature, confirming `None` when absent, and round-tripping
one through `_build_contents` back onto a reconstructed `Part`. Full test suite green (3669 passed,
1 skipped, only the pre-existing known-red `test_chat_with_committee_streams_phases_and_revised_text`
failing — same as before this change, unrelated). `ruff`/`mypy` clean on every touched file.

**Billing, separately**: the founder paid to unblock the pro-tier model on the current Gemini key's
account (the same account §8 switched to after the original's credits ran out). Live-verified
directly (a real `generate_content` call, not just a docs check): `gemini-3.1-pro-preview` now
succeeds where it previously 429'd with `RESOURCE_EXHAUSTED, limit: 0`. `gemini-2.5-pro` itself now
404s for new users ("no longer available ... use models/gemini-3.1-pro-preview" — the API's own
message). `.env`'s `GEMINI_DEFAULT_MODEL`/`GEMINI_REASONING_MODEL`/`DEFAULT_MODEL`/
`DEEP_REASONING_MODEL` moved back to `gemini-3.1-pro-preview` (matching `config.py`'s own original,
never-changed code defaults — the flash downgrade was always meant to be temporary);
`ROUTING_MODEL` stays on `gemini-flash-latest` on purpose, matching the cheap/fast-routing,
strong/slow-reasoning split this repo's own Claude model defaults already use (haiku routes, opus
reasons).
