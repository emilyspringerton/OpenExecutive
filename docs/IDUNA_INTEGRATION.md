# OpenExecutive ↔ IDUNA ↔ Gemini Integration (Fast Shipping)

**Status:** ✅ Wired, tested, and adversarially reviewed (2 rounds) — updated 2026-09-21, see
[`../NORTHSTAR.md`](../NORTHSTAR.md) §6 for the full account of what was found and fixed (a critical
authorization gap, a tool-result correlation bug, and the smaller issues both review rounds caught).
The IDUNA side (`/api/v1/openexecutive/provision`) is real and live; the OpenExecutive side now has
a corrected JWT validator (EC algorithm, audience + `openexec.*` permission check, JWKS hardening),
a `google.genai`-based Gemini provider (now supporting EITHER Vertex AI via ADC OR the simpler
Gemini Developer API via a plain `GEMINI_API_KEY` — see below) registered in the provider registry,
and config/`.env.example` wiring. **Live-verified 2026-09-21**: EINHORN_INDUSTRIAL already had a
real, working Gemini Developer API key (`EMILY/var/gemini-api-key.env`) — ran the actual provider
code against it end to end and got the exact same `402 RESOURCE_EXHAUSTED` (billing credits
depleted, not an auth error) both via a raw `curl` and via `GeminiVertexProvider.messages_create`,
proving the request construction and auth wiring are correct. The remaining blocker to a real
completed response is smaller than originally scoped: top up billing on that key (or provision a
Vertex-mode GCP project instead), not "set up GCP/Vertex from scratch." No IDUNA-side live boot
test yet (needs a real IDUNA instance + provisioned credential — see the new `/admin/openexecutive`
Back Office page in IDUNA itself for that half). `packages/core/tests/unit/test_iduna_jwt_validator.py`,
`test_iduna_auth_gate.py`, and `test_gemini_vertex_provider.py` are real, passing, non-mocked-crypto
coverage. Treat everything below this line as the original, optimistic plan/estimate, not a status
report — see NORTHSTAR.md §6 for what's real.
**Effort:** 1-2 hours (original estimate; the real effort given what round-1 review found was
considerably more — see NORTHSTAR.md §6)
**Architecture:** OpenExecutive (Python FastAPI) → IDUNA (Go IAM) + Gemini (Developer API or Vertex AI)

---

## What's Wired

### IDUNA Side (Go)
✅ **Done** — `/api/v1/openexecutive/provision` endpoint  
- POST to provision M2M agent + one-time secret
- GET `/api/v1/openexecutive/health` for service checks
- Integrated into existing IAM store (CreateAgent, SetAgentCredential, GrantAgentPermission)
- Uses Google Application Default Credentials (ADC) from environment

### OpenExecutive Side (Python)
✅ **New files** (not yet deployed):
- `gemini_vertex_provider.py` — Gemini Vertex API adapter (translates Anthropic ↔ Gemini shapes)
- `auth/iduna_auth.py` — IDUNA JWT validation middleware

---

## Deployment Steps (Fast Path)

### 1. Clone OpenExecutive & Add Files
```bash
cd /home/fatbaby/OpenExecutive
cp /path/to/gemini_vertex_provider.py packages/core/openexecutive/providers/
cp /path/to/iduna_auth.py packages/core/openexecutive/auth/

# Update providers registry to include Gemini
# (Edit: packages/core/openexecutive/providers/__init__.py)
```

### 2. Configure Environment

Add to `.env` (or deploy container env vars):

```bash
# --- IDUNA Integration ---
IDUNA_URL=http://localhost:8080              # Or your IDUNA deployment
IDUNA_AGENT_NAME=openexec-prod               # M2M agent name
IDUNA_AGENT_SECRET=<generated-by-provision>  # From IDUNA POST /api/v1/openexecutive/provision

# --- Gemini: EITHER the Developer API key (simpler, no GCP Console work) ---
GEMINI_ENABLED=true
GEMINI_API_KEY=your-gemini-api-key
# --- OR Vertex AI via ADC (needs a real GCP project, Vertex AI enabled, billing on) ---
# GCP_PROJECT_ID=your-gcp-project-id
# GCP_LOCATION=us-central1
GEMINI_DEFAULT_MODEL=gemini-3.1-pro-preview
GEMINI_REASONING_MODEL=gemini-3.1-pro-preview

# Disable Anthropic (optional)
# ANTHROPIC_API_KEY=  # Leave unset or commented out
```

### 3. Provision IDUNA Agent

```bash
# Request M2M credentials from IDUNA
curl -X POST http://localhost:8080/api/v1/openexecutive/provision \
  -H "Authorization: Bearer <admin-jwt>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "openexec-prod",
    "permissions": ["openexec.read", "openexec.admin"]
  }'

# Response:
# {
#   "agent_id": "...",
#   "agent_name": "openexec-prod",
#   "secret": "64-char-hex-string",  ← Store this securely!
#   "jwks_url": "http://localhost:8080/.well-known/jwks.json"
# }

# Set IDUNA_AGENT_SECRET in .env
```

### 4. Boot OpenExecutive

```bash
cd packages/core
pip install -q google-cloud-aiplatform pyjwt  # New deps
uv sync  # Update lock file
source .venv/bin/activate

# Start FastAPI (port 8000) + Next.js UI (port 3000)
make dev
```

### 5. Test

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Query with Gemini (requires valid IDUNA JWT)
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer <iduna-jwt>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is our revenue growth strategy?"
  }'
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                        OpenExecutive                         │
│  (Python FastAPI + Next.js UI)                              │
│                                                              │
│  Uses IDUNAAuthMiddleware to validate inbound JWTs          │
│  Routes all Claude calls through GeminiVertexProvider       │
│  Maintains single coherent executive voice                  │
└────────────┬────────────────────────────────┬───────────────┘
             │                                │
             │ (1) Validate JWT              │ (2) Gemini calls
             │                                │
             ▼                                ▼
    ┌──────────────────────┐       ┌─────────────────────┐
    │  IDUNA (Go, :8080)   │       │  Gemini Vertex API  │
    │                      │       │  (Google Cloud)     │
    │ /.well-known/jwks.json│       │                     │
    │ /api/v1/openexec/... │       │ Uses ADC from env   │
    │                      │       └─────────────────────┘
    └──────────────────────┘
```

**Auth flow:**
1. OpenExecutive receives JWT in `Authorization: Bearer` header
2. Validates against IDUNA's JWKS endpoint (cached, async)
3. Extracts `sub` (agent_name), `permissions`, `roles` from token
4. Request proceeds if valid; 401 if not

**LLM flow:**
1. Agent call arrives (via FastAPI route)
2. System prompt + context passed to GeminiVertexProvider
3. Provider translates Anthropic kwargs → Gemini request
4. Gemini Vertex responds (uses ADC auth automatically)
5. Provider translates response back to Anthropic shape
6. Agent returns unified response

---

## Key Design Decisions

### Google Application Default Credentials (ADC)
- **Why:** No explicit API key needed; relies on environment (service account, gcloud login, etc.)
- **How:** `vertexai.init(project=..., location=...)` auto-discovers ADC
- **Deploy:** Either:
  - Run in GCP (Compute Engine, Cloud Run, etc.) with built-in service account
  - OR set `GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json` in environment
  - OR run `gcloud auth application-default login` locally

### IDUNA M2M Auth
- **Why:** Single source of truth for identity + permissions
- **How:** OpenExecutive's own agent credentials issued via IDUNA, same as Emily Prime or MJOLNIR
- **Benefits:** 
  - Unified audit trail (IDUNA's eventlog captures every auth event)
  - Can suspend OpenExecutive agent centrally if compromised
  - Permissions managed via IDUNA's own backlog/RBAC

### Anthropic-Shaped API (Provider Abstraction)
- **Why:** All existing agent code works unchanged; no refactoring needed
- **How:** GeminiVertexProvider translates messages + tools in/out
- **Fallback:** Keep ANTHROPIC_API_KEY set; if Gemini fails, can fall back to Claude (requires config)

---

## Testing Checklist

- [ ] `go build ./...` passes in IDUNA
- [ ] `make build` passes in OpenExecutive packages/core
- [ ] IDUNA boots with `systemctl start iduna` or `make dev`
- [ ] Can POST /api/v1/openexecutive/provision with admin JWT
- [ ] Can retrieve agent_secret + credentials
- [ ] OpenExecutive boots: `GEMINI_ENABLED=true` + Google ADC available
- [ ] OpenExecutive validates IDUNA JWT at `/.well-known/jwks.json` (check logs)
- [ ] Chat endpoint returns 401 without JWT; 200 with valid JWT
- [ ] Specialist agents route through Gemini Vertex (check logs for model name)
- [ ] Executive voice remains coherent (not Gemini-specific; abstracted away)

---

## If Gemini Fails (Fallback)

Keep `ANTHROPIC_API_KEY` set; update provider registry to prefer Anthropic:

```python
# packages/core/openexecutive/providers/__init__.py
if settings.anthropic_api_key:
    return AnthropicProvider(...)  # Try Claude first
elif settings.gemini_enabled:
    return GeminiVertexProvider(...)  # Fall back to Gemini
else:
    raise RuntimeError("No LLM provider configured")
```

---

## Cost Notes

- **Gemini:** ~$1.50/1M input tokens (flash) or ~$10/1M (pro) for text
- **Anthropic:** ~$3/1M input tokens (Sonnet) or ~$15/1M (Opus)
- **Cache impact:** Gemini's cache is less aggressive than Anthropic (~10% savings vs. 85% cache hit rate)

For a typical session (4-6 specialist calls, ~150K tokens):
- **Claude:** ~$0.45/session
- **Gemini:** ~$0.30/session (20% cheaper, but less cached)

---

## Files Changed

**IDUNA (Go):**
- `internal/http/handlers/openexecutive.go` — New handler
- `main.go` — Route registration

**OpenExecutive (Python, pending deployment):**
- `packages/core/openexecutive/providers/gemini_vertex_provider.py` — New provider
- `packages/core/openexecutive/auth/iduna_auth.py` — New auth middleware
- `packages/core/openexecutive/config.py` — Add Gemini + IDUNA settings (TBD)
- `.env.example` — New env vars (TBD)

---

## Next Steps

1. **Merge IDUNA changes** → `git push` in IDUNA (already done: commit 49e7f85)
2. **Add OpenExecutive files** → Copy `gemini_vertex_provider.py` + `iduna_auth.py` to repo
3. **Update config** → Add Gemini + IDUNA env vars to Settings class
4. **Wire provider registry** → Update `get_provider()` to support Gemini
5. **Boot test** → `make dev` with `GEMINI_ENABLED=true` + GCP creds
6. **Live test** → Provision agent via IDUNA, chat with Gemini backing

**Estimated time to ship:** 1-2 hours  
**Test coverage:** Manual E2E (no new unit tests; uses existing test infrastructure)
