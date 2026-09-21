"""Gemini provider adapter for OpenExecutive.

Translates OpenExecutive's Anthropic-shaped API calls into Google Gen AI SDK
calls, and translates responses back to an Anthropic ``Message``-shaped
(duck-typed) object for compatibility — the same ``types.SimpleNamespace``
pattern ``providers.translator.from_openai_response`` already uses for the
OpenRouter/local-model path.

Deliberately built on ``google.genai`` (the unified Gen AI SDK), NOT
``vertexai.generative_models``: importing the latter emits ``UserWarning: This
feature is deprecated as of June 24, 2025 and will be removed on June 24,
2026`` (verified live against the installed ``google-cloud-aiplatform``
2.1.3) — building new code against an SDK surface already past its own
documented removal date would be wrong on day one.

Supports EITHER of ``google.genai.Client``'s two real backends — Vertex AI
(``vertexai=True, project=, location=``, Application Default Credentials) or
the Gemini Developer API (``api_key=``, ``generativelanguage.googleapis.com``,
no GCP project/ADC/billing-console work beyond whatever created the key). See
``GeminiVertexProvider``'s own docstring for when each applies; the class name
predates the second mode's addition (S506, 2026-09-21 — this monorepo turned
out to already have a live key of that kind sitting in its own established
interim-secrets convention, not the Vertex-mode credential this file
originally assumed was the only path) and is kept for import-path stability.
"""

from __future__ import annotations

import base64
import logging
import uuid
from collections.abc import AsyncIterable, Awaitable
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any

try:
    from google import genai
    from google.genai import types as genai_types

    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

logger = logging.getLogger(__name__)

# Anthropic "server tool" type prefixes that carry no ``input_schema`` — same
# set ``providers.translator._anthropic_tools_to_openai`` skips for the
# OpenRouter path. Gemini has no equivalent for any of these; silently
# dropping them (rather than passing a malformed FunctionDeclaration) matches
# that established precedent instead of inventing a new failure mode here.
_SERVER_TOOL_PREFIXES = ("web_search_", "computer_", "bash_", "code_execution_")

# Fallback generation params when a caller omits them — matches
# GeminiVertexProvider's own pre-fix defaults; named here instead of inline
# so they read as deliberate values, not unexplained literals.
_DEFAULT_MAX_OUTPUT_TOKENS = 2048
_DEFAULT_TEMPERATURE = 1.0


def _sanitize_schema_for_gemini(node: Any) -> Any:
    """Recursively rewrite a JSON-Schema fragment so it survives both
    ``genai_types.Schema`` VALIDATION and Gemini's own wire-format
    SERIALIZATION intact.

    Real bug #1, found live (2026-09-21) running an actual chat turn end to
    end through a real specialist tool schema, not a hand-picked test
    fixture: Pydantic v2's default JSON-Schema output for an
    ``Optional[int]`` field is ``{"type": ["integer", "null"]}`` (the
    standard JSON-Schema-2020-12 / OpenAPI-3.1 nullable-union form) —
    Gemini's ``Schema`` has no concept of a type LIST at all, only a single
    ``type`` plus a separate ``nullable: bool`` flag, and raises a real
    ``pydantic_core.ValidationError`` on the list form.

    Real bug #2, found live the same way, one layer deeper (only visible
    once bug #1 stopped masking it): a schema carrying JSON-Schema's
    ``additionalProperties`` keyword validates fine into ``Schema`` (a real
    field, ``additional_properties``, alias ``additionalProperties``) but
    then fails at the ACTUAL API call with a real ``400 INVALID_ARGUMENT``
    ("Unknown name \"additional_properties\"") — confirmed directly against
    the installed SDK: ``Schema(...).model_dump()`` emits the Python field
    name (``additional_properties``), not the JSON alias, and that's what
    reaches the wire. Not something this codebase can fix inside the SDK, so
    the field is dropped before construction instead — a validation-only
    JSON-Schema keyword Gemini's own function-calling has no real use for
    anyway (this is model-generated tool-call input, not adversarial data
    needing "reject unknown extra fields" strictness).

    Neither was caught by the existing test suite: it only ever exercised
    simple, single-type, no-``additionalProperties`` schemas.
    """
    _DROP_KEYS = ("additionalProperties", "additional_properties")
    if isinstance(node, dict):
        out = {
            k: _sanitize_schema_for_gemini(v)
            for k, v in node.items()
            if k not in _DROP_KEYS
        }
        t = out.get("type")
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            if len(non_null) == 1 and "null" in t:
                out["type"] = non_null[0]
                out["nullable"] = True
            elif non_null:
                # A genuine multi-type union (rare for a tool parameter) —
                # Gemini has no equivalent; take the first real type rather
                # than crash. Not exercised by any known real tool schema
                # today, so no regression test claims this is the "right"
                # choice, only that it degrades instead of raising.
                out["type"] = non_null[0]
                if "null" in t:
                    out["nullable"] = True
        return out
    if isinstance(node, list):
        return [_sanitize_schema_for_gemini(v) for v in node]
    return node


def _translate_tools(tools: list[Any] | None) -> list[genai_types.Tool] | None:
    """Anthropic ``tools[]`` (flat ``{name, description, input_schema}``, no
    ``type``/``function`` wrapper) -> a single Gemini ``Tool`` carrying one
    ``FunctionDeclaration`` per entry.

    Real Anthropic-shaped tool call kwargs never carry a ``type: "function"``
    wrapper (that is the OpenAI/OpenRouter shape) — confirmed directly
    against ``providers.translator._anthropic_tools_to_openai``, whose entire
    job is converting FROM this same flat shape TO that wrapped one for the
    OpenRouter path.
    """
    if not tools:
        return None
    declarations: list[genai_types.FunctionDeclaration] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if "input_schema" not in t and (t.get("type") or "").startswith(_SERVER_TOOL_PREFIXES):
            continue
        params = _sanitize_schema_for_gemini(
            t.get("input_schema") or {"type": "object", "properties": {}}
        )
        declarations.append(
            genai_types.FunctionDeclaration(
                name=t.get("name", ""),
                description=t.get("description", ""),
                # mypy wants a genai_types.Schema; a plain JSON-schema dict is
                # correct at runtime (pydantic coerces it — verified directly:
                # FunctionDeclaration(parameters=<dict>).parameters comes back
                # a real Schema instance), matching every specialist tool's
                # own real `input_schema` shape (orchestrator/router.py).
                parameters=params,  # type: ignore[arg-type]
            )
        )
    if not declarations:
        return None
    return [genai_types.Tool(function_declarations=declarations)]


def _content_blocks_to_parts(
    content: Any, tool_id_to_name: dict[str, str]
) -> list[genai_types.Part]:
    """One Anthropic message's ``content`` (str or list of typed blocks) ->
    Gemini ``Part`` list. Handles ``text``, ``tool_use`` (replayed as a real
    function-call part) and ``tool_result`` (a real function-response part).

    **Correlation is by ``id``, not by name** (round-2 adversarial review
    corrected an earlier, factually wrong version of this comment/fix: both
    ``genai_types.FunctionCall`` and ``FunctionResponse`` carry a real ``id``
    field — "populated by the client to match the corresponding function
    call id", per the SDK's own docstring, verified directly against the
    installed types). Name-only correlation breaks this codebase's own
    designed pattern of *parallel* fan-out — multiple ``tool_use`` blocks in
    one assistant turn calling the SAME tool name with different arguments
    (``orchestrator/router.route_parallel``) — since nothing would then
    distinguish which ``tool_result`` answers which call. Anthropic's own
    ``tool_use_id``/``tool_use.id`` is threaded straight through as Gemini's
    ``FunctionCall.id``/``FunctionResponse.id``, which is unique per call
    regardless of how many calls share a name. ``Part.from_function_call``/
    ``from_function_response`` have no ``id`` parameter, so the ``Part`` is
    constructed directly from a ``FunctionCall``/``FunctionResponse`` object
    instead. ``tool_id_to_name`` (threaded through the whole conversation by
    ``_build_contents``) still supplies the ``name`` field alongside ``id``
    — real, descriptive, but no longer the correlation mechanism itself.
    """
    if isinstance(content, str):
        return [genai_types.Part.from_text(text=content)] if content else []
    if not isinstance(content, list):
        return [genai_types.Part.from_text(text=str(content))]

    parts: list[genai_types.Part] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text = block.get("text", "")
            if text:
                parts.append(genai_types.Part.from_text(text=text))
        elif btype == "tool_use":
            name = block.get("name", "")
            tool_use_id = block.get("id", "")
            if tool_use_id:
                tool_id_to_name[tool_use_id] = name
            # thought_signature lives on the PART, not the FunctionCall (verified against the
            # installed SDK's own field list) -- see _response_content_blocks's own docstring for
            # the real bug this closes. A block from a non-Gemini turn (or an older persisted
            # turn from before this fix) simply has no signature to echo back; Gemini accepts a
            # function-call Part with no thought_signature on a NON-thinking-model turn (the
            # requirement is specifically "if the model emitted one, echo it back"), so omitting
            # the field here is a safe no-op, not a silent swallow of a real error.
            sig_b64 = block.get("gemini_thought_signature")
            part_kwargs: dict[str, Any] = {
                "function_call": genai_types.FunctionCall(
                    id=tool_use_id or None,
                    name=name,
                    args=block.get("input", {}) or {},
                )
            }
            if sig_b64:
                part_kwargs["thought_signature"] = base64.b64decode(sig_b64)
            parts.append(genai_types.Part(**part_kwargs))
        elif btype == "tool_result":
            inner = block.get("content")
            if isinstance(inner, str):
                result_text = inner
            elif isinstance(inner, list):
                result_text = "\n\n".join(
                    b.get("text", "") for b in inner
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            else:
                result_text = ""
            tool_use_id = block.get("tool_use_id", "")
            name = tool_id_to_name.get(tool_use_id, "")
            if not name:
                # No matching tool_use seen (shouldn't happen in a real
                # conversation) — log loudly rather than silently mis-routing
                # the result under a name Gemini will never recognize.
                logger.warning(
                    f"Gemini tool_result for unknown tool_use_id={tool_use_id!r}; "
                    "no matching function_call name found in this conversation"
                )
                name = "unknown_tool"
            # Gemini's function_response.response is a struct (dict), not a
            # bare string — real tool output is wrapped under one key rather
            # than dropped or mis-shaped.
            parts.append(
                genai_types.Part(
                    function_response=genai_types.FunctionResponse(
                        id=tool_use_id or None,
                        name=name,
                        response={"result": result_text},
                    )
                )
            )
    return parts


def _extract_system_text(system: Any) -> str:
    """Anthropic ``system`` kwarg -> plain text for Gemini's
    ``system_instruction``.

    Real call sites (``agents/base.py``, ``prompts/cache_manager.py``) ALWAYS
    pass a list of typed blocks (``[{"type": "text", "text": ..., "cache_control":
    {...}}, ...]``, e.g. persona+knowledge as one block and the company profile
    as a second) — never a bare string. Concatenating every text block (in
    order, joined by blank lines) is required for the Executive persona and
    company profile to reach Gemini at all; a naive `isinstance(system, str)`
    check would silently drop the entire system prompt on every real call,
    since it is never actually a plain string in production use. A bare
    string is still accepted for robustness (tests, future callers).
    """
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        chunks = [
            b.get("text", "")
            for b in system
            if isinstance(b, dict) and b.get("type") == "text" and b.get("text")
        ]
        return "\n\n".join(chunks)
    return ""


def _build_contents(messages: list[Any]) -> list[genai_types.Content]:
    """Anthropic ``messages[]`` -> Gemini ``Content[]``.

    The system prompt is intentionally NOT handled here — it goes on
    ``GenerateContentConfig.system_instruction`` (see ``messages_create``),
    Gemini's own native mechanism, instead of being prepended as a synthetic
    user turn. This repo's own README/CLAUDE.md calls prompt-caching
    architecture "critical" (breaking it is a ~10x cost increase) and expects
    the system block to be handled distinctly from ordinary turn content;
    folding it into `contents` here would defeat that on every Gemini call.
    """
    contents: list[genai_types.Content] = []
    tool_id_to_name: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        gemini_role = "model" if role == "assistant" else "user"
        parts = _content_blocks_to_parts(msg.get("content", ""), tool_id_to_name)
        if parts:
            contents.append(genai_types.Content(role=gemini_role, parts=parts))
    return contents


def _finish_reason_to_anthropic(finish_reason: Any, has_tool_use: bool) -> str:
    """Gemini ``FinishReason`` -> Anthropic ``stop_reason``.

    ``MAX_TOKENS`` wins over ``has_tool_use`` (round-2 review correction —
    the first version had this backwards): a function call can be cut off
    mid-generation by the token budget, and real Anthropic responses report
    ``max_tokens`` in exactly that case, not ``tool_use`` — several call
    sites (e.g. ``orchestrator/executive.py``'s own truncation handling)
    would otherwise treat a truncated, incomplete call's args as a complete
    one and execute it. Only when the turn ended for a real reason (not
    truncation) does ``has_tool_use`` win over a plain STOP — several call
    sites (e.g. ``orchestrator/executive.py``'s own
    ``stop_reason != "tool_use"`` check) branch on that exact string to
    decide whether to run tools.
    """
    name = getattr(finish_reason, "name", str(finish_reason))
    if name == "MAX_TOKENS":
        return "max_tokens"
    if has_tool_use:
        return "tool_use"
    if name in ("STOP", "FINISH_REASON_UNSPECIFIED"):
        return "end_turn"
    # SAFETY / RECITATION / PROHIBITED_CONTENT / etc. — the turn ended for a
    # content-policy reason, not a token or tool-use boundary. Anthropic has
    # no exact equivalent string; "end_turn" is the least-wrong mapping (the
    # turn is genuinely over, not truncated) rather than fabricating a
    # `stop_reason` value no caller has ever seen from a real Claude response.
    return "end_turn"


def _response_content_blocks(candidate: Any) -> tuple[list[SimpleNamespace], bool]:
    """One Gemini ``Candidate`` -> Anthropic-shape content blocks.

    Returns ``(blocks, has_tool_use)``. Each function-call part becomes a
    real, typed ``tool_use`` block (not JSON serialized into a text block —
    the pre-fix code's own ``_translate_response`` did exactly that, which
    would have made every specialist consultation invisible to
    ``consult_specialist``'s own tool-use routing).

    ``thought_signature`` (real bug found live, 2026-09-21 — every multi-step
    tool-use turn was failing on its second Gemini call with `400
    INVALID_ARGUMENT: Function call is missing a thought_signature`):
    Gemini's "thinking" models attach an opaque ``bytes`` signature to the
    ``Part`` that carries a ``function_call`` (NOT to the ``FunctionCall``
    itself — verified directly against the installed SDK's own
    ``Part``/``FunctionCall`` field lists), and require that exact signature
    to be echoed back verbatim when that function call is replayed as
    conversation history on a later iteration WITHIN the same turn's
    tool-use loop (``executive.py``'s in-memory ``messages`` list, and the
    equivalent loops in ``workflows/executive_research.py``/
    ``executive_reflection.py`` -- not cross-turn persistence: only the
    final synthesized text is ever written to the session store). This
    codebase's internal Anthropic-shaped ``tool_use`` block had nowhere to
    carry it, so it was silently dropped — every second-and-later iteration
    of any tool-using turn then failed outright. Base64-encoded here
    (``bytes`` isn't JSON-serializable, and this attribute rides along on a
    plain dict through debug/event collectors that do serialize to JSON)
    onto a ``gemini_thought_signature`` attribute; ``_content_blocks_to_parts``
    below reads it back and re-attaches it to the replayed ``Part``. Other
    providers' blocks never set this attribute, so ``getattr(..., None)``
    elsewhere stays a safe, generic no-op for them.
    """
    blocks: list[SimpleNamespace] = []
    has_tool_use = False
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    for part in parts:
        # Real gap found in review (2026-09-21): an early `continue` here after a text branch
        # would silently drop a function_call (and its thought_signature) on any Part that
        # somehow carries both -- in practice Gemini 2.5 never sends both on one Part, but nothing
        # in the SDK's own types guarantees that, and Gemini 3's docs describe signatures riding
        # on more part kinds. No `continue`: both branches are checked independently instead.
        text = getattr(part, "text", None)
        if text:
            blocks.append(SimpleNamespace(type="text", text=text))
        fc = getattr(part, "function_call", None)
        if fc is not None:
            has_tool_use = True
            sig = getattr(part, "thought_signature", None)
            blocks.append(
                SimpleNamespace(
                    type="tool_use",
                    # A per-response part index (the old `i`-based id) collides
                    # across different turns of the same conversation — turn 2's
                    # first function call would mint the identical id as turn
                    # 1's first function call, and the orchestrator's own
                    # results-by-id lookup (executive.py) can then apply turn
                    # 2's tool_result to turn 1's call. uuid4 is unique per
                    # call, not just per response.
                    id=f"toolu_gemini_{uuid.uuid4().hex[:12]}",
                    name=getattr(fc, "name", ""),
                    input=dict(getattr(fc, "args", None) or {}),
                    gemini_thought_signature=(
                        base64.b64encode(sig).decode("ascii") if sig else None
                    ),
                )
            )
    return blocks, has_tool_use


def _usage_namespace(usage_meta: Any) -> SimpleNamespace:
    """Gemini ``usage_metadata`` -> the Anthropic-shape ``usage`` namespace.

    ``output_tokens`` includes ``thoughts_token_count`` alongside
    ``candidates_token_count`` (round-2 review found these are separate SDK
    fields, and Gemini 2.5 Pro — this provider's own configured default —
    cannot have thinking disabled; its thinking tokens are billed at the
    output rate). Omitting them under-reported real run cost by a large,
    silent factor whenever the model reasoned at all before answering.
    """
    input_tokens = getattr(usage_meta, "prompt_token_count", None) or 0
    candidates_tokens = getattr(usage_meta, "candidates_token_count", None) or 0
    thoughts_tokens = getattr(usage_meta, "thoughts_token_count", None) or 0
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=candidates_tokens + thoughts_tokens,
        # Gemini's context-cache accounting is a distinct, opt-in feature
        # (explicit cached_content, not Anthropic's automatic cache_control
        # breakpoints) — reporting its "already counted in
        # prompt_token_count" cached tokens as Anthropic's
        # cache_read_input_tokens would double-count savings this call never
        # actually asked Gemini for. Left at 0 until this provider explicitly
        # wires Vertex context caching (not attempted here).
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        cost=None,
        server_tool_use=SimpleNamespace(web_search_requests=0),
    )


def _response_to_namespace(response: Any, model_name: str) -> SimpleNamespace:
    """A real Gemini ``GenerateContentResponse`` -> the Anthropic-shape
    ``SimpleNamespace`` every caller in this codebase expects (attribute
    access — ``message.content``, ``block.type``/``block.text``,
    ``message.usage.input_tokens`` — never ``dict`` access). The pre-fix code
    returned a plain ``dict`` here, which would have raised ``AttributeError``
    on the very first real call (confirmed against
    ``agents/base.py``'s own ``message.content`` / ``b.type`` / ``b.text``
    usage and ``providers.provider.LLMProvider``'s own doc comment: callers
    get back an "``anthropic.types.Message``-shaped object").
    """
    usage_meta = getattr(response, "usage_metadata", None)
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        # A prompt blocked by safety (or otherwise producing zero
        # candidates) still bills real input tokens — usage_meta is read
        # here too (round-2 review: the first version hardcoded 0s even
        # when the response carried real usage_metadata), not just on the
        # real-content path below.
        return SimpleNamespace(
            id=getattr(response, "response_id", "") or "",
            type="message",
            role="assistant",
            model=model_name,
            content=[SimpleNamespace(type="text", text="")],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=_usage_namespace(usage_meta),
        )

    candidate = candidates[0]
    blocks, has_tool_use = _response_content_blocks(candidate)
    if not blocks:
        blocks = [SimpleNamespace(type="text", text="")]

    return SimpleNamespace(
        id=getattr(response, "response_id", "") or "",
        type="message",
        role="assistant",
        model=getattr(response, "model_version", None) or model_name,
        content=blocks,
        stop_reason=_finish_reason_to_anthropic(
            getattr(candidate, "finish_reason", None), has_tool_use
        ),
        stop_sequence=None,
        usage=_usage_namespace(usage_meta),
    )


class GeminiVertexProvider:
    """Adapter to use Gemini as an ``LLMProvider``, via either of the two real
    backends ``google.genai.Client`` supports:

    - **Vertex AI** (``project_id`` set, no ``api_key``): Application Default
      Credentials (ADC) from the environment (service account, ``gcloud auth
      application-default login``, or an attached GCP compute service
      account) — no explicit key configuration needed, but a real GCP project
      with Vertex AI enabled and billing on is a real prerequisite.
    - **Gemini Developer API** (``api_key`` set): a plain API key against
      ``generativelanguage.googleapis.com`` — no GCP project/ADC/billing
      console work needed beyond whatever created the key itself. This is
      the simpler of the two and was added when this monorepo turned out to
      already have a live key of this kind (see ``config.py``'s own
      ``gemini_api_key`` field), not a Vertex-mode one — the class name
      predates that discovery and is kept for import-path stability.

    ``api_key`` takes priority when both are set — matches the real
    ``google.genai.Client`` constructor's own mutually-exclusive shape
    (passing both ``vertexai=True`` and ``api_key`` is not itself invalid,
    but the API-key auth path is what actually gets used).
    """

    def __init__(
        self,
        *,
        project_id: str | None = None,
        location: str = "us-central1",
        api_key: str | None = None,
    ) -> None:
        if not GENAI_AVAILABLE:
            raise RuntimeError(
                "google-genai (with google-cloud-aiplatform) is required for "
                "Gemini support. Install with: "
                "pip install google-genai google-cloud-aiplatform"
            )
        if not api_key and not project_id:
            raise ValueError(
                "GeminiVertexProvider requires either api_key (Gemini Developer "
                "API) or project_id (Vertex AI via ADC)"
            )
        self.project_id = project_id
        self.location = location
        self.uses_api_key = api_key is not None
        if api_key:
            self._client = genai.Client(api_key=api_key)
            logger.info("GeminiVertexProvider initialized: Gemini Developer API (api_key)")
        else:
            self._client = genai.Client(vertexai=True, project=project_id, location=location)
            logger.info(
                f"GeminiVertexProvider initialized: Vertex AI, project={project_id}, location={location}"
            )

    def _config(self, **kwargs: Any) -> genai_types.GenerateContentConfig:
        system_text = _extract_system_text(kwargs.get("system"))
        max_tokens = kwargs.get("max_tokens", _DEFAULT_MAX_OUTPUT_TOKENS)
        temperature = kwargs.get("temperature", _DEFAULT_TEMPERATURE)
        tools = _translate_tools(kwargs.get("tools"))
        return genai_types.GenerateContentConfig(
            system_instruction=system_text or None,
            max_output_tokens=max_tokens,
            temperature=temperature,
            # mypy's declared Union is broader (Tool | Callable | mcp.types.Tool
            # | ClientSession) than our own list[Tool]; list[] is invariant so
            # mypy won't accept the narrower list as a match even though it's
            # a valid member of the union — verified directly at runtime.
            tools=tools,  # type: ignore[arg-type]
        )

    def messages_create(self, **kwargs: Any) -> Awaitable[Any]:
        """Non-streaming completion. Accepts Anthropic-shaped kwargs, returns
        an Anthropic-``Message``-shaped ``SimpleNamespace``."""

        async def _create() -> SimpleNamespace:
            model_name = kwargs.get("model", "gemini-3.1-pro-preview")
            contents = _build_contents(kwargs.get("messages", []))
            config = self._config(**kwargs)
            response = await self._client.aio.models.generate_content(
                model=model_name, contents=contents, config=config
            )
            return _response_to_namespace(response, model_name)

        return _create()

    @asynccontextmanager
    async def messages_stream(self, **kwargs: Any):
        """Streaming completion.

        Real token-level Gemini streaming (``generate_content_stream``) is
        deliberately NOT wired here yet — verified live that the SDK
        supports it (an ``AsyncIterator[GenerateContentResponse]``, same
        response shape as the non-streaming path). A live Gemini Developer
        API key now exists (see ``GeminiVertexProvider``'s own docstring),
        so this is no longer a credentials gap — it's a correctness one: a
        subtly wrong incremental tool-call-argument accumulation would be
        worse than an honest, correctly-translated non-streaming call
        chunked for delivery, and that accumulation logic hasn't been
        written or verified yet. Real streaming is real, separate, deferred
        work — this wrapper is built on the now-CORRECT translation logic
        above (unlike the pre-fix stub, which reused the same broken tool
        schema and dict-shaped response), not a cosmetic no-op.
        """
        response_ns = await self.messages_create(**kwargs)

        yield_events: list[dict[str, Any]] = [
            {"type": "message_start", "message": response_ns}
        ]
        for block in response_ns.content:
            yield_events.append({"type": "content_block_start", "content_block": block})
            if getattr(block, "type", None) == "text":
                yield_events.append(
                    {
                        "type": "content_block_delta",
                        "delta": SimpleNamespace(type="text_delta", text=block.text),
                    }
                )
            yield_events.append({"type": "content_block_stop"})
        yield_events.append(
            {
                "type": "message_delta",
                "delta": SimpleNamespace(stop_reason=response_ns.stop_reason),
            }
        )
        yield_events.append({"type": "message_stop"})

        stream = _GeminiFakeStream(yield_events, response_ns)
        yield stream


class _GeminiFakeStream:
    """The async-iterable + ``get_final_message()`` object
    ``providers.provider.LLMProvider.messages_stream``'s own contract
    requires, built from a list of already-translated Anthropic-shape
    events. Each event is a ``SimpleNamespace`` with a real ``.type``
    attribute (never a plain ``dict``) so ``hasattr(event, "type")``/
    ``event.type`` checks at call sites (e.g. ``orchestrator/executive.py``)
    see the same real shape a genuine Anthropic stream event would have.
    """

    def __init__(self, events: list[dict[str, Any]], final_message: SimpleNamespace) -> None:
        self._events = [self._to_namespace(e) for e in events]
        self._final_message = final_message

    @staticmethod
    def _to_namespace(event: dict[str, Any]) -> SimpleNamespace:
        return SimpleNamespace(**event)

    def __aiter__(self) -> AsyncIterable[SimpleNamespace]:
        return self._iter()

    async def _iter(self):
        for event in self._events:
            yield event

    async def get_final_message(self) -> SimpleNamespace:
        return self._final_message
