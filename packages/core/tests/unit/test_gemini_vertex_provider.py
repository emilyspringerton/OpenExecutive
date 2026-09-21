"""Real, direct coverage for the Gemini/Vertex provider's Anthropic<->Gemini
translation logic (S506), against the REAL installed `google.genai` SDK types
(no mocking of the SDK's own data classes) — this is exactly the level a
mock would have hidden the pre-fix bugs at (wrong tool schema, dict instead
of an attribute-accessible response, tool_result keyed by the wrong field,
duplicate synthetic tool_use ids, the system prompt being silently dropped).
"""
from __future__ import annotations

import asyncio

from google.genai import types as genai_types

from openexecutive.providers.gemini_vertex_provider import (
    GeminiVertexProvider,
    _build_contents,
    _extract_system_text,
    _GeminiFakeStream,
    _response_to_namespace,
    _translate_tools,
)


def test_translate_tools_uses_real_anthropic_input_schema_shape() -> None:
    tools = [
        {
            "name": "consult_specialist",
            "description": "desc",
            "input_schema": {
                "type": "object",
                "properties": {"domain": {"type": "string"}},
            },
        }
    ]
    result = _translate_tools(tools)
    assert result is not None
    assert len(result[0].function_declarations) == 1
    assert result[0].function_declarations[0].name == "consult_specialist"


def test_translate_tools_skips_server_tools_with_no_input_schema() -> None:
    assert _translate_tools([{"type": "web_search_20250305", "name": "web_search"}]) is None


def test_translate_tools_handles_type_none_without_crashing() -> None:
    result = _translate_tools([{"name": "ws", "type": None}])
    assert result is not None
    assert result[0].function_declarations[0].name == "ws"


def test_translate_tools_empty_or_none_returns_none() -> None:
    assert _translate_tools(None) is None
    assert _translate_tools([]) is None


def test_extract_system_text_joins_real_multi_block_shape() -> None:
    real_system = [
        {"type": "text", "text": "You are the Executive.", "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": "Company: Acme Corp.", "cache_control": {"type": "ephemeral"}},
    ]
    assert _extract_system_text(real_system) == "You are the Executive.\n\nCompany: Acme Corp."


def test_extract_system_text_accepts_plain_string_too() -> None:
    assert _extract_system_text("plain") == "plain"


def test_extract_system_text_empty_or_none_is_empty_string() -> None:
    assert _extract_system_text(None) == ""
    assert _extract_system_text([]) == ""


def test_build_contents_tool_result_correlates_by_real_id_and_carries_name() -> None:
    """Correlation is by id (both FunctionCall and FunctionResponse carry a
    real `id` field for this, per the installed SDK's own docstring); `name`
    is still populated for both, but is descriptive, not the correlation
    mechanism (an earlier version of this fix got that backwards)."""
    messages = [
        {"role": "user", "content": "question"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "toolu_01ABC", "name": "consult_specialist", "input": {"domain": "cfo"}}
            ],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_01ABC", "content": "answer"}],
        },
    ]
    contents = _build_contents(messages)
    assert contents[1].parts[0].function_call.id == "toolu_01ABC"
    assert contents[1].parts[0].function_call.name == "consult_specialist"
    assert contents[2].parts[0].function_response.id == "toolu_01ABC"
    assert contents[2].parts[0].function_response.name == "consult_specialist"


def test_build_contents_parallel_same_name_calls_correlate_correctly_by_id() -> None:
    """The real bug a name-only correlation had: this codebase's own
    designed pattern is PARALLEL fan-out of the same tool name
    (orchestrator.router.route_parallel) -- two tool_use blocks in one
    assistant turn calling `consult_specialist` with different arguments.
    Real, unique Anthropic tool_use ids (not names) must disambiguate which
    result answers which call, even when results arrive out of call order."""
    messages = [
        {"role": "user", "content": "compare cfo and gc views"},
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": "call_cfo_001", "name": "consult_specialist", "input": {"specialist": "cfo"}},
                {"type": "tool_use", "id": "call_gc_002", "name": "consult_specialist", "input": {"specialist": "gc"}},
            ],
        },
        {
            "role": "user",
            "content": [
                # Deliberately out of call order.
                {"type": "tool_result", "tool_use_id": "call_gc_002", "content": "LEGAL: blocked"},
                {"type": "tool_result", "tool_use_id": "call_cfo_001", "content": "FINANCE: affordable"},
            ],
        },
    ]
    contents = _build_contents(messages)
    fc1, fc2 = contents[1].parts[0].function_call, contents[1].parts[1].function_call
    fr1, fr2 = contents[2].parts[0].function_response, contents[2].parts[1].function_response

    assert fc1.id == "call_cfo_001"
    assert fc2.id == "call_gc_002"
    assert fr1.id == "call_gc_002" and fr1.response["result"] == "LEGAL: blocked"
    assert fr2.id == "call_cfo_001" and fr2.response["result"] == "FINANCE: affordable"


def test_build_contents_unknown_tool_result_id_falls_back_without_crashing() -> None:
    messages = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "never-seen", "content": "x"}]},
    ]
    contents = _build_contents(messages)
    assert contents[0].parts[0].function_response.name == "unknown_tool"


def test_response_to_namespace_text_block() -> None:
    resp = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=[genai_types.Part.from_text(text="hi")]),
                finish_reason=genai_types.FinishReason.STOP,
            )
        ],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=5
        ),
    )
    ns = _response_to_namespace(resp, "gemini-2.5-pro")
    assert ns.content[0].type == "text"
    assert ns.content[0].text == "hi"
    assert ns.stop_reason == "end_turn"
    assert ns.usage.input_tokens == 10
    assert ns.usage.output_tokens == 5


def test_response_to_namespace_tool_use_block_sets_stop_reason() -> None:
    resp = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_function_call(name="consult_specialist", args={"domain": "cfo"})],
                ),
                finish_reason=genai_types.FinishReason.STOP,
            )
        ],
    )
    ns = _response_to_namespace(resp, "gemini-2.5-pro")
    assert ns.content[0].type == "tool_use"
    assert ns.content[0].name == "consult_specialist"
    assert ns.content[0].input == {"domain": "cfo"}
    assert ns.stop_reason == "tool_use"


def test_response_to_namespace_max_tokens_wins_over_tool_use() -> None:
    """Round-2 regression: a function call truncated mid-generation by the
    token budget must report max_tokens, not tool_use -- otherwise a caller
    (e.g. orchestrator/executive.py) treats the truncated, incomplete call's
    args as a complete one and executes it."""
    resp = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model",
                    parts=[genai_types.Part.from_function_call(name="consult_specialist", args={"domain": "cfo"})],
                ),
                finish_reason=genai_types.FinishReason.MAX_TOKENS,
            )
        ],
    )
    ns = _response_to_namespace(resp, "gemini-2.5-pro")
    assert ns.stop_reason == "max_tokens"


def test_response_to_namespace_includes_thinking_tokens_in_output_tokens() -> None:
    """Round-2 regression: Gemini 2.5 Pro's thinking tokens (a separate SDK
    field, thoughts_token_count) are billed at the output rate and cannot be
    disabled -- omitting them silently under-reports real run cost."""
    resp = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(role="model", parts=[genai_types.Part.from_text(text="hi")]),
                finish_reason=genai_types.FinishReason.STOP,
            )
        ],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=20, thoughts_token_count=900
        ),
    )
    ns = _response_to_namespace(resp, "gemini-2.5-pro")
    assert ns.usage.output_tokens == 920


def test_response_to_namespace_no_candidates_still_reports_real_input_usage() -> None:
    """Round-2 regression: a prompt blocked by safety returns candidates=[]
    but still carries real usage_metadata.prompt_token_count -- the
    no-candidates branch must not hardcode 0 and silently record a free call
    for input tokens that were actually billed."""
    resp = genai_types.GenerateContentResponse(
        candidates=[],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(prompt_token_count=500),
    )
    ns = _response_to_namespace(resp, "gemini-2.5-pro")
    assert ns.usage.input_tokens == 500


def test_response_to_namespace_two_tool_calls_get_distinct_ids() -> None:
    """Regression guard: a per-part-index id would collide across separate
    turns of the same conversation (found in adversarial review)."""
    resp = genai_types.GenerateContentResponse(
        candidates=[
            genai_types.Candidate(
                content=genai_types.Content(
                    role="model",
                    parts=[
                        genai_types.Part.from_function_call(name="a", args={}),
                        genai_types.Part.from_function_call(name="b", args={}),
                    ],
                ),
                finish_reason=genai_types.FinishReason.STOP,
            )
        ],
    )
    ns = _response_to_namespace(resp, "gemini-2.5-pro")
    ids = [b.id for b in ns.content]
    assert len(set(ids)) == 2


def test_response_to_namespace_no_candidates_returns_safe_default() -> None:
    resp = genai_types.GenerateContentResponse(candidates=[])
    ns = _response_to_namespace(resp, "gemini-2.5-pro")
    assert ns.content[0].type == "text"
    assert ns.content[0].text == ""
    assert ns.usage.input_tokens == 0


def test_provider_construction_needs_no_live_credentials() -> None:
    # Construction resolves ADC lazily (only on an actual API call), unlike
    # the deprecated vertexai.generative_models.GenerativeModel, which
    # eagerly resolved the project at construction time.
    provider = GeminiVertexProvider(project_id="fake-project", location="us-central1")
    assert provider.project_id == "fake-project"


def test_fake_stream_iterates_and_returns_final_message() -> None:
    from types import SimpleNamespace

    final = SimpleNamespace(content=[SimpleNamespace(type="text", text="hi")], stop_reason="end_turn")
    events = [
        {"type": "message_start", "message": final},
        {"type": "content_block_delta", "delta": SimpleNamespace(type="text_delta", text="hi")},
        {"type": "message_stop"},
    ]
    stream = _GeminiFakeStream(events, final)

    async def _run() -> list[str]:
        seen = []
        async for e in stream:
            seen.append(e.type)
        return seen

    seen = asyncio.run(_run())
    assert seen == ["message_start", "content_block_delta", "message_stop"]
    assert asyncio.run(stream.get_final_message()) is final
