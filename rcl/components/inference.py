"""Unified multi-provider inference with tool-calling loops.

Provides async manual tool-calling loops for Gemini, OpenAI, and Anthropic,
all returning the same AFCResult dataclass. Benchmark adapters call
run_inference() and never think about providers.

For simple text generation (reflector/mutator/judge), use
rcl.components.llm_client.create_generate_fn() instead.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result type
# ---------------------------------------------------------------------------

@dataclass
class AFCResult:
    """Structured result from a multi-turn tool-calling inference run."""
    final_text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    n_tool_calls: int = 0
    usage: Dict[str, int] = field(default_factory=dict)
    duration_sec: float = 0.0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _parse_provider(model: str) -> tuple:
    """Return (provider, base_model_name).

    Model format must be 'provider/model-name' (e.g. 'openai/gpt-5.4-nano',
    'anthropic/claude-opus-4-6', 'google/gemini-3-flash-preview').
    """
    if "/" not in model:
        raise ValueError(
            f"Model must be in 'provider/model-name' format, got '{model}'. "
            f"Examples: 'openai/gpt-5.4-nano', 'anthropic/claude-opus-4-6', "
            f"'google/gemini-3-flash-preview'."
        )
    provider, name = model.split("/", 1)
    provider = provider.lower()
    if provider not in ("openai", "anthropic", "google"):
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Supported: openai, anthropic, google."
        )
    return provider, name


# ---------------------------------------------------------------------------
# Retryable error detection
# ---------------------------------------------------------------------------

def _is_retryable(err_str: str) -> bool:
    return any(s in err_str for s in [
        "429", "RESOURCE_EXHAUSTED", "499", "CANCELLED", "503",
        "ReadTimeout", "ConnectTimeout", "TimeoutError",
        "RateLimitError", "APITimeoutError", "overloaded",
        "rate limit", "too many requests", "temporarily unavailable",
    ])


# ═══════════════════════════════════════════════════════════════════════════
# GEMINI
# ═══════════════════════════════════════════════════════════════════════════

def create_gemini_client():
    """Create a Gemini client, auto-detecting Vertex AI vs API key mode."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project and not api_key:
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
        return genai.Client(vertexai=True, project=project, location=location)
    return genai.Client(api_key=api_key)


def _build_gemini_tool_declarations(mcp_tools: list):
    """Convert MCP tool list to Gemini FunctionDeclarations."""
    from google.genai import types
    declarations = []
    for tool in mcp_tools:
        params = {}
        if tool.inputSchema:
            params = {
                "type": tool.inputSchema.get("type", "OBJECT").upper(),
                "properties": tool.inputSchema.get("properties", {}),
                "required": tool.inputSchema.get("required", []),
            }
        declarations.append(
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description or "",
                parameters=params,
            )
        )
    return types.Tool(function_declarations=declarations)


async def _run_gemini_manual(
    client,
    model: str,
    prompt: str,
    mcp_client,
    system_prompt: Optional[str] = None,
    max_steps: int = 50,
    max_output_tokens: int = 65536,
    step_timeout: float = 120,
    max_retries: int = 5,
    thinking_level: Optional[str] = "HIGH",
    cancelled: Optional[asyncio.Event] = None,
) -> AFCResult:
    """Gemini manual tool-calling loop with MCP."""
    from google.genai import types

    mcp_tools = await mcp_client.list_tools()
    tool_decl = _build_gemini_tool_declarations(mcp_tools)

    config = types.GenerateContentConfig(
        max_output_tokens=max_output_tokens,
        tools=[tool_decl],
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="AUTO")
        ),
    )
    if thinking_level:
        config.thinking_config = types.ThinkingConfig(
            include_thoughts=True,
            thinking_level=thinking_level,
        )
    if system_prompt:
        config.system_instruction = system_prompt

    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
    tool_calls = []
    usage = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0}
    t0 = time.time()

    for step in range(max_steps):
        if cancelled and cancelled.is_set():
            break

        response = None
        step_error = None
        for attempt in range(max(1, max_retries)):
            try:
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model, contents=contents, config=config,
                    ),
                    timeout=step_timeout,
                )
                break
            except asyncio.TimeoutError:
                step_error = f"Step {step} timed out after {step_timeout}s"
                logger.warning("Gemini loop: %s (attempt %d/%d)", step_error, attempt + 1, max_retries)
            except Exception as e:
                err_str = str(e)
                if _is_retryable(err_str) and attempt < max_retries - 1:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning("Gemini step %d retry %d/%d, waiting %ds: %s",
                                   step, attempt + 1, max_retries, wait, err_str[:120])
                    await asyncio.sleep(wait)
                    step_error = err_str
                else:
                    return AFCResult(
                        tool_calls=tool_calls,
                        n_tool_calls=sum(1 for tc in tool_calls if tc.get("type") == "tool_call"),
                        usage=usage, duration_sec=round(time.time() - t0, 1),
                        error=f"Step {step}: {err_str}",
                    )

        if response is None:
            return AFCResult(
                tool_calls=tool_calls,
                n_tool_calls=sum(1 for tc in tool_calls if tc.get("type") == "tool_call"),
                usage=usage, duration_sec=round(time.time() - t0, 1),
                error=step_error or f"Step {step} failed after {max_retries} attempts",
            )

        um = getattr(response, "usage_metadata", None)
        if um:
            usage["prompt_tokens"] += getattr(um, "prompt_token_count", 0) or 0
            usage["candidates_tokens"] += getattr(um, "candidates_token_count", 0) or 0
            usage["thoughts_tokens"] += getattr(um, "thoughts_token_count", 0) or 0
            usage["total_tokens"] += getattr(um, "total_token_count", 0) or 0

        if not response.candidates or not response.candidates[0].content.parts:
            break

        parts = response.candidates[0].content.parts
        function_calls = []
        for part in parts:
            if getattr(part, "thought", False) and hasattr(part, "text") and part.text:
                tool_calls.append({"type": "reasoning", "output": part.text})
            elif hasattr(part, "function_call") and part.function_call:
                function_calls.append(part.function_call)
            elif hasattr(part, "text") and part.text:
                tool_calls.append({"type": "output_text", "output": part.text})

        if not function_calls:
            break

        response_parts = []
        for fc in function_calls:
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}
            try:
                mcp_result = await mcp_client.call_tool(tool_name, tool_args)
                tool_output = mcp_result[0].text if mcp_result else ""
            except Exception as e:
                tool_output = f"Error: {e}"
                logger.warning("MCP tool %s failed: %s", tool_name, e)

            tool_calls.append({
                "type": "tool_call", "tool_name": tool_name,
                "arguments": tool_args, "output": tool_output,
            })
            response_parts.append(types.Part.from_function_response(
                name=tool_name, response={"result": tool_output},
            ))

        contents.append(response.candidates[0].content)
        contents.append(types.Content(role="user", parts=response_parts))

    final_text = ""
    for tc in reversed(tool_calls):
        if tc.get("type") == "output_text":
            final_text = tc["output"]
            break

    return AFCResult(
        final_text=final_text, tool_calls=tool_calls,
        n_tool_calls=sum(1 for tc in tool_calls if tc.get("type") == "tool_call"),
        usage=usage, duration_sec=round(time.time() - t0, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# OPENAI
# ═══════════════════════════════════════════════════════════════════════════

_openai_async_client = None


def _get_openai_async_client():
    global _openai_async_client
    if _openai_async_client is None:
        import openai
        _openai_async_client = openai.AsyncOpenAI()
    return _openai_async_client


def _is_reasoning_model(model: str) -> bool:
    base = model.split("/", 1)[-1] if "/" in model else model
    return any(base.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))


def _uses_responses_api(model: str, thinking: Optional[str], has_tools: bool) -> bool:
    """gpt-5+ with reasoning + tools requires Responses API."""
    if not has_tools or not thinking or thinking.lower() == "none":
        return False
    return _is_reasoning_model(model)


def _mcp_tools_to_openai(mcp_tools: list) -> list:
    """Convert MCP tools to OpenAI Chat Completions format."""
    return [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description or "",
            "parameters": t.inputSchema or {"type": "object", "properties": {}},
        },
    } for t in mcp_tools]


def _tools_to_responses_format(tools: list) -> list:
    """Convert Chat Completions tool format to Responses API format."""
    return [{
        "type": "function",
        "name": t["function"]["name"],
        "description": t["function"].get("description", ""),
        "parameters": t["function"].get("parameters", {"type": "object", "properties": {}}),
    } for t in tools]


def _messages_to_responses_input(messages: list) -> tuple:
    """Convert OpenAI messages to Responses API (instructions, input_items)."""
    instructions = None
    items = []
    for msg in messages:
        role = msg["role"]
        if role == "system":
            instructions = msg["content"]
        elif role == "user":
            items.append({"role": "user", "content": msg["content"]})
        elif role == "assistant":
            if msg.get("content"):
                items.append({"role": "assistant", "content": msg["content"]})
            for tc in msg.get("tool_calls", []):
                fn = tc["function"]
                items.append({
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": fn["name"],
                    "arguments": fn["arguments"] if isinstance(fn["arguments"], str) else json.dumps(fn["arguments"]),
                })
        elif role == "tool":
            items.append({
                "type": "function_call_output",
                "call_id": msg["tool_call_id"],
                "output": msg["content"],
            })
    return instructions, items


def _parse_chat_response(resp):
    """Parse Chat Completions response."""
    msg = resp.choices[0].message
    tool_calls = None
    if msg.tool_calls:
        tool_calls = []
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({"id": tc.id, "name": tc.function.name, "arguments": args})

    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.prompt_tokens or 0,
            "completion_tokens": resp.usage.completion_tokens or 0,
            "total_tokens": resp.usage.total_tokens or 0,
        }
        detail = getattr(resp.usage, "completion_tokens_details", None)
        if detail:
            usage["reasoning_tokens"] = getattr(detail, "reasoning_tokens", 0) or 0
    return msg.content, tool_calls, usage


def _parse_responses_api(resp):
    """Parse Responses API response."""
    content = None
    tool_calls = []
    for item in resp.output:
        if item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    content = (content or "") + part.text
        elif item.type == "function_call":
            try:
                args = json.loads(item.arguments)
            except (json.JSONDecodeError, TypeError):
                args = {}
            tool_calls.append({"id": item.call_id, "name": item.name, "arguments": args})
        elif item.type == "reasoning":
            pass  # reasoning summaries not captured in trace for now

    usage = {}
    if resp.usage:
        usage = {
            "prompt_tokens": resp.usage.input_tokens or 0,
            "completion_tokens": resp.usage.output_tokens or 0,
            "total_tokens": resp.usage.total_tokens or 0,
        }
    return content, tool_calls or None, usage


def _build_assistant_msg(content, tool_calls):
    """Build OpenAI-format assistant message for conversation history."""
    msg = {"role": "assistant", "content": content or ""}
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": tc["id"], "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else tc["arguments"],
                },
            }
            for tc in tool_calls
        ]
    return msg


async def _run_openai_manual(
    model: str,
    prompt: str,
    mcp_client,
    system_prompt: Optional[str] = None,
    max_steps: int = 50,
    max_output_tokens: int = 65536,
    step_timeout: float = 120,
    max_retries: int = 5,
    thinking_level: Optional[str] = "HIGH",
    cancelled: Optional[asyncio.Event] = None,
) -> AFCResult:
    """OpenAI manual tool-calling loop with MCP."""
    client = _get_openai_async_client()
    base_model = model.split("/", 1)[-1] if "/" in model else model
    mcp_tools = await mcp_client.list_tools()
    tools = _mcp_tools_to_openai(mcp_tools)
    use_responses = _uses_responses_api(model, thinking_level, bool(tools))

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    trace = []
    usage_totals = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0}
    t0 = time.time()

    for step in range(max_steps):
        if cancelled and cancelled.is_set():
            break

        content = None
        step_tool_calls = None
        step_usage = {}
        step_error = None

        for attempt in range(max(1, max_retries)):
            try:
                step_t0 = time.time()
                if use_responses:
                    instructions, input_items = _messages_to_responses_input(messages)
                    kwargs = {
                        "model": base_model,
                        "input": input_items,
                        "tools": _tools_to_responses_format(tools),
                    }
                    if instructions:
                        kwargs["instructions"] = instructions
                    if thinking_level and thinking_level.lower() != "none":
                        kwargs["reasoning"] = {"effort": thinking_level.lower()}
                    resp = await asyncio.wait_for(
                        client.responses.create(**kwargs),
                        timeout=step_timeout,
                    )
                    content, step_tool_calls, step_usage = _parse_responses_api(resp)
                else:
                    kwargs = {"model": base_model, "messages": messages}
                    if _is_reasoning_model(model):
                        kwargs["max_completion_tokens"] = max_output_tokens
                    else:
                        kwargs["max_tokens"] = max_output_tokens
                    if thinking_level and thinking_level.lower() != "none" and not tools:
                        kwargs["reasoning_effort"] = thinking_level.lower()
                    if tools:
                        kwargs["tools"] = tools
                        kwargs["tool_choice"] = "auto"
                    resp = await asyncio.wait_for(
                        client.chat.completions.create(**kwargs),
                        timeout=step_timeout,
                    )
                    content, step_tool_calls, step_usage = _parse_chat_response(resp)
                logger.info("Step %d LLM call took %.1fs", step, time.time() - step_t0)
                step_error = None
                break
            except asyncio.TimeoutError:
                step_error = f"Step {step} timed out after {step_timeout}s"
                logger.warning("OpenAI loop: %s (attempt %d/%d)", step_error, attempt + 1, max_retries)
            except Exception as e:
                err_str = str(e)
                if _is_retryable(err_str) and attempt < max_retries - 1:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning("OpenAI step %d retry %d/%d, waiting %ds: %s",
                                   step, attempt + 1, max_retries, wait, err_str[:120])
                    await asyncio.sleep(wait)
                    step_error = err_str
                else:
                    return AFCResult(
                        tool_calls=trace,
                        n_tool_calls=sum(1 for tc in trace if tc.get("type") == "tool_call"),
                        usage=usage_totals, duration_sec=round(time.time() - t0, 1),
                        error=f"Step {step}: {err_str}",
                    )

        if step_error is not None:
            return AFCResult(
                tool_calls=trace,
                n_tool_calls=sum(1 for tc in trace if tc.get("type") == "tool_call"),
                usage=usage_totals, duration_sec=round(time.time() - t0, 1),
                error=step_error,
            )

        usage_totals["prompt_tokens"] += step_usage.get("prompt_tokens", 0)
        usage_totals["candidates_tokens"] += step_usage.get("completion_tokens", 0)
        usage_totals["thoughts_tokens"] += step_usage.get("reasoning_tokens", 0)
        usage_totals["total_tokens"] += step_usage.get("total_tokens", 0)

        if content:
            trace.append({"type": "output_text", "output": content})

        if not step_tool_calls:
            break

        messages.append(_build_assistant_msg(content, step_tool_calls))

        for tc in step_tool_calls:
            tool_name = tc["name"]
            tool_args = tc["arguments"]
            try:
                mcp_result = await mcp_client.call_tool(tool_name, tool_args)
                tool_output = mcp_result[0].text if mcp_result else ""
            except Exception as e:
                tool_output = f"Error: {e}"
                logger.warning("MCP tool %s failed: %s", tool_name, e)

            trace.append({
                "type": "tool_call", "tool_name": tool_name,
                "arguments": tool_args, "output": tool_output,
            })
            messages.append({
                "role": "tool", "tool_call_id": tc["id"], "content": tool_output,
            })

    final_text = ""
    for tc in reversed(trace):
        if tc.get("type") == "output_text":
            final_text = tc["output"]
            break

    return AFCResult(
        final_text=final_text, tool_calls=trace,
        n_tool_calls=sum(1 for tc in trace if tc.get("type") == "tool_call"),
        usage=usage_totals, duration_sec=round(time.time() - t0, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# ANTHROPIC
# ═══════════════════════════════════════════════════════════════════════════

_anthropic_client = None


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.AsyncAnthropic()
    return _anthropic_client


def _mcp_tools_to_anthropic(mcp_tools: list) -> list:
    """Convert MCP tools to Anthropic tool format."""
    return [{
        "name": t.name,
        "description": t.description or "",
        "input_schema": t.inputSchema or {"type": "object", "properties": {}},
    } for t in mcp_tools]


async def _run_anthropic_manual(
    model: str,
    prompt: str,
    mcp_client,
    system_prompt: Optional[str] = None,
    max_steps: int = 50,
    max_output_tokens: int = 65536,
    step_timeout: float = 120,
    max_retries: int = 5,
    thinking_level: Optional[str] = "HIGH",
    cancelled: Optional[asyncio.Event] = None,
) -> AFCResult:
    """Anthropic manual tool-calling loop with MCP."""
    client = _get_anthropic_client()
    base_model = model.split("/", 1)[-1] if "/" in model else model
    mcp_tools = await mcp_client.list_tools()
    tools = _mcp_tools_to_anthropic(mcp_tools)

    messages = [{"role": "user", "content": prompt}]
    trace = []
    usage_totals = {"prompt_tokens": 0, "candidates_tokens": 0, "thoughts_tokens": 0, "total_tokens": 0}
    t0 = time.time()

    # Thinking config
    thinking_str = str(thinking_level).lower() if thinking_level else "none"
    use_thinking = thinking_str and thinking_str != "none"

    # Respect per-model max_tokens limits
    effective_max_tokens = min(max_output_tokens, 64000) if "haiku" in base_model else max_output_tokens

    for step in range(max_steps):
        if cancelled and cancelled.is_set():
            break

        step_error = None
        response = None

        for attempt in range(max(1, max_retries)):
            try:
                kwargs = dict(
                    model=base_model,
                    max_tokens=effective_max_tokens,
                    messages=messages,
                    tools=tools,
                )
                if system_prompt:
                    kwargs["system"] = system_prompt
                if use_thinking:
                    budget_map = {"low": 2048, "medium": 8192, "high": 16384}
                    budget = budget_map.get(thinking_str, 16384)
                    kwargs["temperature"] = 1
                    # Use adaptive thinking for opus/sonnet 4+
                    if "opus-4" in base_model or "sonnet-4" in base_model:
                        effort_map = {"low": "low", "medium": "medium", "high": "max"} if "opus" in base_model else {"low": "low", "medium": "medium", "high": "high"}
                        kwargs["thinking"] = {"type": "adaptive"}
                        kwargs["output_config"] = {"effort": effort_map.get(thinking_str, "high")}
                        kwargs["max_tokens"] = max(effective_max_tokens, budget + effective_max_tokens)
                    else:
                        kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                        kwargs["max_tokens"] = max(effective_max_tokens, budget + effective_max_tokens)

                # Use streaming for large max_tokens to avoid SDK timeout
                async with client.messages.stream(**kwargs) as stream:
                    response = await asyncio.wait_for(
                        stream.get_final_message(),
                        timeout=step_timeout,
                    )
                break
            except asyncio.TimeoutError:
                step_error = f"Step {step} timed out after {step_timeout}s"
                logger.warning("Anthropic loop: %s (attempt %d/%d)", step_error, attempt + 1, max_retries)
            except Exception as e:
                err_str = str(e)
                if _is_retryable(err_str) and attempt < max_retries - 1:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning("Anthropic step %d retry %d/%d, waiting %ds: %s",
                                   step, attempt + 1, max_retries, wait, err_str[:120])
                    await asyncio.sleep(wait)
                    step_error = err_str
                else:
                    return AFCResult(
                        tool_calls=trace,
                        n_tool_calls=sum(1 for tc in trace if tc.get("type") == "tool_call"),
                        usage=usage_totals, duration_sec=round(time.time() - t0, 1),
                        error=f"Step {step}: {err_str}",
                    )

        if response is None:
            return AFCResult(
                tool_calls=trace,
                n_tool_calls=sum(1 for tc in trace if tc.get("type") == "tool_call"),
                usage=usage_totals, duration_sec=round(time.time() - t0, 1),
                error=step_error or f"Step {step} failed after {max_retries} attempts",
            )

        # Accumulate usage
        resp_usage = getattr(response, "usage", None)
        if resp_usage:
            usage_totals["prompt_tokens"] += getattr(resp_usage, "input_tokens", 0) or 0
            usage_totals["candidates_tokens"] += getattr(resp_usage, "output_tokens", 0) or 0
            usage_totals["total_tokens"] += (
                (getattr(resp_usage, "input_tokens", 0) or 0) +
                (getattr(resp_usage, "output_tokens", 0) or 0)
            )

        # Parse response blocks
        tool_use_blocks = []
        text_content = ""
        for block in response.content:
            if block.type == "thinking":
                trace.append({"type": "reasoning", "output": block.thinking})
            elif block.type == "text":
                text_content += block.text
            elif block.type == "tool_use":
                tool_use_blocks.append(block)

        if text_content:
            trace.append({"type": "output_text", "output": text_content})

        if not tool_use_blocks:
            break

        # Build assistant message with all content blocks
        assistant_content = []
        for block in response.content:
            if block.type == "thinking":
                assistant_content.append({"type": "thinking", "thinking": block.thinking})
            elif block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_content})

        # Execute tool calls and build tool results
        tool_results = []
        for block in tool_use_blocks:
            tool_name = block.name
            tool_args = block.input or {}
            try:
                mcp_result = await mcp_client.call_tool(tool_name, tool_args)
                tool_output = mcp_result[0].text if mcp_result else ""
            except Exception as e:
                tool_output = f"Error: {e}"
                logger.warning("MCP tool %s failed: %s", tool_name, e)

            trace.append({
                "type": "tool_call", "tool_name": tool_name,
                "arguments": tool_args, "output": tool_output,
            })
            tool_results.append({
                "type": "tool_result", "tool_use_id": block.id,
                "content": tool_output,
            })

        messages.append({"role": "user", "content": tool_results})

    final_text = ""
    for tc in reversed(trace):
        if tc.get("type") == "output_text":
            final_text = tc["output"]
            break

    return AFCResult(
        final_text=final_text, tool_calls=trace,
        n_tool_calls=sum(1 for tc in trace if tc.get("type") == "tool_call"),
        usage=usage_totals, duration_sec=round(time.time() - t0, 1),
    )


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API — provider-agnostic dispatch
# ═══════════════════════════════════════════════════════════════════════════

async def run_inference(
    model: str,
    prompt: str,
    mcp_client,
    system_prompt: Optional[str] = None,
    max_steps: int = 50,
    max_output_tokens: int = 65536,
    step_timeout: float = 120,
    max_retries: int = 5,
    thinking_level: Optional[str] = "HIGH",
    cancelled: Optional[asyncio.Event] = None,
    gemini_client=None,
) -> AFCResult:
    """Run multi-turn tool-calling inference with automatic provider dispatch.

    Args:
        model: Model identifier in 'provider/model-name' format
               (e.g. "openai/gpt-5.4-nano", "anthropic/claude-opus-4-6",
               "google/gemini-3-flash-preview").
        prompt: User prompt / question.
        mcp_client: An MCP client (from fastmcp) connected to a tool server.
        system_prompt: Optional system instructions (includes playbook).
        gemini_client: Pre-created Gemini client (reused across calls for efficiency).
                       Only needed for Gemini models; ignored for other providers.

    Returns:
        AFCResult with the agent's trace, final text, and usage stats.
    """
    provider, base_model = _parse_provider(model)
    kwargs = dict(
        prompt=prompt,
        mcp_client=mcp_client,
        system_prompt=system_prompt,
        max_steps=max_steps,
        max_output_tokens=max_output_tokens,
        step_timeout=step_timeout,
        max_retries=max_retries,
        thinking_level=thinking_level,
        cancelled=cancelled,
    )

    if provider == "openai":
        return await _run_openai_manual(model=model, **kwargs)
    elif provider == "anthropic":
        return await _run_anthropic_manual(model=model, **kwargs)
    else:  # google
        if gemini_client is None:
            gemini_client = create_gemini_client()
        return await _run_gemini_manual(client=gemini_client, model=base_model, **kwargs)
