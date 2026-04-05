"""Gemini AFC (Automatic Function Calling) inference client.

Gemini-specific inference helpers for BrowseComp+ task execution.
These use the google-genai SDK for multi-turn tool-calling with MCP.

For simple text completions (judging, reflection, mutation), use
rcl.components.llm_client.create_generate_fn() instead.
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

logger = logging.getLogger(__name__)


@dataclass
class AFCResult:
    """Structured result from an AFC inference run."""
    final_text: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    n_tool_calls: int = 0
    usage: Dict[str, int] = field(default_factory=dict)
    duration_sec: float = 0.0
    error: Optional[str] = None
    raw_response: Optional[Dict] = None


def create_gemini_client() -> genai.Client:
    """Create a Gemini client, auto-detecting Vertex AI vs API key mode."""
    from rcl.components.llm_client import _create_genai_client
    return _create_genai_client(genai)


def _build_tool_declarations(mcp_tools: list) -> types.Tool:
    """Convert MCP tool list to Gemini FunctionDeclarations."""
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


async def run_manual_async(
    client: genai.Client,
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
    """Run Gemini inference with a manual tool loop and per-step timeouts.

    Drives the tool-calling loop explicitly: each generate_content call gets
    its own timeout and retry logic, and MCP tools are dispatched individually.
    Returns partial results on mid-trajectory failure.
    """
    mcp_tools = await mcp_client.list_tools()
    tool_decl = _build_tool_declarations(mcp_tools)

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
                step_t0 = time.time()
                response = await asyncio.wait_for(
                    client.aio.models.generate_content(
                        model=model, contents=contents, config=config,
                    ),
                    timeout=step_timeout,
                )
                logger.info("Step %d generate_content took %.1fs", step, time.time() - step_t0)
                break
            except asyncio.TimeoutError:
                step_error = f"Step {step} timed out after {step_timeout}s"
                logger.warning("Manual loop: %s (attempt %d/%d)", step_error, attempt + 1, max_retries)
            except Exception as e:
                err_str = str(e)
                if _is_retryable(err_str) and attempt < max_retries - 1:
                    wait = min(2 ** attempt * 5, 60)
                    logger.warning("Manual loop step %d retry %d/%d, waiting %ds: %s",
                                   step, attempt + 1, max_retries, wait, err_str[:120])
                    await asyncio.sleep(wait)
                    step_error = err_str
                else:
                    return AFCResult(
                        tool_calls=tool_calls,
                        n_tool_calls=sum(1 for tc in tool_calls if tc.get("type") == "tool_call"),
                        usage=usage,
                        duration_sec=round(time.time() - t0, 1),
                        error=f"Step {step}: {err_str}",
                    )

        if response is None:
            return AFCResult(
                tool_calls=tool_calls,
                n_tool_calls=sum(1 for tc in tool_calls if tc.get("type") == "tool_call"),
                usage=usage,
                duration_sec=round(time.time() - t0, 1),
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
                mcp_t0 = time.time()
                mcp_result = await mcp_client.call_tool(tool_name, tool_args)
                tool_output = mcp_result[0].text if mcp_result else ""
                logger.info("MCP %s took %.1fs", tool_name, time.time() - mcp_t0)
            except Exception as e:
                tool_output = f"Error: {e}"
                logger.warning("MCP tool %s failed: %s", tool_name, e)

            tool_calls.append({
                "type": "tool_call",
                "tool_name": tool_name,
                "arguments": tool_args,
                "output": tool_output,
            })

            response_parts.append(types.Part.from_function_response(
                name=tool_name,
                response={"result": tool_output},
            ))

        contents.append(response.candidates[0].content)
        contents.append(types.Content(role="user", parts=response_parts))

    final_text = ""
    for tc in reversed(tool_calls):
        if tc.get("type") == "output_text":
            final_text = tc["output"]
            break

    return AFCResult(
        final_text=final_text,
        tool_calls=tool_calls,
        n_tool_calls=sum(1 for tc in tool_calls if tc.get("type") == "tool_call"),
        usage=usage,
        duration_sec=round(time.time() - t0, 1),
    )


def _is_retryable(err_str: str) -> bool:
    return any(s in err_str for s in [
        "429", "RESOURCE_EXHAUSTED", "499", "CANCELLED", "503",
        "ReadTimeout", "ConnectTimeout", "TimeoutError",
    ])
