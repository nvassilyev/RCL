"""SystemAdapter for AppWorld via manual tool-calling loop.

Supports Gemini, OpenAI, and Anthropic models. Each worker gets a dedicated
AppWorld server via port pool.

The manual loop is more resilient than AFC because:
- Each tool call is isolated; a server error at step N doesn't kill steps 1..N-1
- AppWorldClient retry logic (reset freezegun → reset server → restart) works
  because the environment persists across retries within a step
- The LLM can observe and adapt to tool errors naturally
"""

import atexit
import concurrent.futures
import logging
import queue
import random
import re
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import json

from rcl.core.data_structures import ExecutionTrace, Playbook
from rcl.core.interfaces import SystemAdapter
from rcl.core.trace_writer import (
    TraceWriter,
    build_rollout_descriptors,
    rollout_metadata,
)
from rcl.components.inference import _parse_provider

from .appworld_client import AppWorldClient

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTIONS = """\
I am your supervisor, and you are an AI Assistant whose job is to complete my day-to-day tasks fully autonomously.

To do this, you will need to interact with app(s) (e.g., spotify, venmo etc) using their associated APIs on my behalf. \
Use the execute_python tool to run code one step at a time, observe the output, then decide the next step.

Key APIs:
  execute_python(code="print(apis.api_docs.show_app_descriptions())")
  execute_python(code="print(apis.api_docs.show_api_descriptions(app_name='spotify'))")
  execute_python(code="print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))")

**Key instructions**:

A. General instructions:
- Act fully on your own. Never ask for confirmation.
- Never invent or guess values - always look them up via APIs.
- Never leave placeholders - always fill in real values.

B. App-specific instructions:
- All credentials are in the Supervisor app: `apis.supervisor.show_account_passwords()`
- References to friends/family refer to phone contacts.
- Get current date/time from `datetime.now()` or phone API, never your internal clock.
- Paginated APIs: Always loop through all pages. Don't stop at first page.

C. Code-operation instructions:
- Use the execute_python tool to run code. Execute one step at a time, observe its output, then decide the next step.
- Always use print() to see API return values - otherwise output is hidden.
- Always check API specs before calling an API.

D. Task-completion:
- Call `apis.supervisor.complete_task(answer=...)` when done.
- Keep answers minimal (just the value, not full sentences).
- Numbers must be numeric ("10" not "ten")."""

PLAYBOOK_HEADER = """

# Playbook

You have been given a curated playbook of strategies, common mistakes, and proven solutions. \
Read it carefully and actively apply its guidance throughout your task execution.

"""

INFRA_RETRY_LIMIT = 2
MAX_LLM_RETRIES = 5
LLM_CALL_TIMEOUT = 120


def _build_system_prompt(playbook: Playbook, app_descriptions: str = "") -> str:
    """Build system prompt: instructions + app descriptions + playbook."""
    parts = [SYSTEM_INSTRUCTIONS]
    if app_descriptions:
        parts.append(f"\nAvailable apps:\n{app_descriptions}")
    if len(playbook) > 0:
        parts.append(PLAYBOOK_HEADER + playbook.to_prompt())
    return "\n".join(parts)


def _find_free_port_block(count: int = 10) -> int:
    """Find a base port where `count` consecutive ports are all free.

    Uses os.getpid() + time as seed to ensure different processes get
    different port ranges even when launched simultaneously with the same
    global random seed.
    """
    import os
    # Use urandom for true randomness — avoids port collisions between
    # processes launched simultaneously with the same global random seed
    rng = random.Random(int.from_bytes(os.urandom(8), 'big'))
    for _ in range(100):
        candidate = rng.randint(20000, 50000 - count)
        sockets = []
        all_free = True
        for offset in range(count):
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(("127.0.0.1", candidate + offset))
                sockets.append(s)
            except OSError:
                all_free = False
                s.close()
                break
        for s in sockets:
            s.close()
        if all_free:
            return candidate
    raise RuntimeError("Could not find a block of free ports")


def _build_tool_declaration():
    """Build the execute_python tool declaration for Gemini."""
    from google.genai import types
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="execute_python",
                description="Execute Python code in the AppWorld REPL environment. Always use print() to see output.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "code": {
                            "type": "STRING",
                            "description": "Python code to execute",
                        }
                    },
                    "required": ["code"],
                },
            )
        ]
    )


def _is_rate_limit(e: Exception) -> bool:
    msg = str(e)
    return '429' in msg or 'RESOURCE_EXHAUSTED' in msg


def _is_infra_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in [
        "server disconnected", "connection refused", "connect error",
        "remoteerror", "remoteprotocolerror", "500 internal server",
        "502 bad gateway", "503 service unavailable", "404 not found",
    ])


class AppWorldSystemAdapter(SystemAdapter):
    """System adapter for AppWorld using manual tool-calling loop.

    Each concurrent worker gets its own dedicated AppWorld server.
    Uses Gemini's function-calling with manual loop (not AFC) for resilience.
    """

    def __init__(
        self,
        model: str = "google/gemini-3-flash-preview",
        max_remote_calls: int = 100,
        trace_writer: Optional[TraceWriter] = None,
        thinking_level: Optional[str] = "HIGH",
        n_concurrent: int = 5,
        task_timeout: int = 900,  # 15 minutes per task
    ):
        self.model = model
        self.max_remote_calls = max_remote_calls
        self.thinking_level = thinking_level
        self.n_concurrent = n_concurrent
        self.task_timeout = task_timeout
        self.trace_writer = trace_writer

        self._provider, self._base_model = _parse_provider(model)
        self._gemini = None
        if self._provider == "google":
            from rcl.components.inference import create_gemini_client
            self._gemini = create_gemini_client()

        # Server pool
        self._base_port = _find_free_port_block(count=n_concurrent + 10)
        self._pool: Dict[int, AppWorldClient] = {}
        self._servers_we_started: List[AppWorldClient] = []
        self._client = self._get_or_create_slot(0)
        atexit.register(self._shutdown)

    def clone_for_parallel(self) -> "AppWorldSystemAdapter":
        """Create an isolated adapter with its own server pool.

        Frontier-style parallel slot execution must not share AppWorld server
        clients across slots; each clone gets a fresh port block and pool.
        """
        return AppWorldSystemAdapter(
            model=self.model,
            max_remote_calls=self.max_remote_calls,
            trace_writer=self.trace_writer,
            thinking_level=self.thinking_level,
            n_concurrent=self.n_concurrent,
            task_timeout=self.task_timeout,
        )

    def _get_or_create_slot(self, index: int) -> AppWorldClient:
        if index in self._pool:
            return self._pool[index]
        port = self._base_port + index
        client = AppWorldClient(base_url=f"http://127.0.0.1:{port}")
        if not client.health_check():
            if not client.start_server(wait_time=40):
                # Port collision — find a new free port and retry
                for fallback_attempt in range(5):
                    fallback_port = _find_free_port_block(count=1)
                    client = AppWorldClient(base_url=f"http://127.0.0.1:{fallback_port}")
                    if client.health_check() or client.start_server(wait_time=40):
                        break
                else:
                    raise RuntimeError(f"Failed to start AppWorld server on port {port} (and 5 fallback attempts)")
            self._servers_we_started.append(client)
        self._pool[index] = client
        return client

    def _shutdown(self):
        for client in self._servers_we_started:
            try:
                client.stop_server()
            except Exception:
                pass
        self._servers_we_started.clear()

    def execute(
        self,
        task_ids: List[str],
        playbook: Playbook,
        experiment_prefix: str = "eval",
        max_workers: int = 1,
        verbose: bool = True,
        trace_subdir: str = "traces",
        playbook_overrides: Optional[Dict[str, "Playbook"]] = None,
    ) -> List[ExecutionTrace]:
        n_workers = min(self.n_concurrent, len(task_ids))
        if verbose:
            print(f"    Running {len(task_ids)} tasks with {n_workers} workers, {n_workers} servers", flush=True)

        batch_start = time.time()

        # Start all server slots in parallel
        with ThreadPoolExecutor(max_workers=n_workers) as startup_pool:
            list(startup_pool.map(self._get_or_create_slot, range(n_workers)))

        task_queue: queue.Queue = queue.Queue()
        for task_id in task_ids:
            task_queue.put(task_id)

        all_traces: List[ExecutionTrace] = []

        task_timeout = self.task_timeout
        _playbook_overrides = playbook_overrides or {}

        def _run_with_timeout(client, task_id):
            """Run task with per-task timeout using cooperative cancellation."""
            task_playbook = _playbook_overrides.get(task_id, playbook)
            cancelled = threading.Event()
            if task_timeout and task_timeout > 0:
                with ThreadPoolExecutor(max_workers=1) as tp:
                    future = tp.submit(self._execute_single, client, task_id, task_playbook, experiment_prefix, cancelled)
                    try:
                        return future.result(timeout=task_timeout)
                    except concurrent.futures.TimeoutError:
                        cancelled.set()  # Signal the tool loop to stop
                        raise
            return self._execute_single(client, task_id, task_playbook, experiment_prefix, cancelled)

        def _worker(slot: int) -> List[ExecutionTrace]:
            client = self._pool[slot]
            worker_traces = []
            while True:
                try:
                    task_id = task_queue.get_nowait()
                except queue.Empty:
                    break
                t0 = time.time()
                trace = None
                for attempt in range(1 + INFRA_RETRY_LIMIT):
                    try:
                        trace = _run_with_timeout(client, task_id)
                    except concurrent.futures.TimeoutError:
                        elapsed = time.time() - t0
                        if verbose:
                            print(f"    [{task_id}] TIMEOUT after {elapsed:.0f}s (limit={task_timeout}s)", flush=True)
                        trace = ExecutionTrace(
                            task_id=task_id, input_query="", system_output=None,
                            trace=f"Timeout after {elapsed:.0f}s",
                            metadata={
                                "pass_pct": 0.0, "task_completed": False,
                                "error": "timeout", "timed_out": True,
                                "evaluation_details": f"Timeout after {elapsed:.0f}s",
                            },
                        )
                        break  # Don't retry timeouts
                    except Exception as e:
                        if _is_infra_error(e) and attempt < INFRA_RETRY_LIMIT:
                            elapsed = time.time() - t0
                            if verbose:
                                print(f"    [{task_id}] INFRA ERROR (retry {attempt+1}/{INFRA_RETRY_LIMIT}): {e} ({elapsed:.1f}s)", flush=True)
                            client.restart_server()
                            continue
                        elapsed = time.time() - t0
                        is_infra = _is_infra_error(e)
                        label = "INFRA ERROR (final)" if is_infra else "ERROR"
                        logger.error("Task %s %s: %s", task_id, label, e)
                        trace = ExecutionTrace(
                            task_id=task_id, input_query="", system_output=None,
                            trace=f"Error: {e}",
                            metadata={
                                "pass_pct": 0.0, "task_completed": False,
                                "error": str(e), "evaluation_details": f"Error: {e}",
                                "infra_error": is_infra,
                            },
                        )
                        break
                    else:
                        break
                elapsed = time.time() - t0
                trace.metadata["duration_s"] = round(elapsed, 1)
                worker_traces.append(trace)
                if verbose:
                    status = "PASS" if trace.metadata.get("task_completed") else "FAIL"
                    pct = trace.metadata.get("pass_pct", 0) * 100
                    calls = trace.metadata.get("n_tool_calls", 0)
                    print(f"    [{task_id}] pass_pct={pct:.1f}% {status} ({elapsed:.1f}s, {calls} calls)", flush=True)
            return worker_traces

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_worker, i) for i in range(n_workers)]
            for future in as_completed(futures):
                try:
                    all_traces.extend(future.result())
                except Exception as e:
                    if verbose:
                        print(f"    [Worker] ERROR: {e}", flush=True)

        batch_elapsed = time.time() - batch_start

        # Group traces by task_id, preserving duplicates for K-rollout support
        from collections import defaultdict
        trace_queues = defaultdict(list)
        for t in all_traces:
            trace_queues[t.task_id].append(t)

        rollout_descriptors = build_rollout_descriptors(task_ids)
        traces = []
        for task_id, rollout in zip(task_ids, rollout_descriptors):
            if trace_queues[task_id]:
                trace = trace_queues[task_id].pop(0)
            else:
                trace = ExecutionTrace(
                    task_id=task_id, input_query="", system_output=None,
                    trace="Error: missing result",
                    metadata={"pass_pct": 0.0, "task_completed": False,
                              "error": "missing_result", "evaluation_details": "Error: missing result"},
                )
            trace.metadata.update(rollout_metadata(rollout))
            traces.append(trace)
            if self.trace_writer:
                self.trace_writer.write_trace(task_id, {
                    "task_id": task_id,
                    "instruction": trace.input_query,
                    "result": {
                        "pass_pct": trace.metadata.get("pass_pct", 0),
                        "task_completed": trace.metadata.get("task_completed", False),
                        "n_tool_calls": trace.metadata.get("n_tool_calls", 0),
                        "duration_s": trace.metadata.get("duration_s", 0),
                    },
                    "test_report": trace.metadata.get("test_report", ""),
                    "afc_trace": trace.metadata.get("afc_trace", []),
                    "usage": trace.metadata.get("usage", {}),
                    "rollout": rollout_metadata(rollout),
                }, subdir=trace_subdir, artifact_id=str(rollout["artifact_id"]))

        if verbose:
            n_pass = sum(1 for t in traces if t.metadata.get("task_completed"))
            print(f"    Batch: {len(traces)} tasks in {batch_elapsed:.1f}s | pass={n_pass}/{len(traces)}", flush=True)

        return traces

    # ------------------------------------------------------------------
    # Manual tool-calling loop
    # ------------------------------------------------------------------

    def _execute_single(
        self, client: AppWorldClient, task_id: str,
        playbook: Playbook, prefix: str,
        cancelled: Optional[threading.Event] = None,
    ) -> ExecutionTrace:
        """Execute a single task using manual tool-calling loop."""
        # Strip __pp__ prefix for AppWorld API calls (used for playbook_overrides routing)
        raw_task_id = task_id
        appworld_task_id = task_id.removeprefix("__pp__")
        task_info = client.get_task_info(appworld_task_id)
        instruction = task_info.get("instruction", "")
        main_user = task_info.get("main_user", {})

        env_data = client.create_environment(appworld_task_id, f"{prefix}_{raw_task_id}")
        env_id = env_data["env_id"]

        system_prompt = _build_system_prompt(
            playbook, app_descriptions=task_info.get("app_descriptions", ""),
        )
        user_prompt = (
            f"My name is: {main_user.get('first_name', 'User')} {main_user.get('last_name', '')}. "
            f"My personal email is {main_user.get('email', '')} "
            f"and phone number is {main_user.get('phone_number', '')}.\n"
            f"Task: {instruction}"
        )

        # Run the manual tool loop (dispatch by provider)
        if self._provider == "openai":
            tool_calls, n_turns, task_completed_early = self._run_tool_loop_openai(
                client, env_id, system_prompt, user_prompt, cancelled,
            )
        elif self._provider == "anthropic":
            tool_calls, n_turns, task_completed_early = self._run_tool_loop_anthropic(
                client, env_id, system_prompt, user_prompt, cancelled,
            )
        else:
            tool_calls, n_turns, task_completed_early = self._run_tool_loop(
                client, env_id, system_prompt, user_prompt, cancelled,
            )

        # Close environment and get test results
        close_result = client.close_environment(env_id)
        pass_pct = close_result.get("pass_percentage", 0.0) / 100.0
        task_completed = close_result.get("task_completed", False)
        test_report = close_result.get("test_report", "")

        trace_str = self._format_trace(instruction, tool_calls)

        eval_parts = [
            f"Pass percentage: {pass_pct * 100:.1f}%",
            f"Task completed: {task_completed}",
        ]
        if test_report:
            eval_parts.append(f"\n{test_report}")

        return ExecutionTrace(
            task_id=task_id,
            input_query=instruction,
            system_output=task_completed,
            trace=trace_str,
            metadata={
                "pass_pct": pass_pct,
                "task_completed": task_completed,
                "test_report": test_report,
                "n_tool_calls": n_turns,
                "evaluation_details": "\n".join(eval_parts),
                "afc_trace": tool_calls,
                "usage": {},
            },
        )

    def _run_tool_loop(
        self, client: AppWorldClient, env_id: str,
        system_prompt: str, user_prompt: str,
        cancelled: Optional[threading.Event] = None,
    ) -> Tuple[List[Dict], int, bool]:
        """Run manual tool-calling loop (Gemini).

        Returns (tool_calls, n_turns, task_completed_early).
        Checks `cancelled` event between steps for cooperative timeout.
        """
        from google.genai import types

        tool = _build_tool_declaration()
        thinking_config = None
        if self.thinking_level and self.thinking_level.lower() != "none":
            thinking_config = types.ThinkingConfig(
                include_thoughts=True,
                thinking_level=self.thinking_level,
            )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[tool],
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY")
            ),
            max_output_tokens=65536,
        )
        if thinking_config:
            config.thinking_config = thinking_config

        # Start conversation
        contents = [types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_prompt)],
        )]

        tool_calls = []
        n_turns = 0

        for step in range(self.max_remote_calls):
            # Check cooperative cancellation (set by timeout handler)
            if cancelled and cancelled.is_set():
                break

            n_turns += 1

            # Generate with retries
            response = self._generate_with_retries(contents, config)

            # Extract function call and thinking from response
            function_call = None
            model_text = ""
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'function_call') and part.function_call:
                    function_call = part.function_call
                if hasattr(part, 'text') and part.text:
                    if getattr(part, 'thought', False):
                        tool_calls.append({"type": "reasoning", "output": part.text})
                    else:
                        model_text += part.text

            if function_call is None:
                # Model didn't call a tool — nudge it
                contents.append(response.candidates[0].content)
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text="Please use the execute_python tool to run code.")],
                ))
                continue

            code = function_call.args.get("code", "")

            # Execute code via AppWorld (with built-in retry in AppWorldClient)
            exec_result = client.execute_code(env_id, code)
            output = exec_result.get("output", "")
            error = exec_result.get("error")
            task_completed = exec_result.get("task_completed", False)

            result_text = f"Error: {output}" if error else output

            tool_calls.append({
                "type": "tool_call",
                "tool_name": "execute_python",
                "arguments": {"code": code},
                "output": result_text,
            })

            if task_completed:
                return tool_calls, n_turns, True

            # Feed result back to model
            contents.append(response.candidates[0].content)
            contents.append(types.Content(
                role="user",
                parts=[types.Part.from_function_response(
                    name="execute_python",
                    response={"output": result_text},
                )],
            ))

        # Add final text if model produced any
        if model_text:
            tool_calls.append({"type": "text", "output": model_text})

        return tool_calls, n_turns, False

    def _run_tool_loop_openai(
        self, client: AppWorldClient, env_id: str,
        system_prompt: str, user_prompt: str,
        cancelled: Optional[threading.Event] = None,
    ) -> Tuple[List[Dict], int, bool]:
        """Run manual tool-calling loop using OpenAI Responses API."""
        import openai as _openai

        oai_client = _openai.OpenAI()
        model_name = self._base_model

        tools = [{
            "type": "function",
            "name": "execute_python",
            "description": "Execute Python code in the AppWorld REPL environment. Always use print() to see output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"],
            },
        }]

        # Build initial input
        input_items = [{"role": "user", "content": user_prompt}]

        tool_calls = []
        n_turns = 0
        model_text = ""

        for step in range(self.max_remote_calls):
            if cancelled and cancelled.is_set():
                break

            n_turns += 1

            # Call OpenAI Responses API with retries
            resp = None
            for attempt in range(MAX_LLM_RETRIES):
                try:
                    resp_kwargs = {
                        "model": model_name,
                        "instructions": system_prompt,
                        "input": input_items,
                        "tools": tools,
                        "tool_choice": "required",
                    }
                    if self.thinking_level and self.thinking_level.lower() != "none":
                        resp_kwargs["reasoning"] = {"effort": self.thinking_level.lower()}
                    resp = oai_client.responses.create(**resp_kwargs)
                    break
                except Exception as exc:
                    if attempt >= MAX_LLM_RETRIES - 1 or not _is_rate_limit(exc):
                        raise
                    wait = min(30, 2 ** attempt) * (0.8 + 0.4 * random.random())
                    print(f"LLM retry {attempt+1}/{MAX_LLM_RETRIES} (error): {str(exc)[:100]}, waiting {wait:.1f}s", flush=True)
                    time.sleep(wait)

            # Parse response
            function_call = None
            for item in resp.output:
                if item.type == "function_call":
                    try:
                        args = json.loads(item.arguments)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    function_call = {"call_id": item.call_id, "name": item.name, "args": args}
                elif item.type == "message":
                    for part in item.content:
                        if getattr(part, "type", None) == "output_text":
                            model_text += part.text
                elif item.type == "reasoning":
                    summary = getattr(item, "summary", None)
                    if summary:
                        reasoning_text = "\n".join(s.get("text", "") for s in summary if isinstance(s, dict))
                        if reasoning_text:
                            tool_calls.append({"type": "reasoning", "output": reasoning_text})

            if function_call is None:
                # No tool call — add assistant text and nudge
                if model_text:
                    input_items.append({"role": "assistant", "content": model_text})
                    model_text = ""
                input_items.append({"role": "user", "content": "Please use the execute_python tool to run code."})
                continue

            code = function_call["args"].get("code", "")

            exec_result = client.execute_code(env_id, code)
            output = exec_result.get("output", "")
            error = exec_result.get("error")
            task_completed = exec_result.get("task_completed", False)

            result_text = f"Error: {output}" if error else output

            tool_calls.append({
                "type": "tool_call",
                "tool_name": "execute_python",
                "arguments": {"code": code},
                "output": result_text,
            })

            if task_completed:
                return tool_calls, n_turns, True

            # Feed back to model for next turn
            input_items.append({
                "type": "function_call",
                "call_id": function_call["call_id"],
                "name": function_call["name"],
                "arguments": json.dumps(function_call["args"]),
            })
            input_items.append({
                "type": "function_call_output",
                "call_id": function_call["call_id"],
                "output": result_text,
            })

        if model_text:
            tool_calls.append({"type": "text", "output": model_text})

        return tool_calls, n_turns, False

    def _run_tool_loop_anthropic(
        self, client: AppWorldClient, env_id: str,
        system_prompt: str, user_prompt: str,
        cancelled: Optional[threading.Event] = None,
    ) -> Tuple[List[Dict], int, bool]:
        """Run manual tool-calling loop using Anthropic Messages API."""
        import anthropic as _anthropic

        ant_client = _anthropic.Anthropic()
        model_name = self._base_model

        tools = [{
            "name": "execute_python",
            "description": "Execute Python code in the AppWorld REPL environment. Always use print() to see output.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"],
            },
        }]

        messages = [{"role": "user", "content": user_prompt}]
        tool_calls = []
        n_turns = 0
        model_text = ""

        # Thinking config
        thinking_str = (self.thinking_level or "none").lower()
        use_thinking = thinking_str != "none"

        # Respect per-model max_tokens limits
        base_max_tokens = 64000 if "haiku" in model_name else 65536

        for step in range(self.max_remote_calls):
            if cancelled and cancelled.is_set():
                break

            n_turns += 1

            # Build kwargs
            kwargs = dict(
                model=model_name,
                max_tokens=base_max_tokens,
                system=system_prompt,
                messages=messages,
                tools=tools,
                tool_choice={"type": "any"},
            )
            if use_thinking:
                budget_map = {"low": 2048, "medium": 8192, "high": 16384}
                budget = budget_map.get(thinking_str, 16384)
                kwargs["temperature"] = 1
                if "opus-4" in model_name or "sonnet-4" in model_name:
                    effort_map = {"low": "low", "medium": "medium", "high": "max"} if "opus" in model_name else {"low": "low", "medium": "medium", "high": "high"}
                    kwargs["thinking"] = {"type": "adaptive"}
                    kwargs["output_config"] = {"effort": effort_map.get(thinking_str, "high")}
                    kwargs["max_tokens"] = max(base_max_tokens, budget + base_max_tokens)
                else:
                    kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
                    kwargs["max_tokens"] = max(base_max_tokens, budget + base_max_tokens)

            # Call with retries (use streaming for large max_tokens)
            resp = None
            for attempt in range(MAX_LLM_RETRIES):
                try:
                    with ant_client.messages.stream(**kwargs) as stream:
                        for _ in stream:
                            pass
                    resp = stream.get_final_message()
                    break
                except Exception as exc:
                    if attempt >= MAX_LLM_RETRIES - 1 or not _is_rate_limit(exc):
                        raise
                    wait = min(30, 2 ** attempt) * (0.8 + 0.4 * random.random())
                    print(f"LLM retry {attempt+1}/{MAX_LLM_RETRIES} (error): {str(exc)[:100]}, waiting {wait:.1f}s", flush=True)
                    time.sleep(wait)

            # Parse response
            tool_use_block = None
            assistant_content = []
            for block in resp.content:
                if block.type == "thinking":
                    tool_calls.append({"type": "reasoning", "output": block.thinking})
                    assistant_content.append({"type": "thinking", "thinking": block.thinking})
                elif block.type == "text":
                    model_text += block.text
                    assistant_content.append({"type": "text", "text": block.text})
                elif block.type == "tool_use":
                    tool_use_block = block
                    assistant_content.append({
                        "type": "tool_use", "id": block.id,
                        "name": block.name, "input": block.input,
                    })

            if tool_use_block is None:
                # No tool call — nudge
                messages.append({"role": "assistant", "content": assistant_content})
                messages.append({"role": "user", "content": "Please use the execute_python tool to run code."})
                continue

            code = (tool_use_block.input or {}).get("code", "")

            exec_result = client.execute_code(env_id, code)
            output = exec_result.get("output", "")
            error = exec_result.get("error")
            task_completed = exec_result.get("task_completed", False)

            result_text = f"Error: {output}" if error else output

            tool_calls.append({
                "type": "tool_call",
                "tool_name": "execute_python",
                "arguments": {"code": code},
                "output": result_text,
            })

            if task_completed:
                return tool_calls, n_turns, True

            # Feed back to model
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": tool_use_block.id, "content": result_text}],
            })

        if model_text:
            tool_calls.append({"type": "text", "output": model_text})

        return tool_calls, n_turns, False

    def _generate_with_retries(self, contents: list, config):
        """Generate with retries and per-call timeout (Gemini only)."""
        last_error = None
        for attempt in range(MAX_LLM_RETRIES):
            try:
                tp = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                future = tp.submit(
                    self._gemini.models.generate_content,
                    model=self._base_model, contents=contents, config=config,
                )
                try:
                    response = future.result(timeout=LLM_CALL_TIMEOUT)
                except concurrent.futures.TimeoutError:
                    future.cancel()
                    tp.shutdown(wait=False, cancel_futures=True)
                    raise TimeoutError(f"Gemini API call stuck for >{LLM_CALL_TIMEOUT}s")
                finally:
                    tp.shutdown(wait=False)

                if not response.candidates or not response.candidates[0].content.parts:
                    raise ValueError("Empty response from Gemini")

                return response
            except Exception as e:
                last_error = e
                if attempt < MAX_LLM_RETRIES - 1:
                    is_rl = _is_rate_limit(e)
                    wait = min(2 ** attempt * (1 if not is_rl else 2), 30) + random.uniform(0, 2)
                    label = "rate-limit" if is_rl else "error"
                    logger.warning("LLM retry %d/%d (%s): %s, waiting %.1fs",
                                   attempt + 1, MAX_LLM_RETRIES, label, str(e)[:100], wait)
                    time.sleep(wait)
        raise last_error

    def _format_trace(self, instruction: str, tool_calls: List[Dict]) -> str:
        """Format tool calls into readable execution trace."""
        parts = [f"## Task\n{instruction}\n"]
        step = 0
        for tc in tool_calls:
            if tc.get("type") == "reasoning":
                text = tc.get("output", "")
                if text:
                    parts.append(f"[Thinking] {text}")
            elif tc.get("type") == "tool_call":
                step += 1
                code = tc.get("arguments", {}).get("code", "")
                output = tc.get("output", "")
                if len(output) > 500:
                    output = output[:500] + "\n[REST NOT SHOWN FOR BREVITY]"
                parts.append(f"\nStep {step}:\n```python\n{code}\n```\nOutput: {output}")
        return "\n".join(parts)

    def get_ground_truth(self, task_id: str) -> Optional[str]:
        try:
            result = self._client.get_ground_truth(task_id)
            return result.get("ground_truth_code") or None
        except Exception:
            return None

    def load_tasks(self, split: str = "train", limit: Optional[int] = None) -> List[str]:
        return self._client.load_tasks(split, limit)
