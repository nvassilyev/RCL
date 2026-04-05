#!/usr/bin/env python3
"""AppWorld HTTP Server - runs in appworld conda env with pydantic<2.0.

This server exposes AppWorld functionality via HTTP so that agents using
newer SDKs (like google-genai with pydantic>=2.9) can interact with it.

Usage:
    APPWORLD_ROOT=/path/to/appworld \
      /path/to/appworld-conda/bin/python \
      benchmarks/appworld/appworld_server.py --port 8765
"""

import argparse
import json
import os
import traceback
import threading
from flask import Flask, request, jsonify

# =============================================================================
# CRITICAL: Monkey-patch freezegun BEFORE any imports that use it
# This prevents the "pop from empty list" crashes that plague AppWorld
# =============================================================================
def _patch_freezegun():
    """Make freezegun resilient to state corruption.

    The issue is that freeze_factories.pop() is called when stopping a freezer,
    but if the freezer was already stopped or the list is empty, it crashes.
    We replace freeze_factories with a custom list that handles empty pop gracefully.
    """
    try:
        import freezegun.api as fg_api

        # Create a custom list class that doesn't raise on empty pop
        class SafeList(list):
            def pop(self, index=-1):
                if len(self) == 0:
                    return None  # Return None instead of raising
                return super().pop(index)

        # Replace freeze_factories with our safe version
        if hasattr(fg_api, 'freeze_factories'):
            # Copy existing items
            old_items = list(fg_api.freeze_factories)
            fg_api.freeze_factories = SafeList(old_items)

        print("[Server] Patched freezegun freeze_factories for resilience")
        return True
    except Exception as e:
        print(f"[Server] Warning: Could not patch freezegun: {e}")
        return False

_patch_freezegun()
# =============================================================================

app = Flask(__name__)

# Global state
_tasks_cache = {}
_environments = {}
_env_lock = threading.Lock()  # Prevent concurrent environment operations


def _reset_freezegun_state():
    """Reset corrupted freezegun state.

    The freezegun library uses a global stack `freeze_factories` that can get
    corrupted when environments aren't properly closed. This clears that state.
    """
    try:
        import freezegun.api as fg_api
        # Clear the freeze_factories stack if it exists
        if hasattr(fg_api, 'freeze_factories'):
            while fg_api.freeze_factories:
                fg_api.freeze_factories.pop()
            print("Cleared freezegun.freeze_factories stack")
        # Also clear the real_time tracking
        if hasattr(fg_api, 'real_time') and fg_api.real_time:
            fg_api.real_time = None
        # Clear fake_time if exists
        if hasattr(fg_api, 'fake_time') and fg_api.fake_time:
            fg_api.fake_time = None
        return True
    except Exception as e:
        print(f"Warning: Could not reset freezegun state: {e}")
        return False


def _safe_close_all_environments():
    """Safely close all environments, handling freezegun issues."""
    global _environments
    for env_id, env in list(_environments.items()):
        try:
            env.close()
        except Exception as e:
            print(f"Warning: Failed to close env {env_id}: {e}")
    _environments = {}


def _get_reward(env):
    """Safely get reward from environment."""
    try:
        if hasattr(env, 'reward'):
            reward = env.reward() if callable(env.reward) else env.reward
            return float(reward) if reward is not None else None
    except Exception:
        pass
    return None


def _get_task_completed(env):
    """Safely get task_completed from environment."""
    try:
        if hasattr(env, 'task_completed'):
            attr = env.task_completed
            val = attr() if callable(attr) else attr
            return bool(val)
    except Exception:
        pass
    return False


def _get_test_report(eval_result) -> str:
    """Extract a human-readable test report from a TestTracker result.

    Returns a string listing each passed/failed test requirement with
    failure traces, suitable for including in reflector prompts.
    """
    lines = []
    try:
        passes = getattr(eval_result, 'passes', [])
        failures = getattr(eval_result, 'failures', [])
        lines.append(f"Total: {len(passes) + len(failures)} tests, {len(passes)} passed, {len(failures)} failed")
        lines.append("")
        if failures:
            lines.append("FAILED TESTS:")
            for f in failures:
                req = f.get("requirement", "unknown")
                trace = f.get("trace", "")
                lines.append(f"  [FAIL] {req}")
                if trace:
                    # Indent and truncate long traces
                    trace_lines = trace.strip().split('\n')
                    for tl in trace_lines[:5]:
                        lines.append(f"         {tl}")
                    if len(trace_lines) > 5:
                        lines.append(f"         ... ({len(trace_lines) - 5} more lines)")
        if passes:
            lines.append("PASSED TESTS:")
            for p in passes:
                req = p.get("requirement", "unknown")
                lines.append(f"  [PASS] {req}")
    except Exception as e:
        lines.append(f"(Could not extract test report: {e})")
    return "\n".join(lines)


def _load_appworld():
    """Lazy load appworld to avoid import errors."""
    global AppWorld, load_task_ids
    from appworld import AppWorld, load_task_ids
    return AppWorld, load_task_ids


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok"})


@app.route('/reset', methods=['POST'])
def reset():
    """Reset server state - close all environments and clear caches.

    Use this if the server gets into a bad state (e.g., freezegun errors).
    """
    try:
        with _env_lock:
            _safe_close_all_environments()
            _reset_freezegun_state()
        return jsonify({"status": "reset", "message": "All environments closed, freezegun reset"})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/reset_freezegun', methods=['POST'])
def reset_freezegun():
    """Reset just the freezegun state without closing environments.

    Use this if you're getting 'pop from empty list' errors.
    """
    try:
        _reset_freezegun_state()
        return jsonify({"status": "ok", "message": "Freezegun state reset"})
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/load_tasks', methods=['POST'])
def load_tasks():
    """Load task IDs from AppWorld.

    Request body: {"split": "train", "limit": 100}
    Returns: {"task_ids": [...], "count": N}
    """
    try:
        data = request.json
        split = data.get('split', 'train')
        limit = data.get('limit')

        AppWorld, load_task_ids = _load_appworld()
        task_ids = load_task_ids(split)

        if limit:
            task_ids = task_ids[:limit]

        return jsonify({
            "task_ids": task_ids,
            "count": len(task_ids)
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/get_task_info', methods=['POST'])
def get_task_info():
    """Get task information including instruction and API docs.

    Request body: {"task_id": "..."}
    Returns: {"instruction": "...", "allowed_apps": [...], "api_docs": {...}, "main_user": {...}}
    """
    try:
        data = request.json
        task_id = data['task_id']

        AppWorld, _ = _load_appworld()

        # Create a temporary environment to get task info
        # Retry with freezegun reset if we get a pop from empty list error
        env = None
        for attempt in range(3):
            try:
                env = AppWorld(task_id=task_id, experiment_name="info_query")
                break
            except IndexError as ie:
                if "pop from empty list" in str(ie):
                    print(f"Freezegun corruption detected (attempt {attempt+1}), resetting...")
                    _reset_freezegun_state()
                    if attempt == 2:
                        raise
                else:
                    raise
            except Exception as e:
                print(f"ERROR creating env for get_task_info: {e}")
                traceback.print_exc()
                raise

        if env is None:
            raise Exception("Failed to create environment after 3 attempts")
        task_obj = env.task

        # Get main user info for prompt templating (same as paper)
        main_user = {}
        try:
            # Get user profile from supervisor API
            profile_output = env.execute("print(apis.supervisor.show_profile())")
            if profile_output and not profile_output.startswith("Error"):
                import ast
                profile = ast.literal_eval(profile_output)
                main_user = {
                    "first_name": profile.get("first_name", ""),
                    "last_name": profile.get("last_name", ""),
                    "email": profile.get("email", ""),
                    "phone_number": profile.get("phone_number", ""),
                }
        except Exception:
            pass

        # Get app descriptions for the prompt
        app_descriptions = ""
        try:
            app_desc_output = env.execute("print(apis.api_docs.show_app_descriptions())")
            if app_desc_output:
                app_descriptions = app_desc_output
        except Exception:
            pass

        result = {
            "task_id": task_id,
            "instruction": task_obj.instruction,
            "allowed_apps": list(task_obj.allowed_apps),
            "api_docs": task_obj.api_docs,
            "main_user": main_user,
            "app_descriptions": app_descriptions,
        }

        # Close env carefully - freezegun can cause issues
        try:
            env.close()
        except Exception as close_err:
            print(f"Warning: env.close() failed for get_task_info: {close_err}")
        return jsonify(result)
    except Exception as e:
        print(f"ERROR in get_task_info: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/create_environment', methods=['POST'])
def create_environment():
    """Create a new AppWorld environment for a task.

    Request body: {"task_id": "...", "experiment_name": "..."}
    Returns: {"env_id": "...", "instruction": "...", "api_docs": {...}}
    """
    try:
        data = request.json
        task_id = data['task_id']
        experiment_name = data.get('experiment_name', 'rcl_agent')

        AppWorld, _ = _load_appworld()

        # Use lock to prevent concurrent environment creation (freezegun issues)
        # Retry with freezegun reset if we get a pop from empty list error
        with _env_lock:
            env = None
            for attempt in range(3):
                try:
                    env = AppWorld(task_id=task_id, experiment_name=experiment_name)
                    break
                except IndexError as ie:
                    if "pop from empty list" in str(ie):
                        print(f"Freezegun corruption in create_environment (attempt {attempt+1}), resetting...")
                        _reset_freezegun_state()
                        if attempt == 2:
                            raise
                    else:
                        raise

            if env is None:
                raise Exception("Failed to create environment after 3 attempts")

            env_id = f"{task_id}_{id(env)}"
            _environments[env_id] = env

        task_obj = env.task
        result = {
            "env_id": env_id,
            "task_id": task_id,
            "instruction": task_obj.instruction,
            "allowed_apps": list(task_obj.allowed_apps),
            "api_docs": task_obj.api_docs,
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/execute_code', methods=['POST'])
def execute_code():
    """Execute Python code in an AppWorld environment.

    Request body: {"env_id": "...", "code": "..."}
    Returns: {"output": "...", "task_completed": bool, "reward": float, "pass_percentage": float}
    """
    try:
        data = request.json
        env_id = data['env_id']
        code = data['code']

        if env_id not in _environments:
            return jsonify({"error": f"Environment {env_id} not found"}), 404

        env = _environments[env_id]

        try:
            output = env.execute(code)
            error = None
        except Exception as e:
            output = str(e)
            error = str(e)

        # Get pass_percentage if task is completed
        pass_pct = 0.0
        task_completed = _get_task_completed(env)
        if task_completed:
            try:
                eval_result = env.evaluate()
                if hasattr(eval_result, 'pass_percentage'):
                    pass_pct = float(eval_result.pass_percentage)
            except Exception:
                pass

        result = {
            "output": str(output) if output else "",
            "task_completed": task_completed,
            "reward": _get_reward(env),
            "pass_percentage": pass_pct,
            "error": error,
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/close_environment', methods=['POST'])
def close_environment():
    """Close an AppWorld environment and get final evaluation.

    Request body: {"env_id": "..."}
    Returns: {"closed": true, "reward": float, "pass_percentage": float, "task_completed": bool}
    """
    try:
        data = request.json
        env_id = data['env_id']

        if env_id not in _environments:
            return jsonify({"error": f"Environment {env_id} not found"}), 404

        env = _environments[env_id]
        reward = _get_reward(env)
        task_completed = _get_task_completed(env)

        # Get final evaluation
        pass_pct = 0.0
        test_report = ""
        try:
            eval_result = env.evaluate()
            if hasattr(eval_result, 'pass_percentage'):
                pass_pct = float(eval_result.pass_percentage)
            test_report = _get_test_report(eval_result)
        except Exception:
            pass

        # Close env carefully with lock - freezegun can cause issues
        with _env_lock:
            try:
                env.close()
            except IndexError as ie:
                if "pop from empty list" in str(ie):
                    print(f"Freezegun corruption in close_environment, resetting...")
                    _reset_freezegun_state()
                else:
                    print(f"Warning: env.close() failed for close_environment: {ie}")
            except Exception as close_err:
                print(f"Warning: env.close() failed for close_environment: {close_err}")
            if env_id in _environments:
                del _environments[env_id]

        return jsonify({
            "closed": True,
            "reward": reward,
            "pass_percentage": pass_pct,
            "task_completed": task_completed,
            "test_report": test_report,
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/run_task', methods=['POST'])
def run_task():
    """Run a complete task with given code steps.

    This is a convenience endpoint that creates env, runs code, and closes.

    Request body: {
        "task_id": "...",
        "experiment_name": "...",
        "code_steps": ["code1", "code2", ...]
    }
    Returns: {
        "outputs": [...],
        "task_completed": bool,
        "reward": float,
        "trace": [...]
    }
    """
    try:
        data = request.json
        task_id = data['task_id']
        experiment_name = data.get('experiment_name', 'rcl_agent')
        code_steps = data.get('code_steps', [])

        AppWorld, _ = _load_appworld()

        env = AppWorld(task_id=task_id, experiment_name=experiment_name)

        outputs = []
        trace = []

        for i, code in enumerate(code_steps):
            output = env.execute(code)
            outputs.append(output)
            trace.append({
                "step": i,
                "code": code,
                "output": output,
                "task_completed": _get_task_completed(env),
            })

            if _get_task_completed(env):
                break

        result = {
            "task_id": task_id,
            "outputs": outputs,
            "task_completed": _get_task_completed(env),
            "reward": _get_reward(env),
            "trace": trace,
        }

        env.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/get_ground_truth', methods=['POST'])
def get_ground_truth():
    """Get ground truth code for a task.

    Request body: {"task_id": "..."}
    Returns: {"ground_truth_code": "...", "answer": "..."}
    """
    try:
        data = request.json
        task_id = data['task_id']

        # Load ground truth from file
        appworld_root = os.environ.get('APPWORLD_ROOT', '')
        if not appworld_root:
            return jsonify({"error": "APPWORLD_ROOT environment variable is not set"}), 500
        gt_path = f"{appworld_root}/data/tasks/{task_id}/ground_truth"

        # Read compiled solution (ground truth code)
        solution_code = ""
        try:
            with open(f"{gt_path}/compiled_solution.py", "r") as f:
                solution_code = f.read()
        except FileNotFoundError:
            # Try regular solution.py as fallback
            try:
                with open(f"{gt_path}/solution.py", "r") as f:
                    solution_code = f.read()
            except FileNotFoundError:
                pass

        # Read answer
        answer = ""
        try:
            with open(f"{gt_path}/answer.json", "r") as f:
                answer = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

        return jsonify({
            "task_id": task_id,
            "ground_truth_code": solution_code,
            "answer": answer,
        })
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route('/evaluate_task', methods=['POST'])
def evaluate_task():
    """Evaluate a task by running agent code and checking completion.

    Request body: {
        "task_id": "...",
        "experiment_name": "...",
        "full_code": "..." (complete Python code to run)
    }
    Returns: {
        "task_completed": bool,
        "reward": float,
        "output": "...",
        "error": "..." (if any)
    }
    """
    try:
        data = request.json
        task_id = data['task_id']
        experiment_name = data.get('experiment_name', 'rcl_eval')
        full_code = data['full_code']

        AppWorld, _ = _load_appworld()

        env = AppWorld(task_id=task_id, experiment_name=experiment_name)

        try:
            output = env.execute(full_code)
            error = None
        except Exception as e:
            output = str(e)
            error = traceback.format_exc()

        # Get evaluation metrics
        pass_pct = 0.0
        try:
            eval_result = env.evaluate()
            if hasattr(eval_result, 'pass_percentage'):
                pass_pct = float(eval_result.pass_percentage)
        except Exception:
            pass

        result = {
            "task_id": task_id,
            "task_completed": _get_task_completed(env),
            "reward": _get_reward(env),
            "pass_percentage": pass_pct,
            "output": str(output) if output else "",
            "error": error,
        }

        env.close()
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def main():
    parser = argparse.ArgumentParser(description="AppWorld HTTP Server")
    parser.add_argument("--port", type=int, default=8765, help="Port to run on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    args = parser.parse_args()

    print(f"Starting AppWorld server on {args.host}:{args.port}")
    # threaded=False required because AppWorld uses signals that only work in main thread
    app.run(host=args.host, port=args.port, threaded=False)


if __name__ == "__main__":
    main()
