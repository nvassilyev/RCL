"""HTTP client for AppWorld server with robust retry logic."""

import os
import re
import subprocess
import time
from typing import Dict, List, Optional

import httpx


class AppWorldClient:
    """HTTP client for AppWorld server with robust retry and auto-restart."""

    # Server configuration — override via env vars or constructor args.
    # RCL_APPWORLD_* vars take precedence over the older APPWORLD_SERVER_* names.
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    DEFAULT_SERVER_SCRIPT = os.environ.get(
        "RCL_APPWORLD_SERVER_SCRIPT",
        os.environ.get(
            "APPWORLD_SERVER_SCRIPT",
            os.path.join(_REPO_ROOT, "benchmarks", "appworld", "appworld_server.py"),
        ),
    )
    DEFAULT_SERVER_PYTHON = os.environ.get(
        "RCL_APPWORLD_SERVER_PYTHON",
        os.environ.get("APPWORLD_SERVER_PYTHON", "python"),
    )
    DEFAULT_SERVER_CWD = os.environ.get("RCL_APPWORLD_SERVER_CWD", _REPO_ROOT)
    DEFAULT_APPWORLD_ROOT = os.environ.get("APPWORLD_ROOT", _REPO_ROOT)
    DEFAULT_LOG_DIR = os.environ.get(
        "RCL_APPWORLD_SERVER_LOG_DIR",
        os.environ.get("RCL_RUNTIME_LOG_DIR", "/tmp"),
    )

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8765",
        max_retries: int = 3,
        timeout: float = 300.0,
        server_script: Optional[str] = None,
        server_python: Optional[str] = None,
        server_cwd: Optional[str] = None,
        server_log: Optional[str] = None,
    ):
        """Initialize AppWorld client.

        Args:
            base_url: URL of the AppWorld server
            max_retries: Number of retries on failure
            timeout: Request timeout in seconds
            server_script: Path to appworld_server.py
            server_python: Python interpreter path
            server_cwd: Working directory for the server process
            server_log: Path for server log file (auto-generated per port if not set)
        """
        self.base_url = base_url
        self.max_retries = max_retries
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self._server_process = None
        self._port = base_url.rsplit(":", 1)[1].rstrip("/")

        # Server paths
        self.server_script = server_script or self.DEFAULT_SERVER_SCRIPT
        self.server_python = server_python or self.DEFAULT_SERVER_PYTHON
        self.server_cwd = server_cwd or self.DEFAULT_SERVER_CWD
        self.server_log = server_log or f"{self.DEFAULT_LOG_DIR}/appworld_server_{self._port}.log"

    def health_check(self) -> bool:
        """Check if AppWorld server is running."""
        try:
            resp = self.client.get(f"{self.base_url}/health", timeout=10.0)
            return resp.status_code == 200
        except Exception:
            return False

    def reset_server(self) -> bool:
        """Reset server state (close environments, reset freezegun)."""
        try:
            resp = self.client.post(f"{self.base_url}/reset", timeout=30.0)
            return resp.status_code == 200
        except Exception:
            return False

    def reset_freezegun(self) -> bool:
        """Reset just the freezegun state."""
        try:
            resp = self.client.post(f"{self.base_url}/reset_freezegun", timeout=10.0)
            return resp.status_code == 200
        except Exception:
            return False

    def start_server(self, wait_time: int = 20) -> bool:
        """Start the AppWorld server for this client's port.

        Skips if server is already healthy. Kills any zombie process on the
        port before starting.
        """
        if self.health_check():
            return True

        # Kill any zombie process holding the port from a previous crashed run
        self._kill_port_holder()

        print(f"    [Server:{self._port}] Starting...")
        try:
            with open(self.server_log, "a") as log_file:
                env = os.environ.copy()
                env["APPWORLD_ROOT"] = os.environ.get("APPWORLD_ROOT", self.DEFAULT_APPWORLD_ROOT)
                self._server_process = subprocess.Popen(
                    [self.server_python, self.server_script, "--port", self._port],
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=self.server_cwd,
                    env=env,
                )
        except Exception as e:
            print(f"    [Server:{self._port}] Failed to start: {e}")
            return False

        for i in range(wait_time):
            time.sleep(1)
            if self.health_check():
                print(f"    [Server:{self._port}] Ready after {i+1}s")
                return True

        print(f"    [Server:{self._port}] Failed to start within {wait_time}s")
        return False

    def stop_server(self):
        """Stop the server on this client's port.

        First tries the tracked process (if we started it). Falls back to
        killing whatever process holds the port.
        """
        if self._server_process:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._server_process.kill()
            except Exception:
                pass
            self._server_process = None
        else:
            # Kill by port — handles externally started servers
            self._kill_port_holder()

    def _kill_port_holder(self):
        """Kill whatever process is listening on our port (using ss + kill)."""
        try:
            result = subprocess.run(
                ["ss", "-tlnp", f"sport = :{self._port}"],
                timeout=5, capture_output=True, text=True,
            )
            # Parse PID from output like: users:(("python",pid=12345,fd=3))
            for m in re.finditer(r"pid=(\d+)", result.stdout):
                pid = int(m.group(1))
                os.kill(pid, 9)  # SIGKILL
        except Exception:
            pass

    def restart_server(self, wait_time: int = 20) -> bool:
        """Stop and restart the AppWorld server."""
        print(f"    [Server:{self._port}] Restarting...")
        self.stop_server()
        time.sleep(2)
        return self.start_server(wait_time)

    def ensure_server_healthy(self) -> bool:
        """Ensure server is healthy, restart if needed."""
        if self.health_check():
            return True
        print("    [Server] Not responding, attempting restart...")
        return self.restart_server()

    def _request_with_retry(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make request with retry logic and server restart on failure."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Ensure server is healthy before request
                if attempt > 0:
                    self.ensure_server_healthy()
                    time.sleep(2)

                if method == "GET":
                    resp = self.client.get(f"{self.base_url}{endpoint}", **kwargs)
                else:
                    resp = self.client.post(f"{self.base_url}{endpoint}", **kwargs)

                resp.raise_for_status()
                return resp.json()

            except Exception as e:
                last_error = e
                print(f"    [Retry {attempt+1}/{self.max_retries}] {endpoint} failed: {type(e).__name__}")

                # Try to reset freezegun first
                if attempt == 0:
                    self.reset_freezegun()
                # On subsequent failures, try full reset
                elif attempt == 1:
                    self.reset_server()
                # Last resort: restart server
                else:
                    self.restart_server()

                time.sleep(2 ** attempt)

        raise last_error

    # ==========================================================================
    # Task Management
    # ==========================================================================

    def load_tasks(self, split: str = "train", limit: Optional[int] = None) -> List[str]:
        """Load task IDs for a given split."""
        result = self._request_with_retry("POST", "/load_tasks", json={"split": split, "limit": limit})
        return result["task_ids"]

    def get_task_info(self, task_id: str) -> Dict:
        """Get task information including instruction and user info."""
        return self._request_with_retry("POST", "/get_task_info", json={"task_id": task_id})

    def get_ground_truth(self, task_id: str) -> Dict:
        """Get ground truth code for a task (for reflection).

        Returns:
            Dict with 'ground_truth_code' key containing the solution code
        """
        return self._request_with_retry("POST", "/get_ground_truth", json={"task_id": task_id})

    # ==========================================================================
    # Environment Management
    # ==========================================================================

    def create_environment(self, task_id: str, experiment_name: str = "react") -> Dict:
        """Create a new environment for task execution.

        Returns:
            Dict with 'env_id' key
        """
        return self._request_with_retry(
            "POST", "/create_environment",
            json={"task_id": task_id, "experiment_name": experiment_name}
        )

    def execute_code(self, env_id: str, code: str) -> Dict:
        """Execute code in an environment.

        Returns:
            Dict with 'output', 'error', 'task_completed', 'pass_percentage' keys
        """
        return self._request_with_retry("POST", "/execute_code", json={"env_id": env_id, "code": code})

    def close_environment(self, env_id: str) -> Dict:
        """Close an environment and get final results.

        Returns:
            Dict with 'pass_percentage', 'task_completed' keys
        """
        return self._request_with_retry("POST", "/close_environment", json={"env_id": env_id})
