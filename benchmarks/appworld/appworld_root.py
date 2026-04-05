#!/usr/bin/env python3
"""Helpers for validating the AppWorld data root."""

from __future__ import annotations

import os
from pathlib import Path


def validate_appworld_root() -> Path:
    """Return the configured AppWorld root path.

    Reads from the APPWORLD_ROOT environment variable.
    """
    root_str = os.environ.get("APPWORLD_ROOT")
    if not root_str:
        raise EnvironmentError(
            "APPWORLD_ROOT environment variable is not set. "
            "Set it to the root of your AppWorld data directory."
        )

    root = Path(root_str).expanduser()

    if not root.exists():
        raise FileNotFoundError(
            f"APPWORLD_ROOT does not exist: {root}"
        )

    data_dir = root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"APPWORLD_ROOT is missing its data directory: {data_dir}"
        )

    return root
