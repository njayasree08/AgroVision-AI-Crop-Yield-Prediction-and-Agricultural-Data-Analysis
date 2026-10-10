"""Save / load complete preprocessing+model pipelines with joblib.

Security note: joblib files are pickles. Only load files this project wrote to
the local models/ folder. Uploaded files are never loaded with joblib.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import joblib

from .config import MODELS_DIR


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_bundle(bundle: dict, filename: str, directory: Path = MODELS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    bundle = {**bundle, "saved_at": datetime.now().isoformat(timespec="seconds")}
    path = directory / filename
    joblib.dump(bundle, path)
    return path


def load_bundle(filename: str, directory: Path = MODELS_DIR) -> dict | None:
    """Return the saved bundle, or None if missing/unreadable (e.g. version mismatch)."""
    path = directory / filename
    if not path.exists():
        return None
    try:
        bundle = joblib.load(path)
    except Exception:
        return None
    return bundle if isinstance(bundle, dict) and "pipeline" in bundle else None


def save_json(data: dict, filename: str, directory: Path = MODELS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path
