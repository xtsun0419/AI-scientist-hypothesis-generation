from __future__ import annotations

import os
from pathlib import Path

from .config import project_root
from .external_llm import apply_active_llm_to_env


def load_local_env(path: Path | None = None) -> Path:
    env_path = path or project_root() / "configs" / "local.env"
    if not env_path.exists():
        apply_active_llm_to_env()
        return env_path
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    apply_active_llm_to_env(override=True)
    return env_path
