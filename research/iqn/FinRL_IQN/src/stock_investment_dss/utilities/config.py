from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

try:
    import yaml
except ImportError:
    yaml = None


_ENV_LOADED = False


def find_project_root() -> Path:
    """
    Find the project root by walking up from this file until a .env file
    or common project marker is found.
    """
    current = Path(__file__).resolve()

    for parent in current.parents:
        if (parent / ".env").exists():
            return parent

        if (parent / "pyproject.toml").exists():
            return parent

        if (parent / ".git").exists():
            return parent

    return Path.cwd()


def load_environment() -> Path:
    """
    Load .env once.

    Existing OS environment variables are not overwritten.
    This means terminal/CI variables can still override .env when needed.
    """
    global _ENV_LOADED

    project_root = find_project_root()
    env_path = project_root / ".env"

    if not _ENV_LOADED and env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        _ENV_LOADED = True

    return project_root


def get_environment_variable(
    name: str,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    load_environment()

    value = os.getenv(name, default)

    if required and (value is None or value == ""):
        raise RuntimeError(f"Missing required environment variable: {name}")

    return value


def get_boolean_environment_variable(name: str, default: bool = False) -> bool:
    load_environment()

    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()

    if normalized in {"1", "true", "yes", "y", "on"}:
        return True

    if normalized in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(
        f"Environment variable {name} must be boolean-like, got: {raw_value}"
    )


def load_yaml_file(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise ImportError(
            "PyYAML is required to load YAML files. Install it with: pip install pyyaml"
        )

    if not path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML file to contain a dictionary: {path}")

    return data
