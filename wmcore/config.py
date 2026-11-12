"""YAML reading and the root-relative path convention.

Every config file in this repo resolves its paths against a `root`, which is
itself relative to the file — so a config can be moved with its neighbours and
still point at the same places, and an absolute path always wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import yaml


def read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    return data


def root_of(config_path: Path, raw: dict[str, Any], default: str = "..") -> Path:
    """The directory `raw`'s relative paths are resolved against."""
    return (config_path.parent / raw.get("root", default)).resolve()


def resolver(root: Path) -> Callable[[str | Path], Path]:
    """A function turning a config value into a path under `root`."""

    def resolve(value: str | Path) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else (root / candidate).resolve()

    return resolve
