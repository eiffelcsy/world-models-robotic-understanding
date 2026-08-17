"""Module discovery.

A probe lives in one directory under `modules/`:

    modules/<probe_name>/
        __init__.py     imports the module's own files so registration happens
        probe.yaml      readout + ground_truth + extractor + metrics
        ...             the model and whatever it needs

Dropping a directory in adds a probe; deleting it removes one. Nothing outside
the directory has to change.

Importing a module is what puts its components in the registry, so it happens
before any probe is built.
"""

from __future__ import annotations

import fnmatch
import importlib
import sys
from pathlib import Path

PROBE_FILE = "probe.yaml"


def discover(modules_dir: Path, include: list[str] | None = None, exclude: list[str] | None = None) -> list[Path]:
    """Module directories containing a `probe.yaml`, filtered by name globs."""
    if not modules_dir.exists():
        raise FileNotFoundError(f"modules directory not found: {modules_dir}")

    include = include or ["*"]
    exclude = exclude or []
    found = []
    for path in sorted(p for p in modules_dir.iterdir() if p.is_dir()):
        if path.name.startswith((".", "_")) or not (path / PROBE_FILE).exists():
            continue
        if not any(fnmatch.fnmatch(path.name, pattern) for pattern in include):
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in exclude):
            continue
        found.append(path)
    return found


def import_module(modules_dir: Path, name: str) -> None:
    """Import `<modules_dir>/<name>` as a package, registering its components.

    The parent of `modules_dir` goes on `sys.path` so the module is imported by
    package name rather than by file — relative imports inside it then work, and
    a module is free to span several files.
    """
    parent = str(modules_dir.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    importlib.import_module(f"{modules_dir.name}.{name}")
