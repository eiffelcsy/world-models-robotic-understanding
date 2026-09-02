"""Plugin directory discovery.

Both pipelines extend the same way: one directory per plugin, picked up as soon
as it contains a marker file and ignored as soon as it does not.

    modules/<probe_name>/probe.yaml     a probe, for wmeval
    modules/<probe_name>/prompt.yaml    what to ask a world model for, for wmgen
    backends/<model_name>/backend.yaml  how to call a world model, for wmgen

The marker is the caller's, so the same discovery serves all three. A directory
holding only `probe.yaml` is eval-only; only `prompt.yaml`, generation-only.

Importing a plugin is what puts its components in the registry, so it happens
before anything is built from a name.
"""

from __future__ import annotations

import fnmatch
import importlib
import sys
from pathlib import Path


def discover(
    root: Path,
    marker: str,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[Path]:
    """Subdirectories of `root` containing `marker`, filtered by name globs."""
    if not root.exists():
        raise FileNotFoundError(f"plugin directory not found: {root}")

    include = include or ["*"]
    exclude = exclude or []
    found = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        if path.name.startswith((".", "_")) or not (path / marker).exists():
            continue
        if not any(fnmatch.fnmatch(path.name, pattern) for pattern in include):
            continue
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in exclude):
            continue
        found.append(path)
    return found


def import_module(root: Path, name: str) -> None:
    """Import `<root>/<name>` as a package, registering its components.

    The parent of `root` goes on `sys.path` so the plugin is imported by package
    name rather than by file — relative imports inside it then work, and a plugin
    is free to span several files.

    A plugin directory with no `__init__.py` has nothing to register (a backend
    that is pure YAML is the normal case), so a missing package is not an error.
    """
    parent = str(root.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    if not (root / name / "__init__.py").exists():
        return
    importlib.import_module(f"{root.name}.{name}")
