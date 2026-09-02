"""Probe module discovery.

A probe lives in one directory under `modules/`:

    modules/<probe_name>/
        __init__.py     imports the module's own files so registration happens
        probe.yaml      readout + ground_truth + extractor + metrics
        prompt.yaml     optional; what wmgen asks a world model for
        ...             the model and whatever it needs

Dropping a directory in adds a probe; deleting it removes one. Nothing outside
the directory has to change.

The discovery itself is `wmcore.modules`, shared with wmgen — this module only
fixes the marker file a probe is recognised by.
"""

from __future__ import annotations

from pathlib import Path

from wmcore.modules import discover as _discover
from wmcore.modules import import_module  # noqa: F401  (re-exported)

PROBE_FILE = "probe.yaml"


def discover(modules_dir: Path, include: list[str] | None = None, exclude: list[str] | None = None) -> list[Path]:
    """Module directories containing a `probe.yaml`, filtered by name globs."""
    return _discover(modules_dir, PROBE_FILE, include, exclude)


__all__ = ["PROBE_FILE", "discover", "import_module"]
