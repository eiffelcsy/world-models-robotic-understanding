"""YAML config loading.

Two kinds of file describe a run:

  configs/eval.yaml          the run: manifest, output dir, which modules
  modules/<name>/probe.yaml  one probe: readout, ground truth, extractor, metrics

A probe's config lives inside its module directory, so a module is self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wmcore.config import read_yaml, resolver, root_of

from .modules import PROBE_FILE, discover


@dataclass
class ProbeConfig:
    name: str
    module: str
    readout: dict[str, Any]
    ground_truth: dict[str, Any]
    extractor: dict[str, Any]
    metrics: list[dict[str, Any]]
    enabled: bool = True
    description: str = ""
    best_frame: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalConfig:
    manifest: Path
    output_dir: Path
    gt_dir: Path
    modules_dir: Path
    probes: list[ProbeConfig]
    limit: int | None = None
    fail_fast: bool = False
    root: Path = Path(".")


def load_probe_config(module_dir: Path) -> ProbeConfig:
    path = module_dir / PROBE_FILE
    raw = read_yaml(path)
    missing = [k for k in ("ground_truth", "extractor", "metrics") if k not in raw]
    if missing:
        raise ValueError(f"{path}: probe config is missing {missing}")
    return ProbeConfig(
        name=raw.get("name", module_dir.name),
        module=module_dir.name,
        readout=raw.get("readout") or {"type": "last_frame"},
        ground_truth=raw["ground_truth"],
        extractor=raw["extractor"],
        metrics=list(raw["metrics"]),
        enabled=bool(raw.get("enabled", True)),
        description=raw.get("description", ""),
        best_frame=raw.get("best_frame") or {},
        source=path,
        meta=raw.get("meta") or {},
    )


def load_eval_config(path: str | Path) -> EvalConfig:
    path = Path(path).resolve()
    raw = read_yaml(path)
    root = root_of(path, raw)
    resolve = resolver(root)

    modules_dir = resolve(raw.get("modules_dir", "modules"))
    selection = raw.get("modules") or {}
    probes = [
        load_probe_config(module_dir)
        for module_dir in discover(modules_dir, selection.get("include"), selection.get("exclude"))
    ]

    return EvalConfig(
        manifest=resolve(raw["manifest"]),
        output_dir=resolve(raw.get("output_dir", "results")),
        gt_dir=resolve(raw.get("gt_dir", "results")),
        modules_dir=modules_dir,
        probes=probes,
        limit=raw.get("limit"),
        fail_fast=bool(raw.get("fail_fast", False)),
        root=root,
    )
