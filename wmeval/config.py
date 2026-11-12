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

from wmcore.arms import BASE, deep_merge, load_arm, run_manifest, run_subdir
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
    #: Level of understanding the task belongs to (Scene / Interaction / Robot
    #: State). Report-only: groups tasks in the summary, scores no differently.
    level: str = ""
    #: The one score that headlines this probe in the summary (last frame vs best frame).
    headline: str = ""
    #: Fidelity gates that screen the rollout before its score is trusted.
    gates: list[dict[str, Any]] = field(default_factory=list)
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
    #: The prompt arm and repeat being scored (wmcore/arms.py): the manifest is
    #: `<manifest stem>.<arm>.r<repeat>.yaml`, results go to `<output_dir>/<arm>/r<repeat>/`.
    arm: str = BASE
    repeat: int = 0


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
        level=raw.get("level", ""),
        headline=raw.get("headline", ""),
        gates=list(raw.get("gates") or []),
        source=path,
        meta=raw.get("meta") or {},
    )


def load_eval_config(path: str | Path, arm: str | None = None, repeat: int | None = None) -> EvalConfig:
    """The run config; `arm` and `repeat` override the YAML's (the CLI's --arm, --repeat)."""
    path = Path(path).resolve()
    raw = read_yaml(path)
    arm = arm if arm is not None else str(raw.get("arm", BASE))
    repeat = int(repeat if repeat is not None else raw.get("repeat", 0))
    root = root_of(path, raw)
    resolve = resolver(root)

    modules_dir = resolve(raw.get("modules_dir", "modules"))
    selection = raw.get("modules") or {}
    selected = discover(modules_dir, selection.get("include"), selection.get("exclude"))
    selected_names = {load_probe_config(d).name for d in selected}
    # A probe with no file for this arm is not part of the run; an arm may change
    # the decoder its prompt needs (`extractor:`).
    probes = []
    for module_dir in selected:
        overrides = load_arm(module_dir, arm)
        if overrides is None:
            continue
        probe = load_probe_config(module_dir)
        probe.extractor = deep_merge(probe.extractor, overrides.get("extractor") or {})
        probes.append(probe)
    if not probes:
        raise ValueError(f"{path}: no selected probe has an arm named {arm!r} (modules/<probe>/arms/{arm}.yaml)")

    # A dataset can know better than a probe's default: `ground_truth_overrides`
    # swaps one probe's source for this run, leaving its probe.yaml alone.
    overrides = dict(raw.get("ground_truth_overrides") or {})
    for probe in probes:
        if probe.name in overrides:
            probe.ground_truth = overrides.pop(probe.name)
    overrides = {k: v for k, v in overrides.items() if k not in selected_names}  # left out by the arm
    if overrides:
        raise ValueError(f"{path}: ground_truth_overrides names unselected probes: {sorted(overrides)}")

    return EvalConfig(
        manifest=run_manifest(resolve(raw["manifest"]), arm, repeat),
        output_dir=resolve(raw.get("output_dir", "results")) / run_subdir(arm, repeat),
        gt_dir=resolve(raw.get("gt_dir", "results")),
        modules_dir=modules_dir,
        probes=probes,
        limit=raw.get("limit"),
        fail_fast=bool(raw.get("fail_fast", False)),
        root=root,
        arm=arm,
        repeat=repeat,
    )
