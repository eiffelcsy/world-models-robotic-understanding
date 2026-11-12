"""Prompt arms and repeats: one probe asked several ways, each kept apart.

A prompt ablation asks the same probe on the same frames with different prompts
(arms), and asks each prompt more than once (repeats). Both pipelines must keep
every (arm, repeat) apart, or a second arm silently reuses the first arm's clips
and scores:

    <clips_dir>/<probe>/<arm>/r<repeat>/<sample_id>.mp4   one folder per arm and repeat
    <output_dir>/<arm>/r<repeat>/                          its gen and eval reports
    <manifest stem>.<arm>.r<repeat>.yaml                   its manifest

An arm is a file in the probe's own directory, `modules/<probe>/arms/<arm>.yaml`,
holding only what differs from the probe's defaults:

    description: E1, fully saturated and a different color per object
    prompt:      {params: {text: ...}}   # merged over prompt.yaml's `prompt`
    video:       {...}                    # merged over prompt.yaml's `video`
    extractor:   {params: {...}}          # merged over probe.yaml's `extractor`

`base` is the probe as its prompt.yaml and probe.yaml stand, and needs no file.
A run names one arm; a probe without that arm is left out of the run.

A repeat is the same request asked again with a different seed. The seed is
`seed_base + repeat`, the same for every arm, so arm A's repeat 0 and arm B's
repeat 0 share a seed: arms differ only in their prompt (common random numbers).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config import read_yaml

BASE = "base"
ARMS_DIR = "arms"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`override` over `base`, recursing into dicts; lists and scalars are replaced."""
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def arm_path(module_dir: Path, arm: str) -> Path:
    return Path(module_dir) / ARMS_DIR / f"{arm}.yaml"


def load_arm(module_dir: Path, arm: str) -> dict[str, Any] | None:
    """The arm's overrides; {} for `base`; None when this probe has no such arm."""
    if arm == BASE:
        return {}
    path = arm_path(module_dir, arm)
    if not path.exists():
        return None
    raw = read_yaml(path) or {}
    unknown = set(raw) - {"description", "prompt", "video", "extractor"}
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; an arm may set prompt, video, extractor")
    return raw


def list_arms(module_dir: Path) -> list[str]:
    folder = Path(module_dir) / ARMS_DIR
    return [BASE] + sorted(p.stem for p in folder.glob("*.yaml")) if folder.is_dir() else [BASE]


def run_subdir(arm: str, repeat: int) -> Path:
    """`<arm>/r<repeat>`: where one arm's repeat lives under a clips or output dir."""
    if not arm or "/" in arm or arm.startswith("."):
        raise ValueError(f"bad arm name {arm!r}")
    return Path(arm) / f"r{int(repeat)}"


def run_manifest(path: Path, arm: str, repeat: int) -> Path:
    """`manifest.libero.yaml` -> `manifest.libero.<arm>.r<repeat>.yaml`."""
    path = Path(path)
    return path.with_name(f"{path.stem}.{arm}.r{int(repeat)}{path.suffix}")
