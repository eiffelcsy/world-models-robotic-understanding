"""YAML config loading.

Three kinds of file describe a generation run:

  configs/gen.yaml            the run: frames, backend, output, which modules
  modules/<name>/prompt.yaml  one probe: what to ask a world model for
  backends/<name>/backend.yaml  one world model: how to call it

A probe's prompt lives inside its module directory beside `probe.yaml`, so a
module stays self-contained: what to ask and how to decode the answer are one
directory and one commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from wmcore.config import read_yaml, resolver, root_of
from wmcore.modules import discover

PROMPT_FILE = "prompt.yaml"
BACKEND_FILE = "backend.yaml"


@dataclass
class PromptConfig:
    """The generation half of a probe."""

    name: str
    module: str
    prompt: dict[str, Any]
    video: dict[str, Any] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    enabled: bool = True
    description: str = ""
    source: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendConfig:
    """One world model."""

    name: str
    spec: dict[str, Any]  # {type, params} for registry.build
    defaults: dict[str, Any] = field(default_factory=dict)
    source: Path | None = None


@dataclass
class GenConfig:
    frames: dict[str, Any]
    backend: BackendConfig
    prompts: list[PromptConfig]
    clips_dir: Path
    output_dir: Path
    modules_dir: Path
    backends_dir: Path
    manifest_out: Path
    video: dict[str, Any] = field(default_factory=dict)
    limit: int | None = None
    fail_fast: bool = False
    dry_run: bool = False
    refresh: bool = False
    root: Path = Path(".")


def load_prompt_config(module_dir: Path) -> PromptConfig:
    path = module_dir / PROMPT_FILE
    raw = read_yaml(path)
    if "prompt" not in raw:
        raise ValueError(f"{path}: prompt config is missing 'prompt'")
    return PromptConfig(
        name=raw.get("name", module_dir.name),
        module=module_dir.name,
        prompt=raw["prompt"],
        video=raw.get("video") or {},
        checks=list(raw.get("checks") or []),
        enabled=bool(raw.get("enabled", True)),
        description=raw.get("description", ""),
        source=path,
        meta=raw.get("meta") or {},
    )


#: Backend params naming a place on disk. They are resolved against the run's
#: `root` like every other path in the repo, so what they mean does not depend
#: on which directory the pipeline was launched from.
_PATH_PARAMS = ("root", "cwd")


def load_backend_config(backends_dir: Path, name: str, resolve: Callable[[str], Path] | None = None) -> BackendConfig:
    path = backends_dir / name / BACKEND_FILE
    if not path.exists():
        known = ", ".join(p.name for p in discover(backends_dir, BACKEND_FILE)) or "<none>"
        raise FileNotFoundError(f"no backend '{name}' in {backends_dir}. available: {known}")
    raw = read_yaml(path)
    if "type" not in raw:
        raise ValueError(f"{path}: backend config is missing 'type'")

    params = dict(raw.get("params") or {})
    if resolve is not None:
        for key in _PATH_PARAMS:
            if isinstance(params.get(key), str):
                params[key] = str(resolve(params[key]))

    return BackendConfig(
        name=raw.get("name", name),
        spec={"type": raw["type"], "params": params},
        defaults=raw.get("defaults") or {},
        source=path,
    )


def load_gen_config(path: str | Path) -> GenConfig:
    path = Path(path).resolve()
    raw = read_yaml(path)
    root = root_of(path, raw)
    resolve = resolver(root)

    modules_dir = resolve(raw.get("modules_dir", "modules"))
    backends_dir = resolve(raw.get("backends_dir", "backends"))
    selection = raw.get("modules") or {}
    prompts = [
        load_prompt_config(module_dir)
        for module_dir in discover(modules_dir, PROMPT_FILE, selection.get("include"), selection.get("exclude"))
    ]

    if "frames" not in raw:
        raise ValueError(f"{path}: no `frames` block; it says where the conditioning frames come from")
    if "backend" not in raw:
        raise ValueError(f"{path}: no `backend` named; it selects a directory under {backends_dir}")

    return GenConfig(
        frames=raw["frames"],
        backend=load_backend_config(backends_dir, raw["backend"], resolve),
        prompts=prompts,
        clips_dir=resolve(raw.get("clips_dir", "clips")),
        output_dir=resolve(raw.get("output_dir", "results/gen")),
        modules_dir=modules_dir,
        backends_dir=backends_dir,
        manifest_out=resolve(raw.get("manifest_out", "configs/manifest.yaml")),
        video=raw.get("video") or {},
        limit=raw.get("limit"),
        fail_fast=bool(raw.get("fail_fast", False)),
        dry_run=bool(raw.get("dry_run", False)),
        refresh=bool(raw.get("refresh", False)),
        root=root,
    )


def video_params(config: GenConfig, prompt: PromptConfig) -> dict[str, Any]:
    """What to ask for, lowest precedence first.

    A backend's defaults are what that model supports; a probe overrides what
    its prompt depends on (a duration its text names); the run config overrides
    everything, so a smoke run can shorten every clip without editing a probe.
    """
    return {**config.backend.defaults, **prompt.video, **config.video}
