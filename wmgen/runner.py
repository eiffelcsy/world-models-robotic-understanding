"""Generate a clip for every enabled probe on every frame, and write the manifest.

Serial, and it does not resume. One clip is made at a time, in the order the
frames came out of the source — after `limit`/`limit_by` have already cut that
list down, see `wmcore.utils.stratified_sample` — exactly as `wmeval.runner`
scores them. Nothing here tracks a job across runs: interrupting a run abandons
whatever the backend is generating at that moment. If a vendor delivers that
clip anyway, adopt it with the `local_dir` backend rather than paying for it
twice.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from wmcore import registry
from wmcore.arms import run_subdir
from wmcore.modules import import_module
from wmcore.utils import ensure_dir, stratified_sample

from . import backends, checks, prompt, semantics, sources  # noqa: F401  (registration)
from .config import BACKEND_FILE, GenConfig, load_gen_config, video_params
from .report import write_prompts, write_report
from .task import Task
from .types import Clip, Frame, GenContext

log = logging.getLogger("wmgen")


def run(config: GenConfig) -> list[Clip]:
    enabled = [c for c in config.prompts if c.enabled]
    if not enabled:
        raise ValueError(f"no enabled prompts found in {config.modules_dir} (looking for prompt.yaml)")

    # Importing a plugin registers its components, so it happens before anything
    # is built from a name. A probe module carries the eval half's imports too,
    # which wmgen does not need and should not require installed — so a module
    # that will not import is a warning here, and only an error if a prompt type
    # it was supposed to register is then missing.
    for prompt_config in enabled:
        try:
            import_module(config.modules_dir, prompt_config.module)
        except ImportError as exc:
            log.warning("could not import module %s (%s); its YAML is still used", prompt_config.module, exc)
    if (config.backends_dir / config.backend.name / BACKEND_FILE).exists():
        import_module(config.backends_dir, config.backend.name)

    backend = registry.build("backend", config.backend.spec)
    tasks = [Task.from_config(c, video_params(config, c)) for c in enabled]

    ctx = GenContext(
        run_dir=ensure_dir(config.output_dir),
        clips_dir=ensure_dir(config.clips_dir),
        semantics_dir=ensure_dir(config.semantics_dir) if config.semantics else None,
        refresh=config.refresh,
        dry_run=config.dry_run,
        clip_subdir=run_subdir(config.arm, config.repeat),
    )

    frames: list[Frame] = list(registry.build("source", config.frames)(ctx))
    if config.limit:
        if config.limit_by:
            frames = stratified_sample(frames, config.limit, lambda f: f.meta.get(config.limit_by), config.limit_seed)
        else:
            frames = frames[: config.limit]  # whatever order the source yields — sorted-glob, for frames_dir

    # Instruction -> objects, before any prompt is rendered. Cached per sample,
    # so this is free on a rerun; a frame it cannot resolve keeps its place and
    # fails on `requires:` at render time, which is before the backend is called.
    slot_failures = semantics.apply(frames, config.semantics, ctx)
    if slot_failures and config.fail_fast:
        first = slot_failures[0]
        raise RuntimeError(f"semantics/{first['sample_id']}: {first['error']}")

    log.info("arm %s, repeat %d, seed %s", config.arm, config.repeat,
             "none" if config.seed_base is None else config.seed_base + config.repeat)
    log.info(
        "%d frames x %d probes = %d clips%s via %s",
        len(frames),
        len(tasks),
        len(frames) * len(tasks),
        " (dry run: no backend calls)" if config.dry_run else "",
        config.backend.name,
    )

    started = time.time()
    clips: list[Clip] = []
    for frame in frames:
        for task in tasks:
            clip = task.generate(frame, backend, ctx)
            clips.append(clip)
            if clip.status in ("ok", "cached") or config.dry_run:
                log.info("  %-22s %-40s %s", task.name, frame.sample_id, _fmt(clip, config.dry_run))
            else:
                log.warning("  %-22s %-40s %s: %s", task.name, frame.sample_id, clip.status, clip.error)
                if config.fail_fast and clip.status == "error":
                    raise RuntimeError(f"{task.name}/{frame.sample_id}: {clip.error}")

    write_report(
        clips,
        frames,
        config.output_dir,
        config.manifest_out,
        probes=[t.name for t in tasks],
        backend=config.backend.name,
    )
    if config.dry_run:
        log.info("prompts -> %s", write_prompts(clips, config.output_dir))
    log.info("done in %.1fs -> %s", time.time() - started, config.output_dir)
    log.info("manifest -> %s", config.manifest_out)
    return clips


def run_from_file(path: str | Path) -> list[Clip]:
    return run(load_gen_config(path))


def _fmt(clip: Clip, dry_run: bool = False) -> str:
    if dry_run:
        words = len(clip.info.get("prompt", "").split())
        return f"would generate  prompt {clip.info.get('prompt_sha', '?')} ({words} words)"
    if clip.status == "cached":
        return "cached" + (" (prompt drifted)" if clip.info.get("drifted") else "")
    parts = [f"{k}={v}" for k, v in clip.info.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return "  ".join(parts) or "ok"
