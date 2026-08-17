"""Run every enabled probe over every sample and write the results."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from . import video  # noqa: F401  (imported for its readout registrations)
from .config import EvalConfig, load_eval_config
from .manifest import load_manifest
from .modules import import_module
from .probe import Probe
from .report import write_report
from .types import Result, RunContext
from .utils import dump_json, ensure_dir

log = logging.getLogger("wmeval")


def run(config: EvalConfig) -> list[Result]:
    enabled = [c for c in config.probes if c.enabled]
    if not enabled:
        raise ValueError(f"no enabled probes found in {config.modules_dir}")

    samples = load_manifest(config.manifest)
    if config.limit:
        samples = samples[: config.limit]

    # Importing a module registers its components; only then can a probe be built.
    for probe_config in enabled:
        import_module(config.modules_dir, probe_config.module)
    probes = [Probe.from_config(c) for c in enabled]

    ctx = RunContext(run_dir=ensure_dir(config.output_dir), gt_dir=ensure_dir(config.gt_dir))

    log.info("%d samples x %d probes = %d evaluations", len(samples), len(probes), len(samples) * len(probes))
    started = time.time()
    results: list[Result] = []
    for sample in samples:
        for probe in probes:
            result = probe.evaluate(sample, ctx)
            results.append(result)
            if result.status == "ok":
                log.info("  %-22s %-40s %s", probe.name, sample.sample_id, _fmt(result.scores))
            else:
                log.warning("  %-22s %-40s %s: %s", probe.name, sample.sample_id, result.status, result.error)
                if config.fail_fast and result.status == "error":
                    raise RuntimeError(f"{probe.name}/{sample.sample_id}: {result.error}")

    dump_json(config.output_dir / "results.json", [r.to_dict() for r in results])
    write_report(results, config.output_dir, probes=[p.name for p in probes])
    log.info("done in %.1fs -> %s", time.time() - started, config.output_dir)
    return results


def run_from_file(path: str | Path) -> list[Result]:
    return run(load_eval_config(path))


def _fmt(scores: dict[str, float]) -> str:
    return "  ".join(f"{k}={v:.3f}" for k, v in scores.items())
