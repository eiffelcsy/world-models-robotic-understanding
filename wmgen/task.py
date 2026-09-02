"""A task = prompt builder + video params + checks, for one probe.

`Task.generate` is the whole per-frame pipeline, in the order the stages were
specified:

  1. the prompt, rendered for this frame
  2. the clip, from the backend — unless one is already stored
  3. the checks, on what came back
  4. the sidecar, recording what produced it

The mirror of `wmeval.probe.Probe.evaluate`, and it borrows two of that
function's decisions. Failures are returned as `Clip(status="error")` rather
than raised, so one refused generation does not end a 200-clip run. And a clip
already in the store is not made again, because this is the expensive stage —
more so than wmeval's, since it is expensive in money rather than minutes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from wmcore import registry

from .components import ClipUnavailable, VideoBackend, clip_for
from .config import PromptConfig
from .store import sidecar_path, warn_on_drift, write_sidecar
from .types import Clip, Frame, GenContext, Request

log = logging.getLogger("wmgen")


@dataclass
class Task:
    name: str
    prompt: Any
    checks: list[Any] = field(default_factory=list)
    video: dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @classmethod
    def from_config(cls, config: PromptConfig, video: dict[str, Any]) -> "Task":
        return cls(
            name=config.name,
            prompt=registry.build("prompt", config.prompt),
            checks=[registry.build("check", spec) for spec in config.checks],
            video=video,
            description=config.description,
        )

    def generate(self, frame: Frame, backend: VideoBackend, ctx: GenContext) -> Clip:
        dest = clip_for(ctx.clips_dir, self.name, frame.sample_id)
        info: dict[str, Any] = {}

        try:
            prompt = self.prompt(frame, self.video)
            request = Request(frame.sample_id, self.name, frame, prompt, dest)
            info["prompt_sha"] = prompt.sha

            cached = dest.exists() and not ctx.refresh
            if cached:
                info["drifted"] = warn_on_drift(dest, request)

            if ctx.dry_run:
                info["prompt"] = prompt.text
                info["video"] = prompt.video
                return Clip(frame.sample_id, self.name, "cached" if cached else "skipped", dest if cached else None, info)

            if cached:
                try:
                    info.update(self._check(dest, request))
                    return Clip(frame.sample_id, self.name, "cached", dest, info)
                except Exception as exc:  # noqa: BLE001
                    # A stored clip that no longer passes — a truncated download,
                    # a file that lost its tail — must not be an error the store
                    # repeats forever. Drop it and make it again, once.
                    log.warning("  stored clip failed its checks (%s); regenerating %s", exc, dest)
                    dest.unlink(missing_ok=True)
                    sidecar_path(dest).unlink(missing_ok=True)
                    info["regenerated"] = True

            started = time.time()
            job_id = backend.submit(request)
            info["job_id"] = job_id
            result = backend.wait(job_id, request)
            path = backend.fetch(result, request)
            info["gen_seconds"] = round(time.time() - started, 1)

            try:
                info.update(self._check(path, request))
            except Exception:
                # A clip that fails its checks must not stay in the store: it
                # would be reported `cached` on the next run and never made
                # again, which is the one way a bad file becomes permanent.
                path.unlink(missing_ok=True)
                raise

            write_sidecar(path, request, {**backend.describe(), **info})
            return Clip(frame.sample_id, self.name, "ok", path, info)

        except ClipUnavailable as exc:
            return Clip(frame.sample_id, self.name, "skipped", info=info, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - one clip must not end the run
            return Clip(frame.sample_id, self.name, "error", info=info, error=f"{type(exc).__name__}: {exc}")

    def _check(self, path, request: Request) -> dict[str, Any]:
        info: dict[str, Any] = {}
        for check in self.checks:
            info.update(check(path, request))
        return info
