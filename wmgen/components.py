"""Base classes for the four pluggable families.

Each mirrors one of wmeval's. A `FrameSource` is where samples come from, a
`PromptBuilder` decides what to ask, a `VideoBackend` is the world model, and a
`ClipCheck` is the sanity test on what came back.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from wmcore.components import Component
from wmcore.utils import safe_name

from .types import Clip, Frame, GenContext, Prompt, Request

log = logging.getLogger("wmgen")


class ClipUnavailable(Exception):
    """This backend has no clip for this sample, and that is not a failure.

    Raised by a backend adopting files that were generated elsewhere, where a
    frame simply has no video yet. The runner records `skipped` rather than
    `error`, the same distinction wmeval draws for a sample whose video is
    missing from the manifest.
    """


class FrameSource(Component):
    """Where conditioning frames come from.

    Yields every frame the run will generate for. The sample ids it produces are
    the ids the whole pipeline uses, including wmeval's, so a source is also
    deciding how results will be named.
    """

    def __call__(self, ctx: GenContext) -> Iterable[Frame]:
        raise NotImplementedError


class PromptBuilder(Component):
    """A probe's `prompt.yaml` plus one frame -> the request text.

    This is the half of a probe that *asks* for the convention the probe's
    extractor decodes. The two are written together and live in the same
    directory for that reason: a colour convention changed here and not there
    scores zero for reasons that look like a model failure.
    """

    def __call__(self, frame: Frame, video: dict) -> Prompt:
        raise NotImplementedError


class VideoBackend(Component):
    """A world model, behind submit -> wait -> fetch.

    Video generation is a job, not a call: most APIs return an operation id and
    are polled for minutes. The three methods are separate so a backend that is
    synchronous (a local model, an already-generated file) can make `wait` and
    `fetch` trivial while an HTTP one uses all three.

    The runner is serial and does not resume. Interrupting a run therefore
    abandons whatever job is generating on the server at that moment — it is
    paid for and it is lost to this pipeline. If that happens and the vendor
    delivers the file anyway, adopt it with the `local_dir` backend rather than
    generating it again.
    """

    #: Reported into every clip's sidecar, so a run's output says what made it.
    model_id: str = ""

    def submit(self, request: Request) -> str:
        """Start generating. Returns an opaque job id."""
        raise NotImplementedError

    def wait(self, job_id: str, request: Request) -> dict:
        """Block until the job finishes. Returns the backend's result record."""
        raise NotImplementedError

    def fetch(self, result: dict, request: Request) -> Path:
        """Write the finished video to `request.dest` and return that path."""
        raise NotImplementedError

    def describe(self) -> dict:
        """What went into the sidecar about this backend."""
        return {
            "backend": getattr(self, "registry_name", type(self).__name__),
            "model_id": self.model_id,
        }


class ClipCheck(Component):
    """A returned clip -> named diagnostics, and a verdict.

    The `Metric` analogue: a check declares what it looks at, returns numbers
    that land in the run's report, and raises when the clip is unusable. Raising
    is how a 0-byte file becomes `status="error"` instead of a manifest entry
    that fails much later inside wmeval.
    """

    def __call__(self, path: Path, request: Request) -> dict:
        raise NotImplementedError


def clip_for(clips_dir: Path, probe: str, sample_id: str, ext: str = ".mp4") -> Path:
    """Where a clip lives: `<clips_dir>/<probe>/<sample_id>.mp4`.

    This is exactly the layout `wmeval.manifest.scan_run_dir` expects, so a run
    directory built here is one a hand-written manifest scan also understands.
    """
    return Path(clips_dir) / probe / f"{safe_name(sample_id)}{ext}"


__all__ = [
    "Clip",
    "ClipCheck",
    "Component",
    "FrameSource",
    "PromptBuilder",
    "VideoBackend",
    "clip_for",
]
