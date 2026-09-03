"""Base classes for the five pluggable families.

Each mirrors one of wmeval's. A `FrameSource` is where samples come from, a
`SlotSource` says what a frame's task is about, a `PromptBuilder` decides what to
ask, a `VideoBackend` is the world model, and a `ClipCheck` is the sanity test on
what came back.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from wmcore.components import Component
from wmcore.utils import dump_json, load_json, safe_name

from .types import Clip, Frame, GenContext, Prompt, Request

log = logging.getLogger("wmgen")


class ClipUnavailable(Exception):
    """This backend has no clip for this sample, and that is not a failure.

    Raised by a backend adopting files that were generated elsewhere, where a
    frame simply has no video yet. The runner records `skipped` rather than
    `error`, the same distinction wmeval draws for a sample whose video is
    missing from the manifest.
    """


class SlotsUnavailable(Exception):
    """This frame's task cannot be resolved into objects, and that is terminal
    for the frame but not for the run."""


class FrameSource(Component):
    """Where conditioning frames come from.

    Yields every frame the run will generate for. The sample ids it produces are
    the ids the whole pipeline uses, including wmeval's, so a source is also
    deciding how results will be named.
    """

    def __call__(self, ctx: GenContext) -> Iterable[Frame]:
        raise NotImplementedError


class SlotSource(Component):
    """One frame's instruction -> the objects it involves, cached on disk.

    The caching contract is `wmeval.components.GroundTruthSource`'s, for the same
    reason: one file per sample, read when it is there, written when it is not,
    with nothing about the params in the filename. Regenerating is deleting the
    file. Subclasses implement `compute`; everything else is here.

    What `to_meta` writes into `frame.meta` — and so, via `build_manifest`, into
    the manifest wmeval reads:

      relevant_objects  "the marker and the mug", what a prompt template renders
      gt_prompt         ["marker", "mug"], what SAM 3 segments for ground truth
      action, slots_confidence, task_objects   carried for slicing and debugging

    `gt_prompt` is the reason this stage is worth its own file. A probe that
    scopes its prompt to the task but leaves ground truth scoping everything
    scores the model against objects it was explicitly told not to paint.
    `extra_gt_phrases` adds to that list only — put "robot arm" there for a probe
    whose prompt does ask for the arm, and leave it empty for one that does not.
    """

    def compute(self, frame: Frame, ctx: GenContext) -> dict[str, Any]:
        """The slow path: work out this frame's record. Raise to give up on it."""
        raise NotImplementedError

    def preflight(self) -> None:
        """Called once before any frame, when at least one will need `compute`.

        Where a source validates what would otherwise fail identically on every
        sample — an unset API key is a misconfigured run, not a thousand
        individual failures.
        """

    #: What a frame needs before this stage has nothing left to do for it. Both,
    #: never one: a meta_file that scopes the prompt but not the ground truth is
    #: the failure this stage exists to prevent, so half an answer is treated as
    #: no answer rather than silently left half-applied.
    provides = ("relevant_objects", "gt_prompt")

    def satisfied(self, frame: Frame) -> bool:
        """Whether this frame already has slots, from a meta_file or an earlier stage."""
        return all(frame.meta.get(key) for key in self.provides)

    def store_path(self, frame: Frame, ctx: GenContext) -> Path | None:
        if ctx.semantics_dir is None:
            return None
        return Path(ctx.semantics_dir) / f"{safe_name(frame.sample_id)}.json"

    def resolve(self, frame: Frame, ctx: GenContext, overrides: dict[str, dict] | None = None) -> dict[str, Any]:
        """The record for this frame: stored, else computed and stored, then patched."""
        record: dict[str, Any] | None = None
        path = self.store_path(frame, ctx)

        if path is not None and path.exists():
            try:
                record = load_json(path)
            except Exception:  # noqa: BLE001 - a damaged file must not be fatal
                log.warning("unreadable slots %s; recomputing", path)

        if record is None:
            patch = (overrides or {}).get(frame.sample_id)
            if patch and patch.get("target"):
                # A hand-written override that names the target outright is a
                # complete answer; there is nothing to ask an endpoint for.
                record = {"action": "unclear", "target": None, "destination": None, "confidence": None}
            else:
                record = self.compute(frame, ctx)
                record["sample_id"] = frame.sample_id
                if path is not None:
                    dump_json(path, record)

        patch = (overrides or {}).get(frame.sample_id)
        if patch:
            record = {**record, **patch}
            log.info("  slots overridden for %s", frame.sample_id)
        return record

    def to_meta(self, record: dict[str, Any]) -> dict[str, Any]:
        from .semantics import object_phrases, phrase_list  # local: module-level cycle

        phrases = object_phrases(record)
        if not phrases:
            raise SlotsUnavailable("the record names no object")
        return {
            "relevant_objects": phrase_list(phrases),
            "gt_prompt": [*phrases, *getattr(self, "extra_gt_phrases", [])],
            "task_objects": phrases,
            "action": record.get("action"),
            "slots_confidence": record.get("confidence"),
        }


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
    "ClipUnavailable",
    "Component",
    "FrameSource",
    "PromptBuilder",
    "SlotSource",
    "SlotsUnavailable",
    "VideoBackend",
    "clip_for",
]
