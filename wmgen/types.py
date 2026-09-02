"""Core data types passed between generation stages.

The whole pipeline is four typed hops, the mirror of wmeval's:

    frames dir ──source──▶ Frame
    (Frame, prompt.yaml) ──prompt──▶ Prompt
    (Frame, Prompt) ──backend──▶ a video file on disk
    that file ──checks──▶ Clip

Everything wmgen knows about a scene lives in `Frame`; everything it knows about
what it asked for lives in `Prompt`; everything it knows about what came back
lives in `Clip`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Frame:
    """One conditioning frame: the thing a world model is asked to continue.

    `sample_id` is the frame's file stem, which is also how wmeval's manifest
    names the sample and how both stores name their files — so a frame, its
    generated clip and its ground truth are findable from each other by eye.

    `meta` is free-form and is carried all the way through to the manifest, so
    scores can later be sliced by task, lab or scene. It is also what a prompt
    template's placeholders are filled from.
    """

    sample_id: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prompt:
    """What to ask for, rendered for one frame.

    `sha` identifies the request that produced a clip. It covers the text, the
    negative prompt and the video params — everything that would make the model
    answer differently — so a stored clip can say whether it predates the prompt
    now configured.
    """

    text: str
    negative: str = ""
    video: dict[str, Any] = field(default_factory=dict)

    @property
    def sha(self) -> str:
        parts = [self.text, self.negative, repr(sorted(self.video.items()))]
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass
class Request:
    """One unit of work for a backend: this frame, this prompt, this destination."""

    sample_id: str
    probe: str
    frame: Frame
    prompt: Prompt
    dest: Path

    @property
    def video(self) -> dict[str, Any]:
        return self.prompt.video


@dataclass
class Clip:
    """Outcome of one (frame, probe) pair.

    `cached` is not a failure: it is a clip already on disk, which is the normal
    state of every rerun and the reason a rerun costs nothing.
    """

    sample_id: str
    probe: str
    status: str  # ok | cached | skipped | error
    path: Path | None = None
    info: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def usable(self) -> bool:
        """Whether this clip belongs in the manifest handed to wmeval."""
        return self.status in ("ok", "cached") and self.path is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "probe": self.probe,
            "status": self.status,
            "path": str(self.path) if self.path else None,
            "info": self.info,
            "error": self.error,
        }


@dataclass
class GenContext:
    """Shared per-run state handed to every component."""

    run_dir: Path
    clips_dir: Path
    refresh: bool = False  # regenerate clips that already exist
    dry_run: bool = False  # render prompts, call no backend
    extras: dict[str, Any] = field(default_factory=dict)
