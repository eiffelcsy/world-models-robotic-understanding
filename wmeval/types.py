"""Core data types passed between pipeline stages.

The whole pipeline is four typed hops:

    Sample ──gt source──▶ ProbeData (pseudo ground truth)
    Sample ──readout──▶ output frame ──extractor──▶ ProbeData (prediction)
    (prediction, ground truth) ──metrics──▶ Result

Everything a probe needs to know about a scene lives in `Sample`; everything a
metric needs to know about an answer lives in `ProbeData`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Sample:
    """One evaluation unit: a conditioning frame plus the videos generated from it.

    `videos` is keyed by probe name because a world model is usually run once per
    probe on the same frame. A single video shared by every probe is expressed by
    the reserved key `"*"`.
    """

    sample_id: str
    input_frame: Path
    videos: dict[str, Path] = field(default_factory=dict)
    gt_assets: dict[str, Path] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def video_for(self, probe: str) -> Path | None:
        return self.videos.get(probe) or self.videos.get("*")


@dataclass
class ProbeData:
    """A probe answer, from either the world model or the pseudo ground truth.

    `kind` is the contract between an extractor and a metric — a metric declares
    which kinds it accepts and the runner fails loudly on a mismatch rather than
    silently scoring incomparable things.

    Conventions per kind:
      depth        payload["depth"]      float HxW, larger = further from camera
      binary_mask  payload["mask"]       bool HxW
      instances    payload["masks"]      list[bool HxW]
      points       payload["points"]     float (N, 2) array of (x, y) pixels
    """

    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        return self.payload[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.payload.get(key, default)

    @property
    def shape(self) -> tuple[int, int] | None:
        """Pixel grid this answer lives on, when it has one."""
        for key in ("depth", "mask"):
            if key in self.payload:
                return tuple(np.asarray(self.payload[key]).shape[:2])  # type: ignore[return-value]
        masks = self.payload.get("masks")
        if masks:
            return tuple(np.asarray(masks[0]).shape[:2])  # type: ignore[return-value]
        return self.info.get("frame_shape")


@dataclass
class Result:
    """Outcome of one (sample, probe) pair."""

    sample_id: str
    probe: str
    status: str  # ok | skipped | error
    scores: dict[str, float] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    #: Fidelity gates: `ok` rows that fail a gate are still scored but flagged
    #: invalid, and an invalid rate is reported beside every mean (see gates.py).
    valid: bool = True
    invalid_reason: str | None = None
    #: The sample's manifest `meta`, carried so a report can slice by scenario
    #: (RoboTrustBench's `category`) without re-reading the manifest.
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "probe": self.probe,
            "status": self.status,
            "scores": self.scores,
            "info": self.info,
            "error": self.error,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "meta": self.meta,
        }


@dataclass
class RunContext:
    """Shared per-run state handed to every component."""

    run_dir: Path
    gt_dir: Path | None = None  # root of the per-probe ground-truth stores; None = do not store
    refresh_gt: bool = False  # recompute stored ground truth instead of loading it
    extras: dict[str, Any] = field(default_factory=dict)
