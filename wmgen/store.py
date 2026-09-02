"""The clip store.

A generated clip is the expensive artefact of this pipeline the way SAM 3 output
is wmeval's, so the caching contract is deliberately the same one
(`wmeval.components.GroundTruthSource`):

    <clips_dir>/<probe>/<sample_id>.mp4    the video wmeval will score
    <clips_dir>/<probe>/<sample_id>.json   what was asked for to get it

A clip that is there is not generated again. Nothing about the prompt is in the
filename, so a stored clip survives an edit to `prompt.yaml` rather than being
orphaned by it — the sidecar records the prompt's hash and the run *warns*
naming the drift. That matters more here than it does in wmeval: regenerating
ground truth costs GPU minutes, regenerating a clip costs money.

Regenerating is deleting the files, or `--refresh`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from wmcore.utils import dump_json, load_json

from .types import Request

log = logging.getLogger("wmgen")


def sidecar_path(clip: Path) -> Path:
    return Path(clip).with_suffix(".json")


def write_sidecar(clip: Path, request: Request, record: dict[str, Any]) -> None:
    """Record what produced this clip, beside it and in plain text.

    The counterpart of the `.png` wmeval writes next to a stored `.npz`: not
    read back by the pipeline, there so a directory of clips can be understood
    six months later without the config that made it.
    """
    dump_json(
        sidecar_path(clip),
        {
            "sample_id": request.sample_id,
            "probe": request.probe,
            "input_frame": str(request.frame.path),
            "prompt": request.prompt.text,
            "negative": request.prompt.negative,
            "prompt_sha": request.prompt.sha,
            "video": request.prompt.video,
            "meta": request.frame.meta,
            **record,
        },
    )


def read_sidecar(clip: Path) -> dict[str, Any] | None:
    path = sidecar_path(clip)
    if not path.exists():
        return None
    try:
        return load_json(path)
    except Exception:  # noqa: BLE001 - a damaged sidecar must not be fatal
        log.warning("unreadable sidecar %s; treating the clip as unlabelled", path)
        return None


def warn_on_drift(clip: Path, request: Request) -> bool:
    """Say so when a stored clip predates the prompt now configured.

    Returns whether it drifted. The clip is used either way — silently throwing
    away a paid-for generation because a word changed in `prompt.yaml` is the
    worse failure, and the warning names what to do about it.
    """
    stored = read_sidecar(clip)
    if not stored:
        return False
    if stored.get("prompt_sha") == request.prompt.sha:
        return False
    log.warning(
        "%s was generated from a different prompt (%s, now %s); delete it or re-run "
        "with --refresh to regenerate",
        clip,
        stored.get("prompt_sha", "?"),
        request.prompt.sha,
    )
    return True
