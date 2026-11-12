"""Frame sources: where the conditioning frames come from.

wmeval reads its samples from a manifest because by then the videos exist. wmgen
has no manifest yet — producing one is its output — so it starts from frames on
disk instead.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable

from wmcore.config import read_yaml, resolver, root_of
from wmcore.registry import register

from .components import FrameSource
from .types import Frame, GenContext

log = logging.getLogger("wmgen")


@register("source", "frames_dir")
class FramesDir(FrameSource):
    """Every image in a directory, one frame per file.

    The sample id is the file stem, which is what wmeval's manifest and both
    stores key on. `meta_file` optionally attaches per-sample metadata — a task
    string, a camera, a per-sample ground-truth prompt — keyed by sample id:

        meta:
          AUTOLab_..._f000021: {task: "put the red block in the bowl", camera: ext1}
    """

    def __init__(
        self,
        path: str,
        pattern: str = "*.png",
        meta_file: str | None = None,
        recursive: bool = False,
        include: str | None = None,
    ) -> None:
        super().__init__(path=path, pattern=pattern, meta_file=meta_file, recursive=recursive, include=include)

    def __call__(self, ctx: GenContext) -> Iterable[Frame]:
        root = Path(self.path)
        if not root.exists():
            raise FileNotFoundError(f"frames directory not found: {root}")
        glob = root.rglob if self.recursive else root.glob
        paths = sorted(glob(self.pattern))
        if not paths:
            raise ValueError(f"no frames matching '{self.pattern}' under {root}")

        # A frozen subset: restrict to the ids named in `include`, in that order,
        # rather than relying on `limit` + enumeration order, which differs
        # between filesystems and so is not reproducible across machines.
        if self.include:
            wanted = _read_ids(self.include)
            by_stem = {p.stem: p for p in paths}
            missing = [sid for sid in wanted if sid not in by_stem]
            if missing:
                raise ValueError(
                    f"include list {self.include} names {len(missing)} frame(s) not under {root}: "
                    f"{', '.join(missing[:5])}"
                )
            paths = [by_stem[sid] for sid in wanted]

        meta: dict[str, Any] = {}
        if self.meta_file:
            meta = read_yaml(Path(self.meta_file)).get("meta") or {}

        return [Frame(sample_id=p.stem, path=p.resolve(), meta=dict(meta.get(p.stem) or {})) for p in paths]


def _read_ids(path: str) -> list[str]:
    """One sample id per line; blank lines and `#` comments are ignored."""
    file = Path(path)
    if not file.exists():
        raise FileNotFoundError(f"include list not found: {path}")
    ids = [line.split("#", 1)[0].strip() for line in file.read_text(encoding="utf-8").splitlines()]
    return [line for line in ids if line]


@register("source", "frame_list")
class FrameList(FrameSource):
    """An explicit list of frames, for a hand-picked set.

        frames:
          type: frame_list
          params:
            path: configs/frames.yaml

        # configs/frames.yaml
        root: ..
        frames:
          - sample_id: AUTOLab_..._f000021
            path: ../droid/frames/AUTOLab_..._f000021.png
            meta: {task: "put the red block in the bowl"}

    Use it when the frames are scattered, or when a run should be a fixed subset
    that does not change as a directory fills up.
    """

    def __init__(self, path: str) -> None:
        super().__init__(path=path)

    def __call__(self, ctx: GenContext) -> Iterable[Frame]:
        path = Path(self.path).resolve()
        raw = read_yaml(path)
        resolve = resolver(root_of(path, raw, default="."))

        frames = []
        for entry in raw.get("frames") or []:
            frame_path = resolve(entry["path"])
            frames.append(
                Frame(
                    sample_id=str(entry.get("sample_id") or frame_path.stem),
                    path=frame_path,
                    meta=entry.get("meta") or {},
                )
            )
        if not frames:
            raise ValueError(f"{path}: lists no frames")
        return frames
