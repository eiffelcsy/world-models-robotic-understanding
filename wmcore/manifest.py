"""The manifest: the interface between the two halves of the repo.

wmgen writes one; wmeval reads it. It is the only file either pipeline needs the
other to agree about, which is why the schema and its writer live here rather
than in whichever package happened to need them first:

    root: /                       what relative paths below resolve against
    samples:
      - sample_id: AUTOLab_2023-10-27__ext1_f000021
        input_frame: frames/AUTOLab_....png    the frame the model was conditioned on
        videos:
          instance_seg: clips/instance_seg/AUTOLab_....mp4
          "*": fallback_used_by_every_probe.mp4          # optional
        gt_assets:
          depth: gt/AUTOLab_..._depth.npy                # optional, probe-defined keys
        meta: {episode: ..., camera: ext1, frame_index: 21}

A hand-written manifest is equally valid, and is how a model whose output lives
in some other layout is evaluated without changing either pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> None:
    """Write a manifest, dropping any `_`-prefixed diagnostic keys."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: v for k, v in manifest.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, default_flow_style=False)
