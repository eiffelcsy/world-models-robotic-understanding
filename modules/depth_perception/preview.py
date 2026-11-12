"""Regenerate the depth ground truth and look at it.

    python -m modules.depth_perception.preview path/to/frames/

Every frame given is run through DA3 now, whatever is already on disk for it,
and the result replaces the stored one. Parameters come from `probe.yaml`
unless a flag overrides one, so a plain run reproduces what the pipeline
expects; `min_instances` is relaxed to 0 so an empty field is shown, not raised.
"""

from __future__ import annotations

import argparse
import glob as globlib
from pathlib import Path

from wmeval.config import load_probe_config
from wmeval.types import RunContext, Sample
from wmeval.utils import ensure_dir, read_image

from .gt import DepthAnything3Depth

SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _resolve_frames(patterns: list[Path]) -> list[Path]:
    found: list[Path] = []
    for pattern in patterns:
        if pattern.is_dir():
            found.extend(c for c in sorted(pattern.iterdir()) if c.suffix.lower() in SUFFIXES)
        elif pattern.exists():
            found.append(pattern)
        else:
            matched = [Path(m) for m in globlib.glob(str(pattern))]
            if not matched:
                raise FileNotFoundError(f"no frames matched: {pattern}")
            found.extend(sorted(matched))
    unique = sorted({p.resolve() for p in found})
    if not unique:
        raise FileNotFoundError(f"no frames found in: {[str(p) for p in patterns]}")
    return unique


def _params(args: argparse.Namespace) -> dict:
    params = dict(load_probe_config(Path(__file__).resolve().parent).ground_truth.get("params") or {})
    given = {"model_id": args.model_id, "device": args.device}
    params.update({k: v for k, v in given.items() if v is not None})
    params["min_instances"] = 0
    return params


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("frames", nargs="+", type=Path, help="frame image(s), a directory, or a glob")
    parser.add_argument("--model-id")
    parser.add_argument("--device")
    parser.add_argument("--gt-dir", type=Path, default=Path("results"))
    args = parser.parse_args()

    source = DepthAnything3Depth(**_params(args))
    gt_dir = ensure_dir(args.gt_dir)
    ctx = RunContext(run_dir=gt_dir, gt_dir=gt_dir, refresh_gt=True)

    for path in _resolve_frames(args.frames):
        frame = read_image(path)
        sample = Sample(sample_id=path.stem, input_frame=path)
        data = source(sample, frame, ctx)
        destination = source.store_path(sample, ctx)
        print(
            f"{path.name}: depth [{data.info['depth_min']:.3f}, {data.info['depth_max']:.3f}] "
            f"valid {data.info['valid_frac']:.2f} -> {destination}"
        )


if __name__ == "__main__":
    main()
