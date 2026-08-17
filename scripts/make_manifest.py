"""Build a manifest by scanning a world model's output directory.

    python scripts/make_manifest.py --config configs/scan.yaml

Ground-truth asset paths are not invented here — add `gt_assets` to the written
manifest yourself, or point the probes at in-pipeline ground-truth sources.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmeval.manifest import scan_run_dir, write_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/scan.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    root = (config_path.parent / config.get("root", "..")).resolve()

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else (root / path).resolve()

    manifest = scan_run_dir(
        resolve(config["run_dir"]),
        resolve(config["frames_dir"]),
        probes=config.get("probes"),
        video_ext=config.get("video_ext", ".mp4"),
        frame_ext=config.get("frame_ext", ".png"),
    )

    output = resolve(config.get("output", "configs/manifest.yaml"))
    write_manifest(output, manifest)

    n_videos = sum(len(s["videos"]) for s in manifest["samples"])
    print(f"{len(manifest['samples'])} samples, {n_videos} videos -> {output}")
    for orphan in manifest["_orphan_videos"]:
        print(f"  no conditioning frame for: {orphan}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
