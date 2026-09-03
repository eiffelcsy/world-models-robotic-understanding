"""Materialize RoboTrustBench as a wmgen `frames_dir` + `meta_file`.

https://huggingface.co/datasets/Huiqiong0124/RoboTrustBench_Dataset pairs one
conditioning image with one instruction per sample, across four folders —
data/A_Normal, data/B_Constraint-sensitive/<subcategory>,
data/C_Counterfactual/<subcategory>, data/D_Adversarial/<subcategory> — each
holding `<sample>_meta.json` (fields: language_instruction, uuid, image_source,
traj_id) beside `<sample>_varied_camera_1_left_image.png`. That is exactly what
wmgen's `frames_dir` source wants: an image per file, keyed by a sample id, with
a YAML `meta_file` carrying per-sample fields a prompt template reads as
`{task}` (see wmgen/sources.py).

The repo's own filenames aren't usable as-is on Windows: they embed a raw
timestamp with colons (`Fri_Apr_14_16:25:24_2023_meta.json`), which NTFS
rejects. `hf download` / `snapshot_download` mirror the repo's paths verbatim
into their local cache and fail on those files, so this fetches over plain
HTTPS instead, writing wherever we choose.

The sample id is NOT `uuid` alone, even though it looks like the natural
choice (colon-free, matches this repo's existing DROID sample-id convention,
e.g. `REAL+4dbb5646+2023-04-14-16h-25m-24s`) — `uuid` is not unique across the
1207 records. Two things collide on it:

  - the same DROID episode deliberately reappears in more than one category —
    e.g. the same `uuid` as both an A_Normal sample and a paired
    C_Counterfactual variant, presumably the same scene under a normal
    instruction and a stress-tested rephrasing of it;
  - 8 records (all under A_Normal, as of 2026-09) carry a literal `uuid: ""`,
    apparently an upstream data-entry gap, and are otherwise unrelated samples.

Keying on `uuid` alone silently drops every collision (dict overwrite) — a
1207-record dataset produced 1098 files the first time this ran. The sample id
here is instead `<uuid or slugified-traj_id>__<category[-subcategory]>`, which
cannot collide: it reconstructs the record's own path in the repo, and no two
records share a path.

    python scripts/fetch_robotrustbench.py --out ../robotrustbench
    python scripts/fetch_robotrustbench.py --out ../robotrustbench --limit 10   # smoke test

    ../robotrustbench/frames/<sample_id>.png    one per record, always 1207
    ../robotrustbench/meta.yaml                 {task, category, subcategory, source_traj_id, source_uuid} per sample_id
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

REPO = "Huiqiong0124/RoboTrustBench_Dataset"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
CATEGORIES = [
    "data/A_Normal",
    "data/B_Constraint-sensitive",
    "data/C_Counterfactual",
    "data/D_Adversarial",
]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url) as resp:
        return resp.read()


def _next_link(link_header: str | None) -> str | None:
    """Pagination, in case a category ever grows past the API's page size."""
    if not link_header:
        return None
    for part in link_header.split(","):
        if 'rel="next"' in part:
            return part.split(";")[0].strip().strip("<>")
    return None


def _list_files(repo_path: str) -> list[str]:
    files: list[str] = []
    url = f"{API}/{repo_path}?recursive=true"
    while url:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req) as resp:
            entries = json.loads(resp.read())
            url = _next_link(resp.headers.get("Link"))
        files += [e["path"] for e in entries if e["type"] == "file"]
    return files


def _resolve(repo_path: str) -> str:
    return f"{RESOLVE}/{urllib.parse.quote(repo_path)}"


def _slug(text: str) -> str:
    """Filesystem- and str.format-safe: colons, spaces and the rest collapse to `_`."""
    return re.sub(r"[^A-Za-z0-9+-]+", "_", text).strip("_")


def _sample_id(record: dict, category: str, subcategory: str | None) -> str:
    base = record["uuid"] or _slug(record["traj_id"])  # traj_id always present, uuid sometimes ""
    tag = category if subcategory is None else f"{category}-{subcategory}"
    return f"{base}__{_slug(tag)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="../robotrustbench", help="output root (frames/ and meta.yaml land here)")
    ap.add_argument("--limit", type=int, default=None, help="stop after N samples, for a smoke test")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    frames_dir = out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print("listing dataset files...")
    meta_paths: list[str] = []
    for category_path in CATEGORIES:
        meta_paths += sorted(f for f in _list_files(category_path) if f.endswith("_meta.json"))
    if args.limit:
        meta_paths = meta_paths[: args.limit]
    print(f"{len(meta_paths)} samples")

    meta: dict[str, dict] = {}
    for i, meta_path in enumerate(meta_paths, 1):
        record = json.loads(_get(_resolve(meta_path)))

        # data/<category>/[<subcategory>/[<sub-subcategory>/...]]<file>_meta.json — nesting is
        # irregular (B_Constraint-sensitive/Obscured and /Underspecified go one level deeper
        # than every other branch), so this takes whatever sits between category and filename
        # rather than assuming a fixed depth.
        parts = meta_path.split("/")
        category = parts[1]
        subcategory = "/".join(parts[2:-1]) or None
        sample_id = _sample_id(record, category, subcategory)

        if sample_id in meta:
            raise RuntimeError(f"sample id collision on {sample_id!r} at {meta_path} — the id scheme needs another look")

        image_path = "/".join(parts[:-1] + [record["image_source"]])
        dest = frames_dir / f"{sample_id}.png"
        if not dest.exists():
            dest.write_bytes(_get(_resolve(image_path)))

        meta[sample_id] = {
            "task": record["language_instruction"],
            "category": category,
            "subcategory": subcategory,
            "source_traj_id": record["traj_id"],
            "source_uuid": record["uuid"] or None,
        }
        print(f"  [{i}/{len(meta_paths)}] {sample_id}  ({category}{'/' + subcategory if subcategory else ''})")

    meta_out = out / "meta.yaml"
    meta_out.write_text(yaml.safe_dump({"meta": meta}, sort_keys=False), encoding="utf-8")

    assert len(meta) == len(meta_paths), f"{len(meta_paths)} records in, {len(meta)} out — some id still collided"
    print(f"\n{len(meta)} samples -> {frames_dir}")
    print(f"meta -> {meta_out}")


if __name__ == "__main__":
    main()
