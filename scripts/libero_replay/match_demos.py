"""Find the original LIBERO demo behind each lerobot/libero episode in the meta.

lerobot/libero (1693 episodes) is OpenVLA's "no-noops" re-render of the original
LIBERO demos: each demo's actions were replayed in the simulator with no-op actions
dropped, and only successful replays kept. So an episode's actions equal one
original demo's actions after that same filter. This finds that demo per sample by
comparing actions, reading only the `actions` arrays of the original HDF5 files
over HTTP range requests (the files are 0.5-2 GB, mostly images).

Writes data/libero/demo_map.yaml (sample id -> hdf5 url, demo key, action error)
and, per sample, data/libero/demos/<sample_id>.npz with that demo's first
simulator state and its kept actions, which is all the replay needs. Resumable:
samples already cached are not fetched again. Run in venv_libero:

    venv_libero/bin/python scripts/libero_replay/match_demos.py
"""

from __future__ import annotations

import argparse
import multiprocessing
from pathlib import Path

import aiohttp
import fsspec
import h5py
import numpy as np
import pandas as pd
import yaml

SOURCE = "https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/resolve/main"


def is_noop(action: np.ndarray, previous: np.ndarray | None, threshold: float = 1e-4) -> bool:
    """OpenVLA's rule (experiments/robot/libero/regenerate_libero_dataset.py)."""
    if previous is None:
        return float(np.linalg.norm(action[:-1])) < threshold
    return float(np.linalg.norm(action[:-1])) < threshold and action[-1] == previous[-1]


def kept_steps(actions: np.ndarray) -> list[int]:
    kept: list[int] = []
    for i, action in enumerate(actions):
        if is_noop(action, actions[kept[-1]] if kept else None):
            continue
        kept.append(i)
    return kept


def match(job: tuple) -> tuple[str, dict | None, str]:
    """One sample -> its demo, caching that demo's first state and kept actions."""
    sample_id, url, target, guess, cache = job
    cache = Path(cache)
    if cache.exists():
        with np.load(cache) as saved:
            info = {k: saved[k].item() if saved[k].ndim == 0 else saved[k].tolist() for k in ("demo", "action_error")}
        return sample_id, {"url": url, **info}, "cached"
    for attempt in range(4):
        try:
            fs = fsspec.filesystem("https", block_size=1 << 18, client_kwargs={"timeout": aiohttp.ClientTimeout(total=600)})
            best = None
            with fs.open(url, "rb") as handle, h5py.File(handle, "r") as f:
                demos = set(f["data"])
                order = sorted(range(50), key=lambda i: abs(i - guess))
                for index in order:
                    demo = f"demo_{index}"
                    if demo not in demos:
                        continue
                    actions = np.asarray(f["data"][demo]["actions"])
                    kept = kept_steps(actions)
                    if len(kept) != len(target):
                        continue
                    error = float(np.abs(actions[kept] - target).max())
                    if best is None or error < best[0]:
                        best = (error, demo, kept, actions)
                    if error < 1e-6:
                        break
                if best is None:
                    return sample_id, None, f"no demo with {len(target)} kept steps"
                error, demo, kept, actions = best
                first_state = np.asarray(f["data"][demo]["states"][0])
            np.savez(cache, demo=demo, action_error=error, kept_steps=np.asarray(kept),
                     actions=actions[kept], first_state=first_state)
            return sample_id, {"url": url, "demo": demo, "action_error": error}, f"attempt {attempt + 1}"
        except Exception as exc:  # network: retry
            last = repr(exc)[:200]
    return sample_id, None, f"failed: {last}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta", type=Path, default=Path("data/libero/meta.yaml"))
    parser.add_argument("--raw", type=Path, default=Path("data/libero/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/libero/demo_map.yaml"))
    parser.add_argument("--cache", type=Path, default=Path("data/libero/demos"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    meta = yaml.safe_load(args.meta.read_text())["meta"]
    wanted = {v["episode_index"] for v in meta.values()}
    data = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "task_index", "action"])
                      for p in sorted((args.raw / "data").rglob("*.parquet"))])
    # A guess at the demo index to search outward from: the episode's rank among its
    # task's episodes. OpenVLA kept successful replays in demo order, though the
    # dataset need not list them in that order, so every demo may get checked.
    tasks = data.groupby("episode_index").task_index.first()
    rank = {e: int((tasks[tasks == t].index < e).sum()) for e, t in tasks.items() if e in wanted}
    data = data[data.episode_index.isin(wanted)].sort_values(["episode_index", "frame_index"])

    args.cache.mkdir(parents=True, exist_ok=True)
    jobs = []
    for sample_id, entry in sorted(meta.items()):
        target = np.stack(data[data.episode_index == entry["episode_index"]].action.to_numpy())
        url = f"{SOURCE}/{entry['category']}/{entry['bddl'].removesuffix('.bddl')}_demo.hdf5"
        jobs.append((sample_id, url, target, rank[entry["episode_index"]], str(args.cache / f"{sample_id}.npz")))

    out = yaml.safe_load(args.out.read_text()) if args.out.exists() else {}
    with multiprocessing.Pool(args.workers) as pool:
        for sample_id, found, note in pool.imap_unordered(match, jobs):
            print(f"{sample_id}: {found} ({note})", flush=True)
            if found:
                out[sample_id] = {**found, "cache": str(args.cache / f"{sample_id}.npz")}
                args.out.write_text(yaml.safe_dump(out, sort_keys=True))


if __name__ == "__main__":
    main()
