"""Held-out confirmation: base vs final prompt, paired by frame (docs/prompt_framework.md, step 6).

    venv/bin/python scripts/confirm_stats.py <probe> <final_arm> <headline> [secondary ...]
    venv/bin/python scripts/confirm_stats.py robot_pose_estimation T3 pose_score pos_err angle_err heading_score

Reads results/confirmation/<probe>/libero/eval/<arm>/r1/scores.csv for `base` and the
final arm, on the 16 frames in configs/libero_test_ids.txt. A frame whose clip is
invalid in either arm is replaced, for both arms together, by the first regeneration
(r1001, r1002) in which both are valid, so every frame stays a paired comparison.
A frame with no valid pair after all three attempts is kept as a failure: an arm
still invalid on the last attempt scores 0 on the headline there (its other
measurements are kept as measured), paired with the other arm's clip from that
attempt (segmentation doc, section 6.1). Frames are never dropped.
Prints the mean difference with a 10,000-sample bootstrap 95% CI, a two-sided
Wilcoxon signed-rank test, the paired effect size d_z, and per-suite means.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

probe, final, head, *secondary = sys.argv[1:]
root = Path("results/confirmation") / probe / "libero" / "eval"
ids = [l.strip() for l in open("configs/libero_test_ids.txt") if l.strip() and not l.startswith("#")]


def load(arm: str, rep: str) -> dict[str, dict[str, str]]:
    path = root / arm / rep / "scores.csv"
    return {r["sample_id"]: r for r in csv.DictReader(open(path))} if path.exists() else {}


reps = ["r1", "r1001", "r1002"]
tables = {arm: [load(arm, rep) for rep in reps] for arm in ("base", final)}
pairs, notes, first_valid = {}, [], 0
for i in ids:
    for k, rep in enumerate(reps):
        b, f = tables["base"][k].get(i), tables[final][k].get(i)
        if b and f and b["valid"] == "1" and f["valid"] == "1":
            pairs[i] = (b, f)
            first_valid += k == 0
            if k:
                notes.append(f"{i}: both arms from {rep}")
            break
    else:
        last = len(reps) - 1
        while last and not (tables["base"][last].get(i) or tables[final][last].get(i)):
            last -= 1
        rows = []
        for arm in ("base", final):
            row = tables[arm][last].get(i)
            if row is None or row["valid"] != "1":
                # Keep the clip's own measurements (an error stays an error); the
                # headline, and its best-frame version, count the failure as 0.
                row = dict(row or {})
                row[head] = row["best_" + head] = "0"
                row["failed"] = "1"
            rows.append(row)
        pairs[i] = tuple(rows)
        failed = [arm for arm, row in zip(("base", final), rows) if row.get("failed")]
        notes.append(f"{i}: no valid pair after {last + 1} attempts; {', '.join(failed)} scored 0 as a failure ({reps[last]})")
failures = sum(any(row.get("failed") for row in pair) for pair in pairs.values())
print(f"{probe}: base vs {final}, {len(pairs)} frames: {len(pairs) - failures} valid pairs"
      f" ({first_valid} on the first attempt), {failures} failure(s) scored 0")
for n in notes:
    print("  " + n)

rng = np.random.default_rng(0)
lower_better = {"pos_err", "angle_err", "distance_m", "distance_px", "mean_dist", "coverage"}
for key in [head, "best_" + head] + secondary:
    try:
        b = np.array([float(pairs[i][0][key]) for i in pairs])
        f = np.array([float(pairs[i][1][key]) for i in pairs])
    except KeyError:
        continue
    d = f - b
    boot = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = wilcoxon(f, b).pvalue if np.any(d) else 1.0
    dz = d.mean() / d.std(ddof=1) if len(d) > 1 and d.std(ddof=1) > 0 else float("nan")
    better = int((d < 0).sum() if key in lower_better else (d > 0).sum())
    worse = int((d > 0).sum() if key in lower_better else (d < 0).sum())
    print(f"{key:16s} base {b.mean():.3f} (med {np.median(b):.3f})  {final} {f.mean():.3f} (med {np.median(f):.3f})  "
          f"diff {d.mean():+.3f} CI [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}  d_z={dz:.2f}  {final} better/worse {better}/{worse}")

print("\nper suite (descriptive):")
for suite in sorted({i.split("__")[0] for i in pairs}):
    fr = [i for i in pairs if i.startswith(suite)]
    print(f"  {suite:15s} {np.mean([float(pairs[i][0][head]) for i in fr]):.3f} -> {np.mean([float(pairs[i][1][head]) for i in fr]):.3f}  ({len(fr)} frames)")
print("\nper frame:")
for i in pairs:
    print(f"  {i:32s} {float(pairs[i][0][head]):.3f} -> {float(pairs[i][1][head]):.3f}")
