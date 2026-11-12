"""Summarise prompt-ablation arms for one probe: mean scores, per-frame headline, and a contact sheet.

    venv/bin/python scripts/ablation_table.py <probe> <headline> <arm> [<arm> ...]
    venv/bin/python scripts/ablation_table.py robot_keypoint_detection pck paper R1 T1 E1 K1 base T2 E2 E3

Reads results/ablations/<probe>/eval/<arm>/r0/scores.csv (LIBERO tune frames, seed 0).
An invalid clip's scores are shown and flagged; the regenerated one, if any, is not
merged here. The sheet (input row, then each arm's last frame, cropped to the 720x720
render) goes to results/ablations/<probe>/sheet_<arms>.png.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

probe, headline, arms = sys.argv[1], sys.argv[2], sys.argv[3:]
ids = [l.strip() for l in open("configs/libero_tune_ids.txt") if l.strip() and not l.startswith("#")]
root = Path("results/ablations") / probe


def load(arm: str) -> dict[str, dict[str, str]]:
    path = root / "eval" / arm / "r0" / "scores.csv"
    return {r["sample_id"]: r for r in csv.DictReader(open(path))} if path.exists() else {}


def num(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return float("nan")


rows = {arm: load(arm) for arm in arms}
keys = [headline] + [k for k in ("hit", "focus", "on_object", "coverage", "pck", "precision", "recall", "n_pred",
                                 "pos_err", "angle_err", "pos_pck") if k != headline]
keys = [k for k in keys if any(k in r for rs in rows.values() for r in rs.values())]
print(f"{'arm':6s} {'valid':>5s} " + " ".join(f"{k[:9]:>9s}" for k in keys) + f"  best_{headline}")
for arm, r in rows.items():
    if not r:
        print(f"{arm:6s} (no scores)")
        continue
    valid = sum(r[i]["valid"] == "1" for i in r)
    means = " ".join(f"{np.nanmean([num(r[i], k) for i in r]):9.3f}" for k in keys)
    print(f"{arm:6s} {valid:>2d}/{len(r)} {means}  {np.nanmean([num(r[i], 'best_' + headline) for i in r]):.3f}")
    for i in r:
        if r[i]["valid"] != "1":
            print(f"       invalid {i}: {r[i].get('invalid_reason', '')}")
print(f"\nper frame {headline} ({', '.join(i.split('__')[0] for i in ids)}):")
for arm, r in rows.items():
    print(f"  {arm:6s} " + "  ".join(f"{num(r[i], headline):.2f}" + ("*" if r[i]['valid'] != '1' else "") if i in r else "  -  " for i in ids))


def last_frame(arm: str, sample_id: str) -> np.ndarray | None:
    cap, frame = cv2.VideoCapture(f"clips/libero/{probe}/{arm}/r0/{sample_id}.mp4"), None
    while True:
        ok, image = cap.read()
        if not ok:
            return frame
        frame = image


size = 240
sheet = []
for label in ["input"] + arms:
    tiles = []
    for i in ids:
        image = cv2.imread(f"data/libero/frames/{i}.png") if label == "input" else last_frame(label, i)
        tile = cv2.resize(image[:, 280:1000], (size, size)) if image is not None else np.full((size, size, 3), 60, np.uint8)
        r = rows.get(label, {})
        if i in r:
            cv2.putText(tile, f"{num(r[i], headline):.2f}" + ("" if r[i]["valid"] == "1" else " INV"),
                        (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(tile)
    cv2.putText(tiles[0], label, (6, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    sheet.append(np.hstack(tiles))
out = root / f"sheet_{'_'.join(arms)}.png"
cv2.imwrite(str(out), np.vstack(sheet))
print(f"\nsheet: {out}")
