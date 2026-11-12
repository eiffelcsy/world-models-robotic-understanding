"""Item 2: how much the segmentation result depends on the decoder's thresholds.

    venv/bin/python scripts/rigour/decoder_sweep.py

Rescores the same clips with `min_saturation` swept from 0.3 to 0.9 (probe.yaml
uses 0.80) and `min_change` from 20 to 60 (probe.yaml uses 40). At every setting
it also reruns the decoder's two checks:
  floor    frame 0 of every held-out clip (the conditioning frame after the
           generator's compression) must decode to no instance
  ceiling  the true masks painted in distinct flat colours must score near 1
Nothing in probe.yaml is changed.
"""
from __future__ import annotations

import numpy as np

import common as c
import ideal
from funnel import iou, oracle_mask, paint_layers

PROBE = "instance_seg"
SATS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
CHANGES = [20.0, 40.0, 60.0]


def load():
    data = {"test": {}, "tune": {}, "floor": [], "ceiling": []}
    for arm in ("base", "H1"):
        data["test"][arm] = [(i, c.last_frame(c.final_clip(PROBE, arm, i, "r1")[0])) for i in c.ids("test")]
        data["floor"] += [(i, c.frame_at(c.final_clip(PROBE, arm, i, "r1")[0], 0)) for i in c.ids("test")]
    for arm in ("base", "E1", "H1"):
        data["tune"][arm] = [(i, c.last_frame(c.final_clip(PROBE, arm, i, "r0")[0])) for i in c.ids("tune")]
    for i in c.ids("all"):
        data["ceiling"].append((i, ideal.seg(i)))
    return data


def evaluate(data, **params):
    extractor, metrics = c.build(PROBE, **params)
    run = lambda items: [c.score(extractor, metrics, frame, i, c.truth(PROBE, i))[1] for i, frame in items]
    test = {arm: run(items) for arm, items in data["test"].items()}
    tune = {arm: run(items) for arm, items in data["tune"].items()}
    floor = sum(len(extractor(frame, c.sample(i), c.CTX).get("masks") or []) for i, frame in data["floor"])
    ceiling = run(data["ceiling"])
    return test, tune, floor, ceiling


def row(label, data, **params):
    test, tune, floor, ceiling = evaluate(data, **params)
    m = lambda rows, k="mean_iou": float(np.mean([r[k] for r in rows]))
    diff = np.array([h["mean_iou"] - b["mean_iou"] for h, b in zip(test["H1"], test["base"])])
    lo, hi = c.boot_ci(diff)
    matched = lambda rows: int(sum(r["n_matched"] for r in rows))
    return [label, m(test["base"]), m(test["H1"]), f"{diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]",
            f"{matched(test['base'])} / {matched(test['H1'])}",
            m(tune["base"]), m(tune["E1"]), m(tune["H1"]),
            floor, f"{m(ceiling):.3f} (min {min(r['mean_iou'] for r in ceiling):.3f})"]


def oracle_checks(data) -> str:
    """The oracle decoder of funnel.py must also pass the floor and the ceiling."""
    floor, ceiling = [], []
    for i, frame in data["floor"]:
        changed, hsv = paint_layers(frame, c.read_image(c.frame_path(i)))
        floor += [iou(oracle_mask(np.asarray(m, bool), changed, hsv), np.asarray(m, bool)) for m in c.truth(PROBE, i)["masks"]]
    for i, frame in data["ceiling"]:
        changed, hsv = paint_layers(frame, c.read_image(c.frame_path(i)))
        ceiling += [iou(oracle_mask(np.asarray(m, bool), changed, hsv), np.asarray(m, bool)) for m in c.truth(PROBE, i)["masks"]]
    return (f"Oracle decoder checks: on frame 0 of the held-out clips its mean object IoU is {np.mean(floor):.3f} "
            f"(max {np.max(floor):.3f}, {len(floor)} objects); on the painted true masks it is {np.mean(ceiling):.3f} "
            f"(min {np.min(ceiling):.3f}).")


if __name__ == "__main__":
    data = load()
    header = ["setting", "test base", "test H1", "H1 − base [95% CI]", "objects matched of 33 (base / H1)",
              "tune base", "tune E1", "tune H1", "floor: instances on 32 untouched frames", "ceiling"]
    sat_rows = [row(f"min_saturation {s:.1f}" + (" (probe.yaml)" if s == 0.8 else ""), data, min_saturation=s) for s in SATS]
    change_rows = [row(f"min_change {int(v)}" + (" (probe.yaml)" if v == 40 else ""), data, min_change=v) for v in CHANGES]
    c.write("decoder_sweep.md", "\n\n".join([
        "# Segmentation decoder sensitivity (item 2)",
        "Mean last-frame `mean_iou`. Test: 16 held-out frames, seed 1. Tune: 4 frames, seed 0.",
        "## Saturation threshold (min_change 40)", c.md_table(header, sat_rows),
        "## Change threshold (min_saturation 0.80)", c.md_table(header, change_rows),
        oracle_checks(data)]))
