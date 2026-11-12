"""Item 3: one reference table for all five probes.

    venv/bin/python scripts/rigour/references.py

For each probe, on the same LIBERO frames and with the probe's own decoder and
metric:
  null       the untouched conditioning frame, decoded as if it were the answer
  priors     guesses that use no knowledge of this image (a fixed pattern, or the
             mean ground truth of the other frames, leave-one-out)
  ceiling    the ideal answer drawn from the ground truth, through the decoder
  specialist SAM 3 / Depth Anything 3 on the frame (from specialists.py)
  model      Omni's clips, where they exist
skill = (model − best prior) / (ceiling − best prior), on the frames the model ran on.
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")  # the black bars have no depth in any frame

import common as c
import ideal

SETS = ("tune", "test", "all")
SPEC = c.OUT / "specialists"


def table(title: str, key: str, rows: dict[str, dict[str, float]], note: str = "") -> str:
    """rows: label -> {sample_id: score}. Means over tune (4), test (16), all (20)."""
    body = []
    for label, per in rows.items():
        cells = []
        for which in SETS:
            values = [per[i] for i in c.ids(which) if i in per]
            cells.append(f"{np.mean(values):.3f}" + ("" if len(values) == len(c.ids(which)) else f" (n={len(values)})")
                         if values else "—")
        body.append([label] + cells)
    return "\n".join([f"## {title} (`{key}`)\n", c.md_table(["reference", "tune (4)", "test (16)", "all (20)"], body), note])


def skill(rows, model: str, priors: list[str], ceiling: str, which: str) -> str:
    frames = [i for i in c.ids(which) if i in rows[model]]
    mean = lambda label: float(np.mean([rows[label][i] for i in frames]))
    best = max(priors, key=mean)
    span = mean(ceiling) - mean(best)
    per_frame = np.array([rows[model][i] - rows[best][i] for i in frames])
    lo, hi = c.boot_ci(per_frame)
    ratio = f"{per_frame.mean() / span:.2f}" if span > 0.05 else "undefined (prior at or above the ceiling)"
    return (f"- **{model}** ({which}, n={len(frames)}): {mean(model):.3f} against best prior \"{best}\" {mean(best):.3f} "
            f"and ceiling {mean(ceiling):.3f}; difference from prior {per_frame.mean():+.3f} [{lo:+.3f}, {hi:+.3f}], "
            f"above it on {(per_frame > 0).sum()}/{len(frames)} frames; **skill {ratio}**")


def loo(values: dict[str, np.ndarray], sample_id: str, same_suite: bool) -> list[np.ndarray]:
    return [v for j, v in values.items() if j != sample_id and (not same_suite or c.suite(j) == c.suite(sample_id))]


# ---------------------------------------------------------------- segmentation
def seg() -> str:
    from modules.instance_seg.metrics import score_masks

    probe = "instance_seg"
    extractor, metrics = c.build(probe)
    loose, _ = c.build(probe, min_saturation=0.5)  # sensitivity only; see decoder_sweep.py
    rows: dict[str, dict[str, float]] = {k: {} for k in (
        "null: untouched frame", "ceiling: true masks painted", "SAM 3, noun prompt", "SAM 3, referring phrase",
        "Omni base", "Omni H1", "Omni base, decoder at saturation 0.5", "Omni H1, decoder at saturation 0.5")}
    detail = {}
    for i in c.ids("all"):
        gt = c.truth(probe, i)
        rows["null: untouched frame"][i] = c.score(extractor, metrics, c.read_image(c.frame_path(i)), i, gt)[1]["mean_iou"]
        rows["ceiling: true masks painted"][i] = c.score(extractor, metrics, ideal.seg(i), i, gt)[1]["mean_iou"]
        for label, name in (("SAM 3, noun prompt", "noun"), ("SAM 3, referring phrase", "phrase")):
            path = SPEC / f"sam3_{name}__{i}.npz"
            if path.exists():
                s = score_masks(list(np.load(path)["masks"]), gt["masks"])
                rows[label][i] = s["mean_iou"]
                detail.setdefault(label, []).append((i, s))
        for arm, first in (("base", "r1"), ("H1", "r1")) if i in c.ids("test") else (("base", "r0"), ("H1", "r0")):
            video = c.final_clip(probe, arm, i, first)[0]
            if video.exists():
                frame = c.last_frame(video)
                rows[f"Omni {arm}"][i] = c.score(extractor, metrics, frame, i, gt)[1]["mean_iou"]
                rows[f"Omni {arm}, decoder at saturation 0.5"][i] = c.score(loose, metrics, frame, i, gt)[1]["mean_iou"]
    rows = {k: v for k, v in rows.items() if v}
    out = [table("Instance segmentation", "mean_iou", rows)]
    for label, items in detail.items():
        test = [s for i, s in items if i in c.ids("test")]
        out.append(f"- {label}, test: matched {int(sum(s['n_matched'] for s in test))}/{int(sum(s['n_truth'] for s in test))}, "
                   f"matched IoU {np.mean([s['matched_iou'] for s in test]):.3f}, recall {np.mean([s['recall'] for s in test]):.3f}, "
                   f"precision {np.mean([s['precision'] for s in test]):.3f}, count error {np.mean([s['count_error'] for s in test]):.3f}")
    complete = lambda label: label in rows and all(i in rows[label] for i in c.ids("all"))
    if complete("SAM 3, noun prompt") and complete("SAM 3, referring phrase"):
        changed = [i for i in c.ids("all") if abs(rows["SAM 3, noun prompt"][i] - rows["SAM 3, referring phrase"][i]) > 1e-9]
        out.append(f"- The phrase differs from the noun on 5 frames (the spatial suite); the score changes on {len(changed)}: "
                   + ", ".join(f"{i.split('__')[1]} {rows['SAM 3, noun prompt'][i]:.2f}→{rows['SAM 3, referring phrase'][i]:.2f}" for i in changed))
    for sam in ("SAM 3, noun prompt", "SAM 3, referring phrase"):
        if complete(sam):
            for arm in ("Omni base", "Omni H1", "Omni base, decoder at saturation 0.5", "Omni H1, decoder at saturation 0.5"):
                d = np.array([rows[sam][i] - rows[arm][i] for i in c.ids("test")])
                lo, hi = c.boot_ci(d)
                out.append(f"- {sam} − {arm}, test: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    out.append("- Segmentation has no image-free prior worth the name, so no skill score; the null is 0 and the ceiling 0.99. "
               "Tune-set Omni scores are the tuning clips (seed 0) and carry the selection effect.")
    return "\n".join(out)


# ---------------------------------------------------------------------- depth
def depth() -> str:
    probe = "depth_perception"
    extractor, metrics = c.build(probe)
    gts = {i: np.asarray(c.truth(probe, i)["depth"], np.float32) for i in c.ids("all")}

    def both(pred: np.ndarray, i: str) -> tuple[float, float]:
        if pred.shape != gts[i].shape:
            pred = cv2.resize(pred, gts[i].shape[::-1])
        P, T = c.ProbeData("depth", {"depth": pred.astype(np.float32)}), c.ProbeData("depth", {"depth": gts[i]})
        s = {}
        for metric in metrics:
            s.update(metric(P, T))
        return s["delta1"], s["disp_delta1"]

    labels = ["prior: constant depth", "null: untouched frame as depth", "prior: row gradient (far at top)",
              "prior: mean depth of the other 19 frames", "prior: mean depth of the same suite's other frames",
              "ceiling: true depth, linear in depth", "ceiling: true depth, linear in inverse depth",
              "Depth Anything 3", "Omni base", "Omni N2"]
    d1 = {k: {} for k in labels}
    dd = {k: {} for k in labels}
    rng = np.random.default_rng(0)
    for i in c.ids("all"):
        h, w = gts[i].shape
        preds = {
            labels[0]: np.ones((h, w), np.float32) + 1e-6 * rng.random((h, w), dtype=np.float32),
            labels[1]: extractor.decode(c.read_image(c.frame_path(i)))[0],
            labels[2]: np.tile(np.linspace(3, 1, h, dtype=np.float32)[:, None], (1, w)),
            labels[3]: np.nanmean(np.stack(loo(gts, i, False)), axis=0),
            labels[4]: np.nanmean(np.stack(loo(gts, i, True)), axis=0),
            labels[5]: extractor.decode(ideal.depth(i, "linear"))[0],
            labels[6]: extractor.decode(ideal.depth(i, "disparity"))[0],
        }
        if (SPEC / f"da3__{i}.npz").exists():
            preds[labels[7]] = np.load(SPEC / f"da3__{i}.npz")["depth"]
        for arm, label in (("base", labels[8]), ("N2", labels[9])):
            video = c.clip(probe, arm, "r1" if i in c.ids("test") else "r0", i)
            if video.exists():
                preds[label] = extractor.decode(c.last_frame(video))[0]
        for label, pred in preds.items():
            pred = np.where(np.isfinite(pred), pred, np.nanmean(pred)) if label in (labels[3], labels[4]) else pred
            d1[label][i], dd[label][i] = both(pred, i)
    d1, dd = ({k: v for k, v in t.items() if v} for t in (d1, dd))
    priors = labels[:5]
    out = [table("Depth perception", "delta1", d1), "", table("Depth perception, disparity space", "disp_delta1", dd), ""]
    for arm in ("Omni base", "Omni N2"):
        out.append(skill(d1, arm, priors, labels[5], "test") + " (δ1)")
        out.append(skill(dd, arm, priors, labels[6], "test") + " (disparity δ1)")
    if all(i in d1.get(labels[7], {}) for i in c.ids("all")):
        out.append(skill(d1, labels[7], priors, labels[5], "test") + " (δ1)")
        out.append(skill(dd, labels[7], priors, labels[6], "test") + " (disparity δ1)")
        for arm in ("Omni base", "Omni N2"):
            for name, t in (("δ1", d1), ("disparity δ1", dd)):
                d = np.array([t[labels[7]][i] - t[arm][i] for i in c.ids("test")])
                lo, hi = c.boot_ci(d)
                out.append(f"- Depth Anything 3 − {arm}, test, {name}: {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    return "\n".join(out)


# ------------------------------------------------------------------ keypoints
def keypoints() -> str:
    probe = "robot_keypoint_detection"
    extractor, metrics = c.build(probe)
    gts = {i: c.truth(probe, i) for i in c.ids("all")}
    shape = gts[c.ids("all")[0]].info["frame_shape"]

    def mean_points(i: str, same_suite: bool) -> np.ndarray:
        by_label: dict[str, list] = {}
        for j, gt in gts.items():
            if j == i or (same_suite and c.suite(j) != c.suite(i)):
                continue
            for label, point in zip(gt["labels"], np.asarray(gt["points"], float)):
                by_label.setdefault(str(label), []).append(point)
        return np.array([np.mean(v, axis=0) for v in by_label.values()], np.float32).reshape(-1, 2)

    def run(points: np.ndarray, i: str) -> dict[str, float]:
        P = c.ProbeData("points", {"points": points}, {"frame_shape": shape})
        return metrics[0](P, gts[i])

    labels = ["null: untouched frame", "prior: mean position per joint, other 19 frames",
              "prior: mean position per joint, same suite", "ceiling: true keypoints as 4 px dots", "Omni base"]
    pck, f1 = ({k: {} for k in labels} for _ in range(2))
    for i in c.ids("all"):
        results = {
            labels[0]: c.score(extractor, metrics, c.read_image(c.frame_path(i)), i, gts[i])[1],
            labels[1]: run(mean_points(i, False), i),
            labels[2]: run(mean_points(i, True), i),
            labels[3]: c.score(extractor, metrics, ideal.keypoints(i, 4), i, gts[i])[1],
        }
        video = c.clip(probe, "base", "r0", i)
        if video.exists():
            results[labels[4]] = c.score(extractor, metrics, c.last_frame(video), i, gts[i])[1]
        for label, s in results.items():
            pck[label][i], f1[label][i] = s["pck"], s["f1"]
    pck, f1 = ({k: v for k, v in t.items() if v} for t in (pck, f1))
    out = [table("Robot keypoints", "pck", pck), "", table("Robot keypoints, F1", "f1", f1), ""]
    if labels[4] in pck:
        out.append(skill(pck, labels[4], labels[:3], labels[3], "tune") + " (PCK)")
        out.append(skill(f1, labels[4], labels[:3], labels[3], "tune") + " (F1)")
    out.append("- The mean-position prior predicts every joint seen in the other frames, so it also pays for joints that are "
               "out of view here (precision); PCK is recall and ignores that, F1 does not.")
    return "\n".join(out)


# ----------------------------------------------------------------------- pose
def pose() -> str:
    probe = "robot_pose_estimation"
    extractor, metrics = c.build(probe)
    gts = {i: c.truth(probe, i) for i in c.ids("all")}
    grips = {i: np.asarray(gt["points"], float)[0] for i, gt in gts.items()}
    shape = gts[c.ids("all")[0]].info["frame_shape"]

    def guess(i: str, same_suite: bool) -> dict[str, float]:
        centre = np.mean(loo(grips, i, same_suite), axis=0)
        P = c.ProbeData("points", {"points": np.array([centre, centre + [0, 60]], np.float32)}, {"frame_shape": shape})
        return metrics[0](P, gts[i])

    labels = ["null: untouched frame", "prior: mean grip point of the other 19 frames, heading straight down",
              "prior: mean grip point of the same suite, heading straight down",
              "ceiling: true pose as 4 px dots", "Omni base"]
    keys = ("pose_score", "pos_pck", "heading_score")
    tables = {k: {label: {} for label in labels} for k in keys}
    for i in c.ids("all"):
        results = {
            labels[0]: c.score(extractor, metrics, c.read_image(c.frame_path(i)), i, gts[i])[1],
            labels[1]: guess(i, False), labels[2]: guess(i, True),
            labels[3]: c.score(extractor, metrics, ideal.pose(i, 4), i, gts[i])[1],
        }
        video = c.clip(probe, "base", "r0", i)
        if video.exists():
            results[labels[4]] = c.score(extractor, metrics, c.last_frame(video), i, gts[i])[1]
        for label, s in results.items():
            for k in keys:
                tables[k][label][i] = s[k]
    out = []
    for k in keys:
        rows = {label: v for label, v in tables[k].items() if v}
        out += [table("Robot pose", k, rows), ""]
    rows = {label: v for label, v in tables["pose_score"].items() if v}
    if labels[4] in rows:
        out.append(skill(rows, labels[4], labels[:3], labels[3], "tune"))
    else:
        out.append("- No pose clips exist yet, so there is no model row. Any pose result has to beat these priors.")
    return "\n".join(out)


# ---------------------------------------------------------------------- grasp
def grasp() -> str:
    probe = "grasp_region_seg"
    extractor, metrics = c.build(probe)
    arms = ("base", "T2", "T4", "E3")
    labels = ["null: untouched frame", "object known: whole object painted",
              "object known: dot (1 cm radius) on its centroid", "object known: top quarter of the object",
              "object and gripper known: dot (1 cm) on the object point nearest the gripper",
              "ceiling: dot on the demonstrated grasp, 1 cm radius (finger-wide)",
              "ceiling: dot on the demonstrated grasp, 2 cm radius (the hit tolerance)"] + [f"Omni {a}" for a in arms]
    keys = ("grasp_loc", "grasp_score", "hit")
    tables = {k: {label: {} for label in labels} for k in keys}
    for i in c.ids("all"):
        gt = c.truth(probe, i)
        frames = {labels[0]: c.read_image(c.frame_path(i)), labels[1]: ideal.grasp(i, "object"),
                  labels[2]: ideal.grasp(i, "centroid"), labels[3]: ideal.grasp(i, "top"),
                  labels[4]: ideal.grasp(i, "gripper"),
                  labels[5]: ideal.grasp(i, "grasp", 0.01), labels[6]: ideal.grasp(i, "grasp", 0.02)}
        for arm in arms:
            video = c.final_clip(probe, arm, i, "r0")[0]
            if video.exists():
                frames[f"Omni {arm}"] = c.last_frame(video)
        for label, frame in frames.items():
            s = c.score(extractor, metrics, frame, i, gt)[1]
            for k in keys:
                tables[k][label][i] = s[k]
    out = []
    for k in keys:
        out += [table("Grasp region", k, {label: v for label, v in tables[k].items() if v}), ""]
    rows = {label: v for label, v in tables["grasp_loc"].items() if v}
    for arm in arms:
        if f"Omni {arm}" in rows:
            out.append(skill(rows, f"Omni {arm}", labels[:5], labels[5], "tune") + " (grasp_loc)")
    out.append("- The \"object known\" rows are given the true object mask (and, for the last, the gripper's image "
               "position) and know nothing about the grasp. They are not image-free priors: they show what finding the "
               "object alone is worth. `grasp_loc` is the headline from 2026-11-11; `grasp_score` is the old one.")
    return "\n".join(out)


if __name__ == "__main__":
    c.write("references.md", "\n\n".join(["# Reference table (item 3)", seg(), depth(), keypoints(), pose(), grasp()]))
