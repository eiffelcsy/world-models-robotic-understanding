"""Item 1: split each headline score into stages (docs/rigour_checks.md §1).

    venv/bin/python scripts/rigour/funnel.py

Segmentation, per target object on the held-out clips:
  painted   at least half of the object's pixels differ from the conditioning frame
  readable  a decoder that is told the object's mask can recover it at IoU >= 0.5
            (the object's own modal hue, any saturation >= 0.2). An upper bound.
  matched   the probe's own decoder matches it at IoU >= 0.5 (the reported score)
Depth, per held-out clip: delta1 under the affine fit (reported), under the best
monotone map from brightness to depth (upper bound on any encoding), and the same
two numbers for the no-image row gradient.
Grasp and keypoints, per tune clip: the stages their metrics already compute.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.stats import spearmanr

import common as c

HUE = 180


# ---------------------------------------------------------------- segmentation
def blur(image: np.ndarray, k: int = 5) -> np.ndarray:
    return cv2.medianBlur(np.ascontiguousarray(image), k)


def paint_layers(frame: np.ndarray, reference: np.ndarray, min_change: float = 40.0):
    """Changed pixels and HSV of a frame, blurred as the decoder blurs them."""
    f, r = blur(frame), blur(reference)
    changed = np.abs(f.astype(np.int16) - r.astype(np.int16)).max(axis=2) >= min_change
    return changed, cv2.cvtColor(f, cv2.COLOR_RGB2HSV)


def oracle_mask(mask: np.ndarray, changed: np.ndarray, hsv: np.ndarray, sat_floor=0.2, tol_deg=20.0):
    """Best mask a colour decoder could return for this object, given its true mask.

    Hue is the mode of the changed pixels inside the object; the mask is every
    changed pixel of that hue, cleaned like the decoder's, keeping the connected
    pieces that touch the object. Spill that is connected to the object, and
    under-painting, still cost IoU.
    """
    defined = changed & (hsv[..., 1] >= round(sat_floor * 255)) & (hsv[..., 2] >= round(0.15 * 255))
    inside = defined & mask
    if inside.sum() < 0.1 * mask.sum():
        return np.zeros_like(mask)
    counts = np.bincount(hsv[..., 0][inside], minlength=HUE).astype(float)
    padded = np.concatenate([counts[-5:], counts, counts[:5]])
    peak = int(np.convolve(padded, np.ones(5) / 5, mode="same")[5:-5].argmax())
    distance = np.abs(hsv[..., 0].astype(np.int16) - peak)
    distance = np.minimum(distance, HUE - distance)
    binary = (defined & (distance <= tol_deg * HUE / 360)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
    _, labels = cv2.connectedComponents(binary, connectivity=8)
    keep = np.unique(labels[mask & (labels > 0)])
    return np.isin(labels, keep) & (labels > 0)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else 0.0


def seg_objects(arm: str, sample_ids: list[str], first: str, extractor, metrics, min_sat: float):
    """One row per (frame, object): painted, decodable, oracle IoU, strict IoU."""
    from modules.instance_seg.metrics import iou_matrix, match

    rows = []
    for i in sample_ids:
        video, _ = c.final_clip("instance_seg", arm, i, first)
        frame, reference = c.last_frame(video), c.read_image(c.frame_path(i))
        gt = c.truth("instance_seg", i)
        pred, _ = c.score(extractor, metrics, frame, i, gt)
        truths = [np.asarray(m, bool) for m in gt["masks"]]
        strict = {m.truth: m.iou for m in match(iou_matrix(pred.get("masks") or [], truths))}
        changed, hsv = paint_layers(frame, reference)
        saturated = changed & (hsv[..., 1] >= round(min_sat * 255)) & (hsv[..., 2] >= round(0.15 * 255))
        for j, mask in enumerate(truths):
            rows.append({
                "frame": i, "object": j, "area": int(mask.sum()),
                "painted": float(changed[mask].mean()),
                "decodable": float(saturated[mask].mean()),
                "oracle_iou": iou(oracle_mask(mask, changed, hsv), mask),
                "strict_iou": float(strict.get(j, 0.0)),
            })
    return rows


def seg_section() -> str:
    extractor, metrics = c.build("instance_seg")
    min_sat = extractor.min_saturation
    out = ["## Segmentation: held-out frames (16 frames, 33 objects, last frame)\n"]
    table, per_frame = [], {}
    for arm in ("base", "H1"):
        rows = seg_objects(arm, c.ids("test"), "r1", extractor, metrics, min_sat)
        n = len(rows)
        painted = sum(r["painted"] >= 0.5 for r in rows)
        readable = sum(r["oracle_iou"] >= 0.5 for r in rows)
        matched = sum(r["strict_iou"] >= 0.5 for r in rows)
        frames = sorted({r["frame"] for r in rows})
        by_frame = lambda key: [np.mean([r[key] for r in rows if r["frame"] == f]) for f in frames]
        o, s = by_frame("oracle_iou"), by_frame("strict_iou")
        per_frame[arm] = (o, s)
        lo, hi = c.boot_ci(o)
        table.append([arm, n, f"{painted} ({painted / n:.0%})", f"{readable} ({readable / n:.0%})",
                      f"{matched} ({matched / n:.0%})", float(np.mean(o)), f"[{lo:.3f}, {hi:.3f}]", float(np.mean(s))])
        lost_encoding = sum(r["oracle_iou"] >= 0.5 > r["strict_iou"] for r in rows)
        lost_locating = sum(r["painted"] >= 0.5 and r["oracle_iou"] < 0.5 for r in rows)
        out.append(f"- **{arm}**: {n - painted} objects not painted; {lost_locating} painted but not readable even "
                   f"with the true mask (wrong extent, merged or spilled); {lost_encoding} readable but lost by the "
                   f"probe's decoder (encoding); {matched} matched.")
    out.insert(1, c.md_table(
        ["arm", "objects", "painted", "readable (oracle IoU ≥ 0.5)", "matched (decoder IoU ≥ 0.5)",
         "mean oracle IoU / frame", "95% CI", "mean per-object decoder IoU / frame"], table) + "\n")
    d = np.array(per_frame["H1"][0]) - np.array(per_frame["base"][0])
    lo, hi = c.boot_ci(d)
    out.append(f"\nH1 − base on the oracle score: {d.mean():+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] (paired by frame).")
    out.append("\nThe oracle score is the mean over target objects of the best IoU a colour decoder could reach "
               "when told where the object is. It does not charge for extra masks, so it is an upper bound on "
               "`mean_iou`, not a replacement for it.")
    return "\n".join(out)


# ---------------------------------------------------------------------- depth
def isotonic_delta1(pred: np.ndarray, gt: np.ndarray, delta: float = 1.25) -> float:
    """delta1 after the best monotone map from predicted value to log depth, either direction."""
    best = 0.0
    values, inverse = np.unique(np.round(pred, 4), return_inverse=True)
    weights = np.bincount(inverse).astype(float)
    means = np.bincount(inverse, weights=np.log(gt)) / weights
    for sign in (1, -1):
        y, w = means[::sign].copy(), weights[::sign].copy()
        # pool adjacent violators
        blocks_y, blocks_w, blocks_n = [], [], []
        for value, weight in zip(y, w):
            blocks_y.append(value); blocks_w.append(weight); blocks_n.append(1)
            while len(blocks_y) > 1 and blocks_y[-2] > blocks_y[-1]:
                total = blocks_w[-2] + blocks_w[-1]
                merged = (blocks_y[-2] * blocks_w[-2] + blocks_y[-1] * blocks_w[-1]) / total
                count = blocks_n[-2] + blocks_n[-1]
                del blocks_y[-2:], blocks_w[-2:], blocks_n[-2:]
                blocks_y.append(merged); blocks_w.append(total); blocks_n.append(count)
        fitted = np.repeat(blocks_y, blocks_n)[::sign]
        ratio = np.exp(np.abs(fitted[inverse] - np.log(gt)))
        best = max(best, float(np.mean(ratio < delta)))
    return best


def depth_scores(pred: np.ndarray, gt: np.ndarray, metrics) -> dict[str, float]:
    scores = {}
    P, T = c.ProbeData("depth", {"depth": pred}), c.ProbeData("depth", {"depth": gt})
    for metric in metrics:
        scores.update(metric(P, T))
    valid = np.isfinite(gt) & np.isfinite(pred) & (gt > 1e-3)
    p, g = pred[valid].astype(np.float64), gt[valid].astype(np.float64)
    scores["iso_delta1"] = isotonic_delta1(p, g)
    step = max(1, len(p) // 100000)
    scores["spearman"] = abs(float(spearmanr(p[::step], g[::step]).statistic))
    return scores


def row_gradient(shape) -> np.ndarray:
    h, w = shape
    return np.tile(np.linspace(3, 1, h, dtype=np.float32)[:, None], (1, w))  # far at the top


def depth_section() -> str:
    extractor, metrics = c.build("depth_perception")
    keys = ["delta1", "disp_delta1", "iso_delta1", "spearman"]
    table, store = [], {}
    gts = {i: np.asarray(c.truth("depth_perception", i)["depth"], np.float32) for i in c.ids("all")}
    suite_mean, da3 = "mean depth of the same suite's other frames", "Depth Anything 3"
    arms = ["base", "N2", "row gradient", suite_mean]
    if all((c.OUT / "specialists" / f"da3__{i}.npz").exists() for i in c.ids("test")):
        arms.append(da3)
    for arm in arms:
        per = []
        for i in c.ids("test"):
            gt = gts[i]
            if arm == "row gradient":
                pred = row_gradient(gt.shape)
            elif arm == suite_mean:
                with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
                    __import__("warnings").simplefilter("ignore")
                    pred = np.nanmean(np.stack([v for j, v in gts.items() if j != i and c.suite(j) == c.suite(i)]), axis=0)
                pred = np.where(np.isfinite(pred), pred, np.nanmean(pred)).astype(np.float32)
            elif arm == da3:
                pred = np.load(c.OUT / "specialists" / f"da3__{i}.npz")["depth"].astype(np.float32)
            else:
                pred = extractor.decode(c.last_frame(c.clip("depth_perception", arm, "r1", i)))[0]
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, gt.shape[::-1])
            per.append(depth_scores(pred, gt, metrics))
        store[arm] = per
        table.append([arm] + [float(np.mean([s[k] for s in per])) for k in keys])
    out = ["## Depth: held-out frames (16 frames, last frame)\n",
           c.md_table(["prediction", "δ1 (affine, depth)", "δ1 (affine, disparity)", "δ1 (best monotone map)",
                       "|Spearman ρ|"], table), ""]
    for arm in ("base", "N2"):
        for prior in ("row gradient", suite_mean):
            parts = []
            for k in keys:
                d = np.array([a[k] - b[k] for a, b in zip(store[arm], store[prior])])
                lo, hi = c.boot_ci(d)
                parts.append(f"{k} {d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] ({(d > 0).sum()}/16)")
            out.append(f"- {arm} − {prior}: " + "; ".join(parts))
    out.append("\nThe monotone map is the depth analogue of the oracle decoder: it removes any penalty for *how* "
               "brightness encodes distance and keeps only the ordering. The gap between it and the affine scores "
               "is encoding loss; what is left below 1 is ordering error.")
    return "\n".join(out)


# ---------------------------------------------------------------------- grasp
def grasp_section() -> str:
    root = c.ROOT / "results/ablations/grasp_region_seg/eval"
    f = lambda r, k: float(r[k]) if r.get(k) not in (None, "") else float("nan")
    stages = [("valid", lambda r: r["valid"] == "1"),
              ("something painted", lambda r: f(r, "area_pred") > 0),
              ("paint on the target object (on_object ≥ 0.5)", lambda r: f(r, "on_object") >= 0.5),
              ("paint reaches the grasp (hit)", lambda r: f(r, "hit") == 1),
              ("a part, not the whole object (coverage < 0.5)", lambda r: f(r, "coverage") < 0.5),
              ("paint concentrated at the grasp (focus ≥ 0.5)", lambda r: f(r, "focus") >= 0.5)]
    groups = {"all arms": [], "base": [], "T2": []}
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = list(c.scores_csv(arm_dir / "r0" / "scores.csv").values())
        if arm_dir.name != "paper":  # paper asks for a green background: a different task
            groups["all arms"] += rows
        if arm_dir.name in groups:
            groups[arm_dir.name] += rows
    table = []
    for name, test in stages:
        row = [name]
        for g, rows in groups.items():
            alive = [r for r in rows if all(t(r) for _, t in stages[: [s[0] for s in stages].index(name) + 1])]
            row.append(f"{len(alive)}/{len(rows)}")
        table.append(row)
    return "\n".join(["## Grasp region: tune frames (4 frames, seed 0; each stage also requires the ones above)\n",
                      c.md_table(["stage"] + [f"{g}" for g in groups], table),
                      "\nTune clips only: there is no held-out grasp run yet. `paper` is left out of \"all arms\"."])


# ------------------------------------------------------------------ keypoints
def keypoint_section() -> str:
    probe = "robot_keypoint_detection"
    extractor, metrics = c.build(probe)
    root = c.ROOT / "clips/libero" / probe
    table = []
    for arm_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        n_gt = n_pred = hit5 = hit25 = clips = with_dots = 0
        for i in c.ids("tune"):
            video = arm_dir / "r0" / f"{i}.mp4"
            if not video.exists():
                continue
            gt = c.truth(probe, i)
            pred, _ = c.score(extractor, metrics, c.last_frame(video), i, gt)
            p, g = np.asarray(pred["points"], float).reshape(-1, 2), np.asarray(gt["points"], float).reshape(-1, 2)
            diagonal = float(np.hypot(*gt.info["frame_shape"]))
            clips += 1; with_dots += len(p) > 0; n_gt += len(g); n_pred += len(p)
            if len(p):
                nearest = np.linalg.norm(p[:, None] - g[None], axis=-1)
                # one-to-one, nearest first, as the metric matches
                taken_p, taken_g = set(), set()
                for flat in np.argsort(nearest, axis=None):
                    a, b = divmod(int(flat), len(g))
                    if a in taken_p or b in taken_g:
                        continue
                    taken_p.add(a); taken_g.add(b)
                    hit5 += nearest[a, b] <= 0.05 * diagonal
                    hit25 += nearest[a, b] <= 0.025 * diagonal
        if clips:
            table.append([arm_dir.name, clips, f"{with_dots}/{clips}", n_pred, n_gt,
                          f"{hit5}/{n_gt}", f"{hit25}/{n_gt}", f"{hit5}/{n_pred}" if n_pred else "0/0"])
    return "\n".join(["## Robot keypoints: tune frames (seed 0, last frame, current decoder)\n",
                      c.md_table(["arm", "clips", "clips with ≥ 1 dot", "dots decoded", "keypoints in view",
                                  "keypoints hit @5% diag (73 px)", "hit @2.5% (37 px)", "dots that hit (precision)"],
                                 table),
                      "\nTune clips only, scored in memory with the decoder in `probe.yaml` today. "
                      "Pose has no clips yet."])


if __name__ == "__main__":
    c.write("funnel.md", "\n\n".join(["# Score funnel (item 1)", seg_section(), depth_section(),
                                      grasp_section(), keypoint_section()]))
