"""Draw one review sheet per LIBERO sample, for checking the ground truth by eye.

Reads exactly what the eval scores against: the stores in results/libero/ written
by the simulator ground-truth sources, plus the meta and the replay record. Writes
results/libero/review/<sample_id>.png and an index.html that shows them all with
the checklist from docs/libero_ground_truth.md ("Checking it by eye").

Each sheet is three rows of three panels, cropped to the rendered square:

  1  dataset frame        | replay render (the new input) | difference, x4
  2  instance masks       | depth, with object depths      | keypoints + pose
  3  grasp on the input   | dataset frame AT the grasp     | facts and checks

    venv/bin/python scripts/libero_replay/review_gt.py
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wmeval.components import load_probe_data  # noqa: E402

import fetch_libero  # noqa: E402

LEFT, SIZE = 280, 720
PANEL = 480
STORES = {
    "instances": "instance_seg_sim_gt",
    "depth": "depth_perception_sim_gt",
    "keypoints": "robot_keypoints_sim_gt",
    "pose": "robot_pose_recorded_gt",
    "grasp": "grasp_demonstrated_gt",
}
COLORS = [(230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240)]
KEYPOINT_COLORS = [(255, 0, 0), (255, 140, 0), (255, 255, 0), (0, 255, 0), (0, 255, 255), (0, 120, 255), (255, 0, 255)]

CHECKLIST = [
    ("Replay", "Panels 1 and 2 show the same scene, arm pose and object positions; "
               "panel 3 is dark except edges and texture (render quality, not position)."),
    ("Instances", "One tint per object the instruction names, and nothing else tinted. "
                  "Each tint covers the whole visible object and stops at its edge. "
                  "Labels match the instruction's words."),
    ("Depth", "Nearer is warmer. The table gets cooler going away from the camera. "
              "Object depths are ordered as they look: nearer objects have smaller numbers."),
    ("Keypoints", "Each dot sits on the arm at the named link (link0 at the base, hand at the "
                  "gripper), in order along the arm. Links listed as out of view really are off-frame."),
    ("Pose", "The white ring is between the fingers; the arrow points the way the fingers point."),
    ("Grasp", "Panel 7: the cyan outline is exactly the object the robot picks. Panel 8 is the "
              "dataset frame at the grasp: the two magenta finger points sit on the fingertips, "
              "on that same object."),
]


def crop(image: np.ndarray) -> np.ndarray:
    return image[:, LEFT:LEFT + SIZE]


def panel(image: np.ndarray, title: str) -> np.ndarray:
    out = cv2.resize(image, (PANEL, PANEL), interpolation=cv2.INTER_AREA)
    cv2.rectangle(out, (0, 0), (PANEL, 22), (0, 0, 0), -1)
    cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def label(image: np.ndarray, text: str, at: tuple[int, int], color=(255, 255, 255), scale=0.55) -> None:
    x, y = int(at[0]), int(at[1])
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(np.clip(x, 0, image.shape[1] - w - 4))
    cv2.rectangle(image, (x - 3, y - h - 4), (x + w + 3, y + base + 1), (0, 0, 0), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def shifted(points: np.ndarray) -> np.ndarray:
    return np.asarray(points, dtype=float).reshape(-1, 2) - np.array([LEFT, 0])


def text_panel(lines: list[str]) -> np.ndarray:
    out = np.full((SIZE, SIZE, 3), 30, dtype=np.uint8)
    y = 64
    for line in lines:
        for part in [line[i:i + 52] for i in range(0, max(len(line), 1), 52)]:
            label(out, part, (14, y), scale=0.62)
            y += 27
    return out


def instances_panel(frame, masks, names):
    view = frame.copy()
    for i, (mask, name) in enumerate(zip(masks, names)):
        color = np.array(COLORS[i % len(COLORS)])
        view[mask] = (0.45 * view[mask] + 0.55 * color).astype(np.uint8)
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(view, contours, -1, tuple(int(c) for c in color), 2)
    for i, (mask, name) in enumerate(zip(masks, names)):
        rows, cols = np.nonzero(mask)
        label(view, name, (cols.mean() - 40, rows.min() - 6), COLORS[i % len(COLORS)])
    return view


def depth_panel(depth, masks, names):
    finite = np.isfinite(depth)
    lo, hi = np.percentile(depth[finite], [1, 99])
    norm = np.clip((depth - lo) / (hi - lo), 0, 1)
    view = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1].copy()
    view[~finite] = 0
    for mask, name in zip(masks, names):
        rows, cols = np.nonzero(mask)
        label(view, f"{name}: {np.median(depth[mask]):.2f} m", (cols.mean() - 60, rows.mean()))
    label(view, f"warm = near {lo:.2f} m   cool = far {hi:.2f} m", (10, SIZE - 14))
    return view


def robot_panel(frame, keypoints, keypoint_labels, pose):
    view = frame.copy()
    for i, (point, name) in enumerate(zip(shifted(keypoints), keypoint_labels)):
        color = KEYPOINT_COLORS[i % len(KEYPOINT_COLORS)]
        cv2.circle(view, tuple(int(v) for v in point), 9, (0, 0, 0), -1)
        cv2.circle(view, tuple(int(v) for v in point), 7, color, -1)
        label(view, name.replace("panda_", ""), (point[0] + 11, point[1] + 5), color)
    eef, ahead = shifted(pose)
    cv2.arrowedLine(view, tuple(int(v) for v in eef), tuple(int(v) for v in ahead), (255, 255, 255), 3, tipLength=0.3)
    cv2.circle(view, tuple(int(v) for v in eef), 10, (255, 255, 255), 2)
    label(view, "eef", (eef[0] + 12, eef[1] - 10))
    return view


def grasp_panel(frame, mask, grasp, show_mask=True):
    view = frame.copy()
    if show_mask:
        contours = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cv2.drawContours(view, contours, -1, (0, 255, 255), 2)
    centre, *fingers = shifted(grasp)
    for finger in fingers:
        cv2.circle(view, tuple(int(v) for v in finger), 6, (255, 0, 255), -1)
    cv2.line(view, *(tuple(int(v) for v in f) for f in fingers), (255, 0, 255), 1)
    cv2.drawMarker(view, tuple(int(v) for v in centre), (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
    return view


def grasp_moment(raw: Path, entry: dict, episodes: pd.DataFrame) -> np.ndarray | None:
    """The dataset's own frame at the recorded grasp, un-mirrored onto the canvas."""
    episode = episodes[episodes["episode_index"] == entry["episode_index"]].iloc[0]
    key = fetch_libero.VIDEO_KEY
    video = raw / f"videos/{key}/chunk-{int(episode[f'videos/{key}/chunk_index']):03d}/file-{int(episode[f'videos/{key}/file_index']):03d}.mp4"
    if not video.exists():
        return None
    start = int(round(float(episode[f"videos/{key}/from_timestamp"]) * fetch_libero.FPS))
    frame = fetch_libero.read_frames(video, start + int(entry["grasp_frame_index"]), 1)[0]
    return fetch_libero.to_canvas(frame[:, ::-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/libero"))
    parser.add_argument("--gt", type=Path, default=Path("results/libero"))
    parser.add_argument("--out", type=Path, default=Path("results/libero/review"))
    args = parser.parse_args()

    meta = yaml.safe_load((args.data / "meta.yaml").read_text())["meta"]
    report = yaml.safe_load((args.data / "sim_report.yaml").read_text())
    episodes = pd.read_parquet(args.data / "raw/meta/episodes/chunk-000/file-000.parquet")
    episodes["episode_index"] = episodes["episode_index"].astype(int)
    args.out.mkdir(parents=True, exist_ok=True)

    cards = []
    for sample_id, entry in sorted(meta.items()):
        gt = {name: load_probe_data(args.gt / store / f"{sample_id}_gt.npz") for name, store in STORES.items()}
        frame = crop(cv2.cvtColor(cv2.imread(str(args.data / "frames" / f"{sample_id}.png")), cv2.COLOR_BGR2RGB))
        original = crop(cv2.cvtColor(cv2.imread(str(args.data / "frames_lerobot" / f"{sample_id}.png")), cv2.COLOR_BGR2RGB))
        difference = np.clip(np.abs(frame.astype(int) - original.astype(int)) * 4, 0, 255).astype(np.uint8)

        masks = [crop(np.asarray(m, bool)) for m in gt["instances"]["masks"]]
        names = gt["instances"].info["labels"]
        depth = crop(np.asarray(gt["depth"]["depth"], np.float32))
        moment = grasp_moment(args.data / "raw", entry, episodes)
        grasp_mask = crop(np.asarray(gt["grasp"]["mask"], bool))
        checks = report[sample_id]

        facts = [
            f"{sample_id}",
            f"task: {entry['task']}",
            f"demo: {checks['demo']}  (frame {entry['frame_index']}, grasp at frame {entry['grasp_frame_index']})",
            f"replay eef error: {checks['max_eef_error_m'] * 1000:.2f} mm (max over steps)",
            f"replay vs dataset image: {checks['image_error']:.1f}/255 mean abs",
            f"instance objects: {', '.join(names)}",
            f"hidden objects: {', '.join(gt['instances'].info.get('hidden') or []) or 'none'}",
            f"grasped object: {entry['sim_grasp_object']}  ('{entry['grasp_concept']}')",
            f"grasp offset from object centre: {gt['grasp'].info['grasp_offset_m'] * 100:.1f} cm",
            f"keypoints in view: {', '.join(l.replace('panda_', '') for l in gt['keypoints']['labels'])}",
            f"keypoints out of view: {', '.join(l.replace('panda_', '') for l in gt['keypoints'].info['outside_frame']) or 'none'}",
            f"depth range: {gt['depth'].info['min_m']:.2f} - {gt['depth'].info['max_m']:.2f} m",
        ]
        grid = [
            [panel(original, "1 dataset frame (lerobot, 256px upscaled)"),
             panel(frame, "2 replay render = new input frame"),
             panel(difference, "3 |1 - 2| x4")],
            [panel(instances_panel(frame, masks, names), "4 instance_seg: masks"),
             panel(depth_panel(depth, masks, names), "5 depth_perception: metres"),
             panel(robot_panel(frame, gt["keypoints"]["points"], gt["keypoints"]["labels"], gt["pose"]["points"]),
                   "6 keypoints (dots) + pose (ring, arrow)")],
            [panel(grasp_panel(frame, grasp_mask, gt["grasp"]["grasp"]), "7 grasp: object (cyan), fingers (magenta)"),
             panel(grasp_panel(crop(moment), grasp_mask, gt["grasp"]["grasp"], show_mask=False) if moment is not None
                   else text_panel(["video not downloaded"]), "8 dataset frame AT the grasp, same points"),
             panel(text_panel(facts), "9 facts")],
        ]
        sheet = np.vstack([np.hstack(row) for row in grid])
        cv2.imwrite(str(args.out / f"{sample_id}.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        cards.append((sample_id, entry["task"]))
        print(sample_id, file=sys.stderr)

    checklist = "".join(f"<li><b>{html.escape(k)}.</b> {html.escape(v)}</li>" for k, v in CHECKLIST)
    body = "".join(
        f'<section id="{s}"><h2>{i + 1}. {html.escape(s)}</h2><p>{html.escape(t)}</p>'
        f'<label><input type="checkbox" data-id="{s}"> checked</label>'
        f'<a href="{s}.png"><img src="{s}.png" loading="lazy"></a></section>'
        for i, (s, t) in enumerate(cards)
    )
    (args.out / "index.html").write_text(f"""<!doctype html><meta charset="utf-8">
<title>LIBERO ground truth review</title>
<style>body{{font:15px system-ui;margin:24px auto;max-width:1460px;padding:0 16px;background:#fff;color:#111}}
@media (prefers-color-scheme:dark){{body{{background:#161616;color:#eee}}}}
img{{width:100%;border:1px solid #888;margin-top:6px}} section{{margin:36px 0}} h2{{margin-bottom:2px}}
.list{{position:sticky;top:0;background:inherit;padding:8px 0;border-bottom:1px solid #888}}</style>
<h1>LIBERO ground truth review</h1>
<p>Each sheet is what the eval scores against. Click a sheet for full size. Ticks are saved in this browser only.</p>
<ol>{checklist}</ol>
<p class="list" id="count"></p>
{body}
<script>
const boxes=[...document.querySelectorAll('input[data-id]')];
const save=()=>{{try{{localStorage.setItem('libero-review',JSON.stringify(boxes.filter(b=>b.checked).map(b=>b.dataset.id)))}}catch(e){{}}}};
const count=()=>{{document.getElementById('count').textContent=boxes.filter(b=>b.checked).length+' / '+boxes.length+' checked'}};
try{{const done=JSON.parse(localStorage.getItem('libero-review')||'[]');boxes.forEach(b=>b.checked=done.includes(b.dataset.id))}}catch(e){{}}
boxes.forEach(b=>b.addEventListener('change',()=>{{save();count()}}));count();
</script>
""")
    print(f"{len(cards)} sheets -> {args.out}/index.html", file=sys.stderr)


if __name__ == "__main__":
    main()
