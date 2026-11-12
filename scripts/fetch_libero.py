"""Materialize a 20-episode LIBERO subset as a wmgen `frames_dir` + `meta_file`.

https://huggingface.co/datasets/lerobot/libero is a LeRobot v3 repackaging of
simulated Panda demonstrations. What makes it worth having beside RoboTrustBench
is that the end-effector pose is *recorded*, not estimated, so the pose probe can
be scored against the simulator's own answer. `observation.state` is

    [eef_x, eef_y, eef_z, axis_angle_x, axis_angle_y, axis_angle_z, finger_l, finger_r]

in the world frame, read from the same simulator step as the image.

What the dataset does NOT carry is a camera or any joint angle. So:

  - the camera comes from LIBERO's source (`AGENTVIEW` below), one pose per scene
    family, joined to a task through its BDDL file. `fovy` is unset there, so it
    is MuJoCo's default of 45 degrees;
  - joints are not in it either. scripts/libero_replay/ recovers them, and every
    other ground truth, by replaying the original demo in MuJoCo; see
    docs/libero_ground_truth.md. `robot_pose_estimation` gets exact ground truth
    from the recorded state either way;
  - there are no grasp labels, but each demonstration shows one grasp that worked.
    `find_closures` recovers it from the gripper, and `grasp_region_seg` is scored
    against it. It is one valid grasp, not every valid one.

Each sample's `meta` holds `world_to_image`, a 3x4 matrix taking a world point
straight to pixels of the frame written here. Camera and canvas placement are
folded into it, so nothing downstream has to know them.

The stored video is mirrored left-right, and is un-mirrored here. Reversed text
on the milk carton shows it, and the recorded end-effector only projects onto the
gripper under that flip (the 180-degree rotation OpenVLA's notes describe does
not fit). Left mirrored, "the left plate" would name the wrong plate.

Frames are placed on a 1280x720 canvas, not left at 256x256. `prompt.yaml` asks
for 16:9 and wmeval's `_align` rescales x and y independently to the input frame,
so a square frame would silently bend every predicted point. The image is scaled
to the canvas height and centered, with black bars either side.

The frame is drawn from the approach, after the arm has left its home pose and
before the gripper first closes. Opening frames would be one pose scored 20 times,
and a frame from after the grasp would show the object somewhere the recorded grasp
no longer lands. Frames where the arm covers the grasp point are skipped.

Only tasks that name one object to pick first are used (`grasp_target`): tasks that
start with a drawer, a knob or a push have no pick to score, and "put both ..." does
not say which object comes first.

    python scripts/fetch_libero.py --out data/libero
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download, snapshot_download

REPO = "lerobot/libero"
REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
LIBERO_RAW = f"https://raw.githubusercontent.com/Lifelong-Robot-Learning/LIBERO/{LIBERO_REVISION}"
LIBERO_TREE = f"https://api.github.com/repos/Lifelong-Robot-Learning/LIBERO/git/trees/{LIBERO_REVISION}?recursive=1"

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
VIDEO_KEY = "observation.images.image"
FPS = 10.0
SOURCE_SIZE = 256
CANVAS_W, CANVAS_H = 1280, 720
FOVY_DEG = 45.0

# A held grasp. Empty fingers close to under 4 mm, and the gripper has to carry
# something away, which rules out a close that only pressed on the object.
MIN_HELD_GAP = 0.004
MIN_TRAVEL = 0.05
MIN_LIFT = 0.03
SETTLED_STEP = 0.0005  # fingers have stopped once the gap changes less than this per frame
MIN_CLOSING_FRAMES = 3  # the fingers take a few frames to start moving

OCCLUSION_RADIUS = 48  # canvas px around the grasp point, compared against frame 0
MAX_PATCH_CHANGE = 8.0  # mean absolute change, on the 0-255 scale

# If the grasp point is on the object, that spot looks different once the object has
# been carried off. It looks the same if the gripper pushed the object there first,
# because frame 0 then shows bare table. Measured: 1.4 and 2.3 for pushed objects, at
# least 11.7 otherwise.
OBJECT_RADIUS = 14  # canvas px
MIN_OBJECT_CHANGE = 5.0

#: Found by inspecting the ground truth, not by a rule. Decided under the earlier
#: SAM 3 grasp mask; kept so the frozen 20-sample set does not change.
UNUSABLE_EPISODES = {
    1497: "SAM 3 finds no solid object under the grasp: dark bowl rim on a dark cabinet",
}

_TOP_QUAT = (0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349)
_LOW_QUAT = (0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979)

#: `_setup_camera` in libero/libero/envs/problems/, by BDDL problem name. Position
#: then MuJoCo (w, x, y, z) quaternion.
AGENTVIEW = {
    "LIBERO_Tabletop_Manipulation": ((0.6586131746834771, 0.0, 1.6103500240372423), _TOP_QUAT),
    "LIBERO_Kitchen_Tabletop_Manipulation": ((0.6586131746834771, 0.0, 1.6103500240372423), _TOP_QUAT),
    "LIBERO_Study_Tabletop_Manipulation": ((0.4586131746834771, 0.0, 1.6103500240372423), _TOP_QUAT),
    "LIBERO_Living_Room_Tabletop_Manipulation": ((0.6065773716836134, 0.0, 0.96), _LOW_QUAT),
    "LIBERO_Floor_Manipulation": ((0.8965773716836134, 5.216182733499864e-07, 0.65), _LOW_QUAT),
}


def quat_to_matrix(wxyz: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = np.asarray(wxyz, dtype=np.float64) / np.linalg.norm(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def source_intrinsics() -> tuple[float, float]:
    """Focal length and principal point, in pixel-index coordinates of the 256px render."""
    focal = 0.5 * SOURCE_SIZE / np.tan(np.radians(FOVY_DEG) / 2)
    return float(focal), SOURCE_SIZE / 2.0 - 0.5


def canvas_placement() -> tuple[float, float]:
    """Scale and left offset taking a source pixel index to a canvas pixel index."""
    scale = CANVAS_H / SOURCE_SIZE
    return scale, (CANVAS_W - SOURCE_SIZE * scale) / 2.0


def world_to_image(problem: str) -> np.ndarray:
    """3x4 matrix: homogeneous world point -> homogeneous canvas pixel.

    MuJoCo cameras look down -z with +y up, so the render is u = c + f*x/(-z) and
    v = c - f*y/(-z).
    """
    position, quat = AGENTVIEW[problem]
    rotation = quat_to_matrix(quat)
    extrinsic = np.hstack([rotation.T, (-rotation.T @ np.asarray(position)).reshape(3, 1)])

    focal, center = source_intrinsics()
    render = np.array([[focal, 0.0, -center], [0.0, -focal, -center], [0.0, 0.0, -1.0]])  # w = -z

    scale, offset = canvas_placement()
    canvas = np.array([[scale, 0, offset + (scale - 1) / 2], [0, scale, (scale - 1) / 2], [0, 0, 1]])
    return canvas @ render @ extrinsic


def project(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = matrix @ np.append(np.asarray(point, dtype=np.float64), 1.0)
    return homogeneous[:2] / homogeneous[2]


def load_task_scenes() -> dict[str, dict[str, str]]:
    """Lower-cased task string -> {suite, problem, bddl}, from LIBERO's BDDL files."""
    with urllib.request.urlopen(LIBERO_TREE, timeout=60) as response:
        tree = json.load(response)["tree"]

    scenes: dict[str, dict[str, str]] = {}
    for entry in tree:
        match = re.fullmatch(r"libero/libero/bddl_files/(\w+)/(.+\.bddl)", entry["path"])
        if not match or match[1] not in SUITES:
            continue
        with urllib.request.urlopen(f"{LIBERO_RAW}/{entry['path']}", timeout=60) as response:
            text = response.read().decode("utf-8")
        problem = re.search(r"\(problem\s+(\S+?)\)", text)[1]
        # The dataset's task strings follow the file name. The BDDL's own `:language`
        # line is worded differently ("pick the akita black bowl...").
        stem = re.sub(r"^[A-Z_]+SCENE\d+_", "", match[2].removesuffix(".bddl"))
        instruction = stem.replace("_", " ").lower()
        if instruction in scenes:
            raise ValueError(f"instruction appears in two BDDL files: {instruction}")
        scenes[instruction] = {"suite": match[1], "problem": problem, "bddl": match[2]}
    return scenes


def grasp_target(instruction: str) -> str | None:
    """The object an instruction picks first, or None if it does not start with a pick."""
    match = re.match(r"pick up (the .+?) and place it\b", instruction) or re.match(
        r"put (the .+?) (?:on|in|to)\b", instruction
    )
    return match[1] if match else None


def phrase_list(phrases: list[str]) -> str:
    return phrases[0] if len(phrases) == 1 else ", ".join(phrases[:-1]) + " and " + phrases[-1]


def object_concept(target: str) -> str:
    """"the black bowl next to the ramekin" -> "black bowl", for the segmenter."""
    return re.split(r" (?:between|next to|on|in|from)\b", target.removeprefix("the "))[0]


def task_objects(instruction: str) -> list[str]:
    """What an instruction moves and where to, as "the <object>" phrases, in order.

    Written to the meta so the segmentation probes need no LLM here: these instructions
    follow a template, and parsing gives the same list on every run.
    """
    phrases: list[str] = []
    for clause in re.split(r" and (?=put )", instruction):
        target = grasp_target(clause)
        if target is None:
            continue
        rest = clause.split(target, 1)[1].removesuffix(" and close it")
        for phrase in (target, "the " + rest.rsplit(" the ", 1)[1]):
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def find_closures(state: np.ndarray, action: np.ndarray) -> list[dict[str, Any]]:
    """Every time the gripper is commanded shut in one episode, in order.

    `frame` is where the fingers stop moving, so the pose there is the grasp and
    `width` is the object between them. `start` is the frame the command began.
    """
    gap = state[:, 6] - state[:, 7]
    closing = action[:, 6] > 0
    starts = np.flatnonzero(closing & ~np.r_[False, closing[:-1]])
    closures = []
    for start in starts:
        opened = np.flatnonzero(~closing[start:])
        stop = int(start + opened[0]) if len(opened) else len(closing)
        settled = [
            i for i in range(start + MIN_CLOSING_FRAMES, stop) if abs(gap[i] - gap[i - 1]) < SETTLED_STEP
        ]
        frame = settled[0] if settled else stop - 1
        travel = float(np.linalg.norm(state[stop - 1, :3] - state[frame, :3]))
        lift = float(state[frame:stop, 2].max() - state[frame, 2])
        closures.append(
            {
                "start": int(start),
                "frame": int(frame),
                "width": float(gap[stop - 1]),
                "held": bool(gap[stop - 1] >= MIN_HELD_GAP and travel >= MIN_TRAVEL and lift >= MIN_LIFT),
            }
        )
    return closures


def first_grasps(data: pd.DataFrame) -> dict[int, dict[str, Any]]:
    """Episode -> its grasp, for episodes whose very first close held.

    A failed close before the real one may have nudged the object, and the recorded
    grasp would then not land where the object is in the frame shown to the model.
    """
    grasps = {}
    for episode, rows in data.groupby("episode_index"):
        rows = rows.sort_values("frame_index")
        state = np.stack(rows["observation.state"]).astype(np.float64)
        closures = find_closures(state, np.stack(rows["action"]).astype(np.float64))
        if closures and closures[0]["held"]:
            frame = closures[0]["frame"]
            grasps[int(episode)] = {**closures[0], "pos": state[frame, :3], "axis_angle": state[frame, 3:6]}
    return grasps


def select_tasks(
    episodes: pd.DataFrame, by_task: dict[int, dict], grasps: dict[int, dict], per_suite: int, seed: int
) -> list[dict]:
    """`per_suite` tasks from each suite, each with its usable episodes in random order."""
    rng = random.Random(seed)
    task_of = episodes.set_index("episode_index")["task_index"].to_dict()
    pools: dict[int, list[int]] = defaultdict(list)
    for episode in sorted(grasps):
        pools[task_of[episode]].append(episode)

    tasks_by_suite: dict[str, list[int]] = defaultdict(list)
    for task_index, scene in sorted(by_task.items()):
        if scene["target"] and pools[task_index]:
            tasks_by_suite[scene["suite"]].append(task_index)

    chosen = []
    for suite in SUITES:
        for task_index in rng.sample(tasks_by_suite[suite], per_suite):
            pool = list(pools[task_index])
            rng.shuffle(pool)
            chosen.append({"task_index": task_index, "suite": suite, "pool": pool})
    return chosen


def download(path: str, out: Path) -> Path:
    return Path(hf_hub_download(REPO, path, repo_type="dataset", revision=REVISION, local_dir=out))


def read_frames(video: Path, first: int, count: int) -> np.ndarray:
    """`count` frames from `first`, by decode order, as RGB. `-ss` would land on a keyframe's guess."""
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", str(video),
        "-vf", f"select=between(n\\,{first}\\,{first + count - 1})", "-vsync", "0", "-frames:v", str(count),
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    raw = subprocess.run(command, capture_output=True, check=True).stdout
    if len(raw) != count * SOURCE_SIZE * SOURCE_SIZE * 3:
        raise RuntimeError(f"{video}: frames {first}+{count} decoded to {len(raw)} bytes")
    return np.frombuffer(raw, dtype=np.uint8).reshape(count, SOURCE_SIZE, SOURCE_SIZE, 3)


def to_canvas(frame: np.ndarray) -> np.ndarray:
    scale, offset = canvas_placement()
    size = int(round(SOURCE_SIZE * scale))
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    left = int(round(offset))
    canvas[:, left : left + size] = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LANCZOS4)
    return canvas


def clear_frames(frames: list[np.ndarray], point: np.ndarray, candidates: range) -> list[int]:
    """Candidates where the patch around `point` still looks as it did in frame 0."""
    x, y = (int(round(v)) for v in point)
    window = (slice(max(y - OCCLUSION_RADIUS, 0), y + OCCLUSION_RADIUS), slice(max(x - OCCLUSION_RADIUS, 0), x + OCCLUSION_RADIUS))
    reference = frames[0][window].astype(np.float32)
    return [i for i in candidates if np.abs(frames[i][window] - reference).mean() <= MAX_PATCH_CHANGE]


def object_was_there(first: np.ndarray, last: np.ndarray, point: np.ndarray) -> bool:
    """Whether the grasp point sat on the object in `first`, going by `last` without it."""
    x, y = (int(round(v)) for v in point)
    window = (slice(max(y - OBJECT_RADIUS, 0), y + OBJECT_RADIUS), slice(max(x - OBJECT_RADIUS, 0), x + OBJECT_RADIUS))
    return float(np.abs(first[window].astype(np.float32) - last[window]).mean()) >= MIN_OBJECT_CHANGE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path("data/libero"))
    parser.add_argument("--per-suite", type=int, default=5, help="episodes per suite; 4 suites")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-fraction", type=float, default=0.3, help="of the frames before the first close")
    parser.add_argument("--max-fraction", type=float, default=0.9, help="of the frames before the first close")
    args = parser.parse_args()

    raw_dir = args.out / "raw"
    frames_dir = args.out / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for stale in frames_dir.glob("*.png"):  # frames_dir is globbed, so an old pick would be generated for
        stale.unlink()

    tasks = pd.read_parquet(download("meta/tasks.parquet", raw_dir))
    episodes = pd.read_parquet(download("meta/episodes/chunk-000/file-000.parquet", raw_dir))
    episodes = episodes[[c for c in episodes.columns if not c.startswith("stats/")]].copy()

    scenes = load_task_scenes()
    by_task: dict[int, dict] = {}
    for instruction, row in tasks.iterrows():
        key = str(instruction).strip().lower()
        if key not in scenes:
            raise KeyError(f"no BDDL file for task: {instruction}")
        by_task[int(row["task_index"])] = {**scenes[key], "instruction": str(instruction), "target": grasp_target(key)}
    unknown = {s["problem"] for s in by_task.values()} - set(AGENTVIEW)
    if unknown:
        raise KeyError(f"no agentview camera recorded for {sorted(unknown)}")

    # The episode table has no `task_index`, so the task of an episode has to come
    # from the data files: 377 small parquets, about 25 MB in all.
    snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=raw_dir, allow_patterns="data/*")
    data = pd.concat([pd.read_parquet(p) for p in sorted((raw_dir / "data").rglob("*.parquet"))], ignore_index=True)
    episodes["episode_index"] = episodes["episode_index"].astype(int)
    episodes["task_index"] = episodes["episode_index"].map(data.groupby("episode_index")["task_index"].first())

    grasps = first_grasps(data)
    print(f"{len(grasps)} of {len(episodes)} episodes open with a held grasp", file=sys.stderr)

    focal, center = source_intrinsics()
    scale, offset = canvas_placement()
    meta: dict[str, dict] = {}
    for pick in select_tasks(episodes, by_task, grasps, args.per_suite, args.seed):
        scene = by_task[pick["task_index"]]
        matrix = world_to_image(scene["problem"])
        for episode_index in pick["pool"]:
            if episode_index in UNUSABLE_EPISODES:
                continue
            grasp = grasps[episode_index]
            episode = episodes[episodes["episode_index"] == episode_index].iloc[0]
            chunk = int(episode[f"videos/{VIDEO_KEY}/chunk_index"])
            file = int(episode[f"videos/{VIDEO_KEY}/file_index"])
            video = download(f"videos/{VIDEO_KEY}/chunk-{chunk:03d}/file-{file:03d}.mp4", raw_dir)
            start = int(round(float(episode[f"videos/{VIDEO_KEY}/from_timestamp"]) * FPS))
            frames = [to_canvas(f[:, ::-1]) for f in read_frames(video, start, grasp["start"])]
            candidates = range(int(grasp["start"] * args.min_fraction), int(grasp["start"] * args.max_fraction))
            point = project(matrix, grasp["pos"])
            last = read_frames(video, start + int(episode["length"]) - 1, 1)[0]
            if not object_was_there(frames[0], to_canvas(last[:, ::-1]), point):
                print(f"episode {episode_index}: object was moved before the grasp, skipping", file=sys.stderr)
                continue
            clear = clear_frames(frames, point, candidates)
            if clear:
                break
            print(f"episode {episode_index}: arm covers the grasp point throughout, skipping", file=sys.stderr)
        else:
            raise RuntimeError(f"no usable episode for task: {scene['instruction']}")

        # Seeded per episode, so dropping one episode leaves the other samples alone.
        frame_index = random.Random(f"{args.seed}:{episode_index}").choice(clear)
        length = int(episode["length"])
        row = data[(data["episode_index"] == episode_index) & (data["frame_index"] == frame_index)]
        if len(row) != 1:
            raise ValueError(f"episode {episode_index}: {len(row)} rows at frame {frame_index}")
        state = np.asarray(row.iloc[0]["observation.state"], dtype=np.float64)

        phrases = task_objects(scene["instruction"].lower())
        sample_id = f"{pick['suite']}__ep{episode_index:04d}_f{frame_index:03d}"
        cv2.imwrite(str(frames_dir / f"{sample_id}.png"), cv2.cvtColor(frames[frame_index], cv2.COLOR_RGB2BGR))

        meta[sample_id] = {
            "task": scene["instruction"],
            "category": pick["suite"],
            "subcategory": scene["problem"],
            "source_dataset": REPO,
            "source_revision": REVISION,
            "episode_index": int(episode_index),
            "frame_index": frame_index,
            "episode_length": length,
            "bddl": scene["bddl"],
            "eef_pos": [float(v) for v in state[:3]],
            "eef_axis_angle": [float(v) for v in state[3:6]],
            "gripper_qpos": [float(v) for v in state[6:8]],
            "relevant_objects": phrase_list(phrases),
            "gt_prompt": [object_concept(p) for p in phrases],
            "grasp_object": scene["target"],
            "grasp_concept": object_concept(scene["target"]),
            "grasp_frame_index": grasp["frame"],
            "grasp_pos": [float(v) for v in grasp["pos"]],
            "grasp_axis_angle": [float(v) for v in grasp["axis_angle"]],
            "grasp_width": grasp["width"],
            "world_to_image": [[float(v) for v in r] for r in matrix],
            "intrinsics": {
                "fx": focal * scale,
                "fy": focal * scale,
                "cx": center * scale + offset + (scale - 1) / 2,
                "cy": center * scale + (scale - 1) / 2,
            },
        }
        print(f"{sample_id}: {scene['instruction']}", file=sys.stderr)

    (args.out / "meta.yaml").write_text(yaml.safe_dump({"meta": meta}, sort_keys=True), encoding="utf-8")
    print(f"{len(meta)} samples -> {frames_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
