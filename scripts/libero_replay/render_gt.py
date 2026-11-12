"""Replay each LIBERO sample in the simulator and record its exact ground truth.

lerobot/libero is OpenVLA's no-noops re-render: every original LIBERO demo was
replayed from its first simulator state with no-op actions dropped. This script
repeats that replay for the 20 samples in the meta (demo found by
match_demos.py), stops at the sample's frame, and reads everything the five
probes need straight off the simulator:

  rgb        the agentview camera at 720x720. This becomes the conditioning frame,
             so the frame and its ground truth come from one render
  depth      metric depth along the camera axis, in metres
  segment    per-pixel object id: which LIBERO object (or the robot) each pixel
             shows. Instance and grasp masks are cut from it
  keypoints  world positions of the 7 Panda links RoboPEPP and DREAM use
  eef        end-effector position and orientation
  camera     the 3x4 world-to-pixel matrix of the render

and, at the grasp frame, the inner faces of the two finger pads and the grasped
object, read from finger contacts after the gripper closes.

The replay is checked against the dataset as it goes: the replayed end-effector
must stay on the recorded one (`max_eef_error_m`), and the 256x256 render at the
sample's frame must match the dataset frame (`image_error`). A sample that drifts
is reported and not written.

Run in venv_libero (Python 3.10, MuJoCo 2.3.7, robosuite 1.4.1):

    LIBERO_CONFIG_PATH=venv_libero/.libero MUJOCO_GL=cgl \\
        venv_libero/bin/python scripts/libero_replay/render_gt.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml
from robosuite.utils.camera_utils import get_camera_transform_matrix, get_real_depth_map
from robosuite.utils.transform_utils import quat2axisangle

from libero.libero.envs import OffScreenRenderEnv

BDDL_ROOT = Path("third_party/LIBERO/libero/libero/bddl_files")
CAMERA = "agentview"
SIZE = 720  # rendered square; it fills the height of the 1280x720 canvas
CANVAS = (1280, 720)
OFFSET = (CANVAS[0] - SIZE) // 2
DATASET_SIZE = 256
DUMMY_ACTION = [0, 0, 0, 0, 0, 0, -1]

# RoboPEPP / DREAM's Panda keypoints, and the robosuite bodies at the same frames.
KEYPOINTS = {
    "panda_link0": "robot0_link0",
    "panda_link2": "robot0_link2",
    "panda_link3": "robot0_link3",
    "panda_link4": "robot0_link4",
    "panda_link6": "robot0_link6",
    "panda_link7": "robot0_link7",
    "panda_hand": "robot0_right_hand",
}
ROBOT_PREFIXES = ("robot0_", "gripper0_", "mount0_")
FINGER_PREFIX = "gripper0_"
PAD_GEOMS = ("gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision")


def object_of_body(name: str, objects: list[str]) -> str | None:
    if name.startswith(ROBOT_PREFIXES):
        return "robot"
    matches = [o for o in objects if name == o or name.startswith(o + "_")]
    return max(matches, key=len) if matches else None


def segment_ids(sim, objects: list[str], names: list[str]) -> np.ndarray:
    """(H, W) index into `names` of the object each pixel shows; 0 is anything else."""
    raw = sim.render(camera_name=CAMERA, width=SIZE, height=SIZE, segmentation=True)[::-1]
    kind, ident = raw[..., 0], raw[..., 1]
    model = sim.model
    geom_owner = np.zeros(model.ngeom, dtype=np.int16)
    for geom in range(model.ngeom):
        owner = object_of_body(model.body_id2name(int(model.geom_bodyid[geom])), objects)
        geom_owner[geom] = names.index(owner) if owner in names else 0
    out = np.zeros(kind.shape, dtype=np.int16)
    is_geom = (kind == 5) & (ident >= 0)  # mjOBJ_GEOM
    out[is_geom] = geom_owner[ident[is_geom]]
    return out


def held_object(sim, objects: list[str]) -> list[str]:
    """Objects touching a finger geom right now."""
    model, data = sim.model, sim.data
    touched = []
    for i in range(data.ncon):
        contact = data.contact[i]
        bodies = [model.body_id2name(int(model.geom_bodyid[g])) for g in (contact.geom1, contact.geom2)]
        if not any(b.startswith(FINGER_PREFIX) for b in bodies):
            continue
        for body in bodies:
            owner = object_of_body(body, objects)
            if owner not in (None, "robot"):
                touched.append(owner)
    return touched


def pad_faces(sim) -> np.ndarray:
    """(2, 3) world points: the inner face of each finger pad, where it meets the object.

    Each pad is a thin box whose local y is the closing axis; its inner face is
    the centre moved half the box's thickness towards the other pad. The two
    fingers are driven separately, so an object can push them off the grip
    site's centre: these, not the grip site +- half the gap, are where they are.
    """
    ids = [sim.model.geom_name2id(name) for name in PAD_GEOMS]
    centres = [sim.data.geom_xpos[i].copy() for i in ids]
    faces = []
    for this, (i, centre) in enumerate(zip(ids, centres)):
        axis = sim.data.geom_xmat[i].reshape(3, 3)[:, 1]
        towards = np.sign(float(axis @ (centres[1 - this] - centre))) or 1.0
        faces.append(centre + towards * float(sim.model.geom_size[i][1]) * axis)
    return np.stack(faces)


def world_to_pixel(sim) -> np.ndarray:
    """3x4 matrix from world points to (x, y) pixels of the upright SIZE render.

    robosuite's extrinsics already use the OpenCV convention (y down), so this
    lands on the image as it is shown, not as MuJoCo stores it bottom row first.
    Checked by projecting the links onto the arm in the segmentation render.
    """
    return np.asarray(get_camera_transform_matrix(sim, CAMERA, SIZE, SIZE)[:3], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/libero"))
    parser.add_argument("--max-eef-error", type=float, default=0.01, help="metres")
    parser.add_argument("--max-image-error", type=float, default=12.0, help="mean abs, 0-255")
    parser.add_argument("--settle-steps", type=int, default=-1,
                        help="dummy steps after set_init_state; -1 tries 0 and 10 and keeps the closer")
    parser.add_argument("--only", nargs="*", help="sample ids")
    args = parser.parse_args()

    meta_path = args.data / "meta.yaml"
    meta = yaml.safe_load(meta_path.read_text())["meta"]
    demos = yaml.safe_load((args.data / "demo_map.yaml").read_text())
    data = pd.concat([pd.read_parquet(p, columns=["episode_index", "frame_index", "observation.state"])
                      for p in sorted((args.data / "raw" / "data").rglob("*.parquet"))])

    sim_dir = args.data / "sim"
    sim_dir.mkdir(exist_ok=True)
    original_dir = args.data / "frames_lerobot"
    original_dir.mkdir(exist_ok=True)

    report_path = args.data / "sim_report.yaml"
    report = yaml.safe_load(report_path.read_text()) if report_path.exists() else {}
    for sample_id, entry in sorted(meta.items()):
        if args.only and sample_id not in args.only:
            continue
        if sample_id not in demos:
            print(f"{sample_id}: no matched demo, skipped", file=sys.stderr)
            continue
        source = demos[sample_id]
        with np.load(source["cache"]) as cached:  # written by match_demos.py
            first_state = cached["first_state"]
            actions = cached["actions"]

        recorded = data[data.episode_index == entry["episode_index"]].sort_values("frame_index")
        recorded = np.stack(recorded["observation.state"].to_numpy())
        frame = int(entry["frame_index"])
        grasp_frame = int(entry["grasp_frame_index"])
        last = min(len(actions), grasp_frame + 15)

        bddl = BDDL_ROOT / entry["category"] / entry["bddl"]
        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=DATASET_SIZE,
                                 camera_widths=DATASET_SIZE)
        env.seed(0)

        def replay(settle: int, stop: int, on_step=None) -> float:
            env.reset()
            obs = env.set_init_state(first_state)
            for _ in range(settle):
                obs, *_ = env.step(DUMMY_ACTION)
            error = 0.0
            for step in range(stop + 1):
                error = max(error, float(np.linalg.norm(obs["robot0_eef_pos"] - recorded[step, :3])))
                if on_step is not None:
                    on_step(step, obs)
                if step < stop:
                    obs, *_ = env.step(actions[step].tolist())
            return error

        options = [0, 10] if args.settle_steps < 0 else [args.settle_steps]
        trials = {s: replay(s, min(frame + 20, len(actions) - 1)) for s in options}
        settle = min(trials, key=trials.get)

        capture: dict = {}
        objects = list(env.env.objects_dict) + list(env.env.fixtures_dict)
        names = ["none", "robot", *objects]
        touches: list[str] = []

        def on_step(step: int, obs) -> None:
            if step == frame:
                sim = env.sim
                rgb = sim.render(camera_name=CAMERA, width=SIZE, height=SIZE)[::-1]
                _, depth_buffer = sim.render(camera_name=CAMERA, width=SIZE, height=SIZE, depth=True)
                depth = get_real_depth_map(sim, depth_buffer[::-1])
                capture.update(
                    rgb=np.ascontiguousarray(rgb),
                    depth=np.asarray(depth, dtype=np.float32).reshape(SIZE, SIZE),
                    segment=segment_ids(sim, objects, names),
                    camera=world_to_pixel(sim),
                    keypoints_world=np.stack([sim.data.body_xpos[sim.model.body_name2id(b)] for b in KEYPOINTS.values()]),
                    eef_pos=np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
                    eef_axis_angle=np.asarray(quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float64),
                    gripper_qpos=np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
                    joint_pos=np.asarray(obs["robot0_joint_pos"], dtype=np.float64),
                    dataset_render=np.ascontiguousarray(obs["agentview_image"][::-1]),
                )
            if step == grasp_frame:
                capture["grasp_fingers_world"] = pad_faces(env.sim)
            if grasp_frame <= step <= last:
                touches.extend(held_object(env.sim, objects))

        eef_error = replay(settle, last, on_step)
        # BDDL names a region where the task targets part of a fixture
        # ("wooden_cabinet_1_top_side"); the instance is the object it belongs to.
        interest = []
        for name in env.env.obj_of_interest:
            owner = object_of_body(name, objects)
            if owner not in (None, "robot") and owner not in interest:
                interest.append(owner)
        env.close()

        # The dataset frame, un-mirrored by fetch_libero, vs this replay's 256 render.
        original = cv2.cvtColor(cv2.imread(str(args.data / "frames" / f"{sample_id}.png")), cv2.COLOR_BGR2RGB)
        if not (original_dir / f"{sample_id}.png").exists():
            cv2.imwrite(str(original_dir / f"{sample_id}.png"), cv2.cvtColor(original, cv2.COLOR_RGB2BGR))
        else:
            original = cv2.cvtColor(cv2.imread(str(original_dir / f"{sample_id}.png")), cv2.COLOR_BGR2RGB)
        crop = cv2.resize(original[:, OFFSET:OFFSET + SIZE], (DATASET_SIZE, DATASET_SIZE), interpolation=cv2.INTER_AREA)
        image_error = float(np.abs(crop.astype(float) - capture["dataset_render"].astype(float)).mean())

        grasped = max(set(touches), key=touches.count) if touches else None
        ok = eef_error <= args.max_eef_error and image_error <= args.max_image_error and grasped is not None
        report[sample_id] = {
            "demo": source["demo"],
            "settle_steps": settle,
            "max_eef_error_m": round(eef_error, 5),
            "image_error": round(image_error, 2),
            "grasped_object": grasped,
            "obj_of_interest": interest,
            "ok": bool(ok),
        }
        print(f"{sample_id}: {report[sample_id]}", file=sys.stderr, flush=True)
        if not ok:
            continue

        # Canvas: the 720x720 render centred on 1280x720, as fetch_libero places frames.
        canvas = np.zeros((CANVAS[1], CANVAS[0], 3), dtype=np.uint8)
        canvas[:, OFFSET:OFFSET + SIZE] = capture["rgb"]
        cv2.imwrite(str(args.data / "frames" / f"{sample_id}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        shift = np.array([[1, 0, OFFSET], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        to_canvas = shift @ capture["camera"]

        def on_canvas(layer: np.ndarray, fill) -> np.ndarray:
            out = np.full((CANVAS[1], CANVAS[0]), fill, dtype=layer.dtype)
            out[:, OFFSET:OFFSET + SIZE] = layer
            return out

        keypoints_world = capture["keypoints_world"]
        homogeneous = np.hstack([keypoints_world, np.ones((len(keypoints_world), 1))]) @ to_canvas.T
        fingers_world = capture["grasp_fingers_world"]
        np.savez_compressed(
            sim_dir / f"{sample_id}.npz",
            depth=on_canvas(capture["depth"], np.float32(np.nan)),
            segment=on_canvas(capture["segment"], np.int16(0)),
            object_names=np.array(names),
            world_to_image=to_canvas,
            keypoints_world=keypoints_world,
            keypoints=(homogeneous[:, :2] / homogeneous[:, 2:3]).astype(np.float32),
            keypoint_labels=np.array(list(KEYPOINTS)),
            eef_pos=capture["eef_pos"],
            eef_axis_angle=capture["eef_axis_angle"],
            gripper_qpos=capture["gripper_qpos"],
            joint_pos=capture["joint_pos"],
            grasp_fingers_world=fingers_world,
        )

        # The meta now describes the rendered frame: its camera, its robot state.
        entry["world_to_image"] = to_canvas.tolist()
        entry["eef_pos"] = capture["eef_pos"].tolist()
        entry["eef_axis_angle"] = capture["eef_axis_angle"].tolist()
        entry["gripper_qpos"] = capture["gripper_qpos"].tolist()
        entry["sim_file"] = f"sim/{sample_id}.npz"
        entry["sim_objects"] = interest
        entry["sim_grasp_object"] = grasped
        entry["sim_demo"] = f"{source['url']}#{source['demo']}"

    meta_path.write_text(yaml.safe_dump({"meta": meta}, sort_keys=True))
    report_path.write_text(yaml.safe_dump(report, sort_keys=True))
    print(f"{sum(r['ok'] for r in report.values())}/{len(report)} samples ok", file=sys.stderr)


if __name__ == "__main__":
    main()
