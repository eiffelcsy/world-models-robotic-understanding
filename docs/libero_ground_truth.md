# Ground truth for the LIBERO evaluation set: rationale, procedure and validation

This document describes how the ground truth for the five LIBERO tasks is obtained:
why each choice was made, every step of the procedure, and how each result was
checked. It is written so that the procedure can be audited and repeated. The code
it describes is in `scripts/fetch_libero.py`, `scripts/libero_replay/`,
`wmeval/simulator.py` and each probe's `gt.py`. Commands to reproduce every step
are in [Appendix B](#appendix-b-reproducing-every-step).

**Contents**

1. [Summary](#1-summary)
2. [Rationale](#2-rationale)
3. [Data provenance](#3-data-provenance)
4. [Procedure](#4-procedure)
5. [Ground truth per task](#5-ground-truth-per-task)
6. [Validation](#6-validation)
7. [Design decisions](#7-design-decisions)
8. [Limitations and threats to validity](#8-limitations-and-threats-to-validity)
9. [Relation to RoboTrustBench](#9-relation-to-robotrustbench)
- [Appendix A: per-sample record](#appendix-a-per-sample-record)
- [Appendix B: reproducing every step](#appendix-b-reproducing-every-step)

---

## 1. Summary

The LIBERO set has 20 conditioning frames: 4 suites, 5 tasks each, one episode per
task. Each frame is replayed in the MuJoCo simulator from the original LIBERO
demonstration that produced it. The conditioning frame and every ground-truth layer
are then read from that single replayed simulator state.

| Task (probe) | Ground truth | Read from | Exact? |
|---|---|---|---|
| Instance segmentation (`instance_seg`) | one binary mask per object the task names | MuJoCo segmentation render, geom → body → object | yes, per pixel |
| Depth perception (`depth_perception`) | metric depth (m) along the camera axis, per pixel | MuJoCo depth buffer, linearised with the camera's near/far planes | yes, per pixel |
| Robot keypoint detection (`robot_keypoint_detection`) | image positions of 7 Panda links | MuJoCo body frame positions, projected with the render's camera | yes, given the keypoint definition (§5.3) |
| Robot pose estimation (`robot_pose_estimation`) | end-effector point and a heading point 8 cm ahead | recorded end-effector pose, projected | yes |
| Grasp region (`grasp_region_seg`) | the grasp the demonstration made, on the exact mask of the object it picked | the replay's finger pads at the grasp + MuJoCo contacts + segmentation | the grasp is exact; it is one valid grasp, not all (§5.5) |

No learned model (SAM 3, Depth Anything 3, RoboPEPP or any other) is involved in
any LIBERO ground truth. The two checks that bound the replay's accuracy both
pass on all 20 samples:

- the replayed end-effector stays within **0.3 mm** of the dataset's recorded one
  at every step (18 of 20 samples: exactly 0);
- the replay's 256 px render matches the dataset frame to a mean absolute
  difference of **3.3–7.8 / 255** (video compression).

---

## 2. Rationale

### 2.1 Why exact ground truth matters here

The evaluation asks a video world model to show its scene understanding on screen,
for example by repainting the scene as a depth map, and then scores what it
paints. A score is only as meaningful as the reference it is compared against.
When the reference is another model's estimate (pseudo ground truth), the score
measures agreement with that model, and that model's errors are in the reference.
They cannot be told apart from the world model's errors. Before this work, the
LIBERO ground truth was also estimated: SAM 3 for masks, Depth Anything 3 (DA3)
for depth, and RoboPEPP was planned for keypoints. A by-eye review then found:

- SAM 3 missed a named object on 8 of 20 frames, kept a same-class distractor on 4,
  and split one cabinet into 5 instances;
- DA3 depth is relative (unknown scale and shift) and has its own errors on
  textureless simulated surfaces;
- the dataset contains no joint angles, so no keypoint ground truth was possible at
  all, and the task had been cut.

Hand-picked SAM 3 point prompts fixed the masks, but they remained an estimate
with a human in the loop. Simulator replay removes the estimation step entirely.

### 2.2 Why a simulator replay is possible

`lerobot/libero` stores only images, actions and an 8-D proprioceptive state. It
has no simulator states, no camera, no depth, no segmentation and no joint angles.
It is, however, an exact derivative of something that does have the states:

- The original LIBERO release (`yifengzhu-hf/LIBERO-datasets`) holds 50
  human-teleoperated demonstrations per task as HDF5. It includes the full MuJoCo
  state at every step and the actions.
- OpenVLA re-rendered those demonstrations. Each demo was reset to its first
  recorded state, waited a few steps with a "do nothing" action, and replayed with
  no-op actions dropped. Only successful replays were kept, and the 256x256 images
  were re-rendered. The four suite sizes in `lerobot/libero` (432 / 454 / 428 /
  379, total 1693) are exactly those of OpenVLA's "no-noops" release.

So each `lerobot/libero` episode corresponds to one original demo. Repeating
OpenVLA's replay from that demo's first state recovers the simulator state behind
every frame. Two measurements guard this claim: the recorded end-effector and the
image (§4.4).

### 2.3 What "exact" means, and its limit

The ground truth is exact with respect to the simulator: the depth is the depth of
the rendered geometry, the masks are the rendered pixels of each object, and so on.
Since the conditioning frame given to the world model is rendered from the same
state by the same renderer (§4.5), frame and labels share one source and there is
no domain gap between them. "Exact" does not extend to
*definitional* choices, such as which point of a robot link counts as its keypoint,
or which grasp counts as the right one. These are stated explicitly in §5 and §8.

---

## 3. Data provenance

| Item | Source | Version |
|---|---|---|
| Episodes, actions, recorded state, video | `huggingface.co/datasets/lerobot/libero` | revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| Original demos (first state, actions) | `huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets` | revision `f13aa24a3da8c43c7225569f28c562979fa0e35a` (last modified 2025-05-18) |
| Task definitions (BDDL), scene XML, assets | `github.com/Lifelong-Robot-Learning/LIBERO` | commit `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| Physics and rendering | MuJoCo | 2.3.7 |
| Robot / environment layer | robosuite | 1.4.1 |
| Python for the replay | CPython | 3.10 (`venv_libero`) |

`lerobot/libero`'s `observation.state` is
`[eef_x, eef_y, eef_z, axis_angle (3), finger_left, finger_right]` in the world
frame. Its video is stored mirrored left–right, and §4.1 undoes this.

---

## 4. Procedure

The pipeline has six stages. Stages A, C and the eval run in the main environment;
B–E in the replay environment.

```
A select samples ──▶ B environment ──▶ C match demos ──▶ D replay + validate
                                                              │
     F ground-truth sources ◀── E render + record ◀───────────┘
```

### 4.1 Stage A — sample selection (`scripts/fetch_libero.py`)

Selection was fixed before the replay existed, and the 20 samples are frozen. Every
rule below exists so that a frame has a well-defined answer for all five tasks.

1. **Tasks that start with a single pick.** The instruction must match
   `pick up <obj> and place it …` or `put <obj> on|in|to …` (`grasp_target`).
   Tasks that start with a drawer, a knob or a push have no pick to score.
   "Put both …" does not say which object comes first.
2. **Episodes whose first gripper close is a held grasp** (`find_closures`). A
   close begins where the gripper action turns positive. The *grasp frame* is the
   first frame, at least 3 frames into the close, at which the finger gap changes
   by less than 0.5 mm per frame (the fingers have stopped on the object). The
   close counts as *held* if, by the end of the close, the fingers are at least
   4 mm apart (empty fingers close below that), the gripper has travelled at least
   5 cm, and it has risen at least 3 cm. Only episodes whose **first** close held
   are used: a failed attempt could have moved the object, and then the recorded
   grasp would not land on the object in an earlier frame.
3. **Five tasks per suite, one episode per task.** With `random.Random(seed=0)`,
   5 eligible tasks are drawn per suite. Within each task the eligible episodes are
   shuffled, and the first that passes steps 4–5 is used.
4. **The object must be at the grasp point before the grasp.** The grasp position
   is projected into the frame. A 28 px patch around it must differ between the
   first and the last frame of the episode by at least 5/255 (mean absolute). If
   the gripper pushed the object to that spot before closing, frame 0 shows bare
   table there and the patch does not change. Measured: 1.4 and 2.3 for pushed
   objects, at least 11.7 otherwise.
5. **A conditioning frame from the approach, with the grasp point visible.** The
   candidates are frames in [30 %, 90 %) of the frames before the first close.
   Opening frames would score one robot pose 20 times, and frames after the grasp
   show the object somewhere the recorded grasp no longer lands. Among the
   candidates, frames where the arm covers the grasp point are dropped: the 96 px
   patch around it must stay within 8/255 of frame 0. One frame is drawn per
   episode with a generator seeded by `"0:<episode>"`, so dropping one episode
   leaves the others unchanged.
6. **One manual exclusion.** Episode 1497 was excluded after inspection, at a time
   when grasp masks still came from SAM 3 (no solid mask under the grasp: a dark
   bowl rim on a dark cabinet). It is kept excluded so that the frozen set does not
   change; with simulator masks it would probably be usable.
7. **Image geometry.** Frames are un-mirrored. Reversed text on the milk carton
   shows the mirroring, and the recorded end-effector projects onto the gripper
   only after this flip. Frames are then placed on a 1280x720 canvas (scaled to the
   canvas height and centred, with black side bars), because the world model's
   prompts ask for 16:9. A square frame would be stretched, and the evaluator
   rescales x and y separately, which would distort every point.
8. **Recorded quantities written to the sample's meta:**
   - the end-effector state at the frame;
   - the grasp frame and the recorded grasp pose and finger gap (the replay
     later replaces these with the finger pads' measured positions, §5.5);
   - `grasp_frame_index`;
   - the BDDL file, the instruction, and the object phrases parsed from it.

### 4.2 Stage B — replay environment

LIBERO pins an older stack (MuJoCo 2.3.7, robosuite 1.4.1, Python ≤ 3.10) that
cannot be installed beside the main environment (Python 3.14). The replay therefore
has its own environment, `venv_libero`, used offline only. Its outputs are plain
`.npz` and YAML files that the main environment reads. Two local fixes were
needed:

- MuJoCo 2.3.7's macOS OpenGL path no longer exists on current macOS, and was
  corrected in `mujoco/cgl/cgl.py`;
- LIBERO's first-import prompt was replaced with a config file
  (`venv_libero/.libero/config.yaml`).

The simulator's physics and rendering were not changed.

### 4.3 Stage C — matching each episode to its original demo (`match_demos.py`)

`lerobot/libero` does not record which original demo an episode came from. The
episode's index within its task is not the demo index (for example, episode 1520
is `demo_13`, not demo 30). The match is made on actions:

1. Apply OpenVLA's no-op rule to each candidate demo's actions. Step *i* is a
   no-op if `‖a_i[:6]‖ < 1e-4` and its gripper command equals that of the last
   kept step. For the first step, only the first condition applies.
2. A candidate matches if it keeps exactly as many steps as the episode has, and
   the maximum absolute difference between its kept actions and the episode's
   actions is the smallest of all candidates (it stops early below 1e-6).
3. Only the `actions` arrays are read, over HTTP range requests. The HDF5 files are
   0.5–2 GB, almost all images, and only a few kilobytes are needed.
4. The matched demo's first MuJoCo state and its kept actions are cached in
   `data/libero/demos/<sample>.npz`, with a record in `data/libero/demo_map.yaml`.

**Result.** All 20 samples matched one demo, with a maximum action difference of
2.9e-8. That is float32 rounding, so the matches are exact and unambiguous.

### 4.4 Stage D — replay and its validation (`render_gt.py`)

For each sample:

1. Build the task environment from its BDDL file (`OffScreenRenderEnv`,
   seed 0), `reset()`, then `set_init_state(first_state)`.
2. Run *s* settle steps of the dummy action `[0,0,0,0,0,0,−1]` (no motion,
   gripper open). Then step through the kept actions. After the settle steps,
   replay step *k* corresponds to dataset frame *k*.
3. **Choose *s* by evidence.** The script replays with *s* = 0 and *s* = 10 and
   keeps the one whose end-effector track is closer to the recording. *s* = 10
   won on all 20 samples, which agrees with OpenVLA's re-render.
4. **Check 1, the trajectory.** At every step from 0 to grasp frame + 15, the
   distance between the replayed `robot0_eef_pos` and the dataset's recorded
   position must stay ≤ 1 cm. The tolerance is loose; the observed worst case is
   0.3 mm, and 18 samples are exactly 0.
5. **Check 2, the image.** At the sample's frame, the replay's 256x256 render
   (the dataset's own resolution) must match the dataset frame to a mean absolute
   difference ≤ 12/255. Observed: 3.32–7.76. This is a check of the scene, not
   only the robot: it would fail if an object were misplaced.
6. A sample that fails either check, or for which no grasped object is found
   (§5.5), is reported and not written. None failed.

Each check has a clear weakness on its own. The end-effector track says nothing
about the objects, and the image check is coarse at 256 px. Together they confirm
the robot state and the scene.

### 4.5 Stage E — rendering and recording

At the sample's frame, the `agentview` camera is rendered at **720x720**. This is
the same camera and the same state as the dataset frame, but at the canvas height,
so no frame is upscaled. Four layers are taken from this one simulator state:

| Layer | How | Convention |
|---|---|---|
| RGB | `sim.render(camera_name="agentview", 720, 720)` | MuJoCo returns rows bottom-up; flipped with `[::-1]` |
| Depth | depth buffer from the same call → `robosuite.get_real_depth_map` | linear depth along the optical axis (not ray length), metres |
| Segmentation | `sim.render(..., segmentation=True)` → (object type, id) per pixel | type 5 = geom; geom → body → object (below) |
| Camera | `robosuite.get_camera_transform_matrix(sim, "agentview", 720, 720)[:3]` | already OpenCV (y down), so no flip; checked by projecting the links onto the arm |

**Geom to object.** Each geom's body name is mapped to an object by prefix. LIBERO
names bodies after their object (`akita_black_bowl_1_main` →
`akita_black_bowl_1`). When several prefixes match, the longest wins. Bodies
starting `robot0_`, `gripper0_` or `mount0_` are the robot. Anything else is
background (id 0).

**Also recorded at the frame:**

- world positions of the 7 keypoint bodies;
- `robot0_eef_pos` and `robot0_eef_quat` (converted to axis–angle);
- gripper and joint positions.

**Grasped object.** The replay continues to the grasp frame + 15 steps. At each
step from the grasp frame on, it records which objects are in contact with a finger
geom (any geom of a `gripper0_` body). The grasped object is the one touched most
often.

**Objects of interest.** These are the BDDL task's `obj_of_interest`. Where BDDL
names a region of a fixture (`wooden_cabinet_1_top_side`), the region is mapped to
the fixture that owns it.

**Canvas.** The RGB, depth (NaN off the image) and segmentation (0 off the image)
are placed on the 1280x720 canvas at x offset 280. The camera matrix is
left-multiplied by the matching translation, so it maps world points directly to
canvas pixels. The new RGB frame **replaces** the conditioning frame. The original
dataset frames are kept in `data/libero/frames_lerobot/`.

**Outputs:**

- `data/libero/sim/<sample>.npz`: depth, segment, object names, camera, keypoints
  (world and canvas), end-effector, gripper, joints;
- the sample's meta gains `sim_file`, `sim_objects`, `sim_grasp_object` and
  `sim_demo`, and its `world_to_image`, `eef_pos`, `eef_axis_angle` and
  `gripper_qpos` are replaced by the replay's values;
- `data/libero/sim_report.yaml` records both checks per sample.

### 4.6 Stage F — ground-truth sources and storage

Each probe has a ground-truth class that reads the record (`wmeval/simulator.py`)
and returns the probe's data type. Keypoints, pose and grasp use their simulator
source by default. `configs/eval.libero.yaml` switches segmentation and depth from
their RoboTrustBench defaults (SAM 3, DA3) to the simulator. The results are
written once to `results/libero/<store>/<sample>_gt.npz`, each with a `.png`
preview beside it. The eval reads these files and never re-derives them.

| Probe | Source class | Store |
|---|---|---|
| `instance_seg` | `sim_instances` | `instance_seg_sim_gt` |
| `depth_perception` | `sim_depth` | `depth_perception_sim_gt` |
| `robot_keypoint_detection` | `sim_keypoints` | `robot_keypoints_sim_gt` |
| `robot_pose_estimation` | `recorded_pose` (`heading_distance` 0.08 m) | `robot_pose_recorded_gt` |
| `grasp_region_seg` | `demonstrated_grasp` | `grasp_demonstrated_gt` |

---

## 5. Ground truth per task

### 5.1 Instance segmentation

**What the model is asked.** Repaint each object the instruction is about in its
own flat colour. The decoded colour regions are matched one-to-one to the ground
truth and scored by IoU (`mean_iou` headline).

**Ground truth.** One boolean mask per object in the task's `obj_of_interest`
(`sim_instances`). The mask is every canvas pixel whose segmentation id is that
object. There are 2 objects on 18 frames, 3 on one, and 4 on one.

**Procedure.** `object_mask(record, name) = (segment == index(name))`. An object
with no visible pixels would be dropped and listed in `hidden`. None was.

**Design decisions.**

- *Which objects.* The objects come from `obj_of_interest`, the objects the task's
  success condition refers to, rather than from parsing the instruction text. This
  is the task author's own declaration, and it resolves references such as "the
  black bowl next to the plate" to the specific bowl body. Free-text phrases cannot
  do that. On all 20 samples the set agrees with the instruction's object phrases.
- *Visible pixels only.* An object partly behind the arm keeps only its visible
  pixels. The model is asked to repaint what is on screen, so amodal masks would
  penalise it for not painting through the arm.
- *Fixtures by owner.* A region named in BDDL (the top of the cabinet) becomes the
  whole fixture. The image shows the fixture, not an invisible region.

**Validation.**

- On every sheet in `results/libero/review/`, each mask covers its whole object up
  to the edge, and the label matches the instruction.
- The segmentation and RGB come from one render, so their alignment is exact by
  construction.

### 5.2 Depth perception

**What the model is asked.** Replace the frame with a grayscale depth map, near
bright and far dark. The decoded depth is fitted to the ground truth (least-squares
scale and shift) and scored with the standard monocular-depth metrics (`delta1`
headline).

**Ground truth.** Metric depth in metres along the camera's optical axis for every
pixel of the 720x720 render, 0.55–3.07 m over the 20 frames (`sim_depth`). The side
bars are NaN and excluded from scoring.

**Procedure.** MuJoCo's depth buffer stores non-linear values *z_b* in [0, 1].
robosuite's `get_real_depth_map` linearises them with the model's near and far
clip planes, *d = n / (1 − z_b (1 − n/f))*.

**Design decisions.**

- *Optical-axis depth, not ray distance.* This is the convention of the
  monocular-depth benchmarks the metrics come from.
- *Metric although the scoring is affine-invariant.* The model's output has no
  scale, so the metric fits scale and shift either way. A metric reference makes
  that fit well-posed and exact. DA3 is itself relative, so with DA3 the fit was
  done against a reference that was also only known up to an affine map.

**Validation.**

- Depth and camera were checked against each other, independently of the depth
  conversion. Each robot link inside the image was projected with the camera
  matrix. Its camera-frame *z* (the third row of the matrix) was compared with the
  depth map at that pixel. For a visible link, the surface must be slightly nearer
  than the link's internal frame origin.
- Over 80 in-view links, *z − depth* is never negative (minimum 4.1 cm). On 58 it
  is 4–10 cm, about a link's radius. On 10 (`link6`) it is 10.5–13.1 cm, where that
  frame's origin sits deep inside the wrist housing. On 12 it is 32–81 cm: those
  links are occluded (§5.3).
- A wrong scale, wrong near/far planes or a misaligned camera would break this
  pattern.
- By eye, depth increases away from the camera on every sheet, and per-object
  median depths are ordered as they look.

### 5.3 Robot keypoint detection

**What the model is asked.** Place a coloured dot on each robot joint. The dots are
matched to the ground truth and scored by PCK at 5 % of the frame diagonal.

**Ground truth.** Image positions of the 7 Panda keypoints used by DREAM and
RoboPEPP: `link0`, `link2`, `link3`, `link4`, `link6`, `link7` and `hand`
(`sim_keypoints`). They come from the robosuite bodies `robot0_link0…7` and
`robot0_right_hand`. Only keypoints that project inside the rendered square are
kept. The `agentview` camera cuts off the upper arm, so 2–5 are in view per frame
(mean 4.0; 80 in total). In view: `link0` and `hand` on 20/20 frames, `link7` on
18, `link6` on 14, `link2` on 8, and `link3`/`link4` on none.

**Procedure.**

1. Read `sim.data.body_xpos` (each body frame's origin in the world) at the frame.
2. Project it with the canvas camera matrix.
3. Drop points outside the 720 px square or on a NaN-depth pixel, and list them in
   `outside_frame`.

**Design decisions.**

- *Keypoint definition.* Using DREAM/RoboPEPP's link set and frame origins keeps
  the definition comparable with the robot-pose literature, and makes it
  unambiguous. A keypoint is a link's *joint frame origin*, not the centre of the
  visible link. Two consequences follow:
  - `link0` lies at the robot base, where the base meets the mount;
  - `link6`'s origin sits about 10–13 cm inside the wrist along the view ray.
  So a dot placed on the visible "joint bump" may be a few pixels off these
  points.
- *Occluded keypoints are kept.* The prompt asks for a dot on *every* joint. A
  joint behind the arm or the table still has a true image position, and
  dropping it would reward a model for leaving it out. The depth test in §5.2
  finds 12 of the 80 in-view keypoints occluded: `link0` on 7 frames (behind the
  table edge or a fixture) and `link2` on 5 (behind the arm). See §8 for the
  alternative.
- *Out-of-view keypoints are dropped.* They have no pixel position on the canvas
  the model draws on.

**Validation.**

- The camera matrix's intrinsics, recovered by RQ decomposition, are
  *f* = 869.12 px with principal point (640, 360) on the canvas. That is exactly
  360 / tan(22.5°) for the 45° `agentview` field of view.
- The camera at selection time, rebuilt from LIBERO's source, and the replay's
  camera project the grasp points to within 0.71 px of each other. That is
  precisely the half-pixel convention difference between the two.
- By eye, the dots lie on the arm in kinematic order on every sheet.

### 5.4 Robot pose estimation

**What the model is asked.** Draw a red dot at the gripper and a green dot a short
way ahead along its pointing direction. The score combines the centre position
error and the heading angle (`pose_score`).

**Ground truth.** Two image points (`recorded_pose`): the end-effector position,
and the point 8 cm ahead of it along the gripper's approach axis (its local +z).

**Procedure.**

1. Read `eef_pos` and `eef_axis_angle` (robosuite's `robot0_eef_pos` and
   `robot0_eef_quat`, the grip site between the fingers) from the replay.
2. Convert the axis–angle to a rotation *R*.
3. Project *p* and *p* + 0.08·*R*[:, 2] with the canvas camera.

**Design decisions.**

- *Recorded, not estimated.* The pose is the simulator's own state. It equals the
  dataset's recorded state to within 0.3 mm, so it is an exact reference.
- *Grip site as the end-effector.* This is the point the controller tracks and the
  dataset records, and the prompt asks for the gripper's centre.
- *8 cm heading.* The heading is defined by the direction, not the length. 8 cm
  puts the second point clear of the first at this camera distance (tens of
  pixels), so the angle is measurable.
- The metric folds the heading difference into [0°, 90°]. This is a scoring choice
  inherited from when the pose came from estimated keypoints. With a recorded pose
  the direction is known, so it could be scored over [0°, 180°].

**Validation.**

- The two trajectory checks (§4.4).
- By eye, the ring sits between the fingers and the arrow follows them on every
  sheet.

### 5.5 Grasp region

**What the model is asked.** Paint the region where the gripper should close on
the object it is about to pick. A painted region scores if it reaches the grasp,
stays near it, lies on the object and is reasonably tight (`grasp_score`).

**Ground truth.** The grasp the demonstration actually made: two finger points
(where the pads closed on the object), their midpoint as the grasp centre, and a
2 cm tolerance converted to pixels at the grasp's depth. Alongside it, the exact
mask of the object picked (`demonstrated_grasp`). The centre sits 1.0–6.2 cm from
the object's centroid.

**Procedure.**

1. **When:** the grasp frame from §4.1 step 2.
2. **Where:** at that replay step, the inner face of each finger pad
   (`grasp_fingers_world` in the record). Each pad is a thin collision box whose
   local y is the closing axis; its inner face is its centre moved half its
   thickness (4 mm) towards the other pad. The centre is the faces' midpoint.
3. **Projected** into the conditioning frame with the canvas camera. This relies
   on the object not having moved before the grasp, which §4.1 step 4 enforces.
4. **Which object:** the object most in contact with the finger geoms in the
   15 steps after the grasp frame (§4.5). Its mask is cut from the segmentation.
5. **Consistency guard:** the sample fails if the centre is more than 3 cm (in
   pixels at its depth) from the object mask. Observed: 0 px on 14 samples, at
   most 8 px otherwise.

**Design decisions.**

- *A demonstrated grasp, not a grasp model.* A grasp-synthesis model would again
  be pseudo ground truth, and needs a per-machine licence.
- *Finger points from the pads.* The first version placed the fingers at the grip
  site ± half the recorded finger gap. Against the simulator's pads that was off by
  6–11 px on `ep0145`, `ep0304` and `ep1191`, where the object pushed the two
  independently driven fingers off-centre (pad midpoint up to 1.3 cm from the grip
  site). Reading the pads removes that model. On symmetric grasps the centre moves
  2–3 px, because the pad centres sit 3.7 mm behind the grip site.
- *The object comes from physics.* Finger contacts identify what was held; on all
  20 samples it is the object the instruction names first.
- *First close only*, to avoid an object displaced by an earlier failed attempt.

**Validation.** The replay that records the pads reproduced every existing layer,
frame and meta value bit for bit. The object masks are unchanged; the grasp points
moved by a median of 3 px (at most 10 px, `ep0145`). On the dataset's own frame at
the grasp, the new points sit on the object's edges where the pads close.

---

## 6. Validation

| # | Check | What it rules out | Result |
|---|---|---|---|
| 1 | Action match (§4.3) | wrong demo | 20/20 at 2.9e-8 |
| 2 | End-effector track, every step (§4.4) | drifting replay, wrong settle, step offset | ≤ 0.3 mm |
| 3 | Image match at the frame (§4.4) | wrong scene or object placement | 3.3–7.8 / 255 |
| 4 | Render vs dataset frame, phase correlation | misregistered conditioning frame | ≤ 1.7 px of 720 |
| 5 | Camera intrinsics by RQ (§5.3) | wrong field of view, resolution or canvas shift | *f* = 869.12 px, exact |
| 6 | Selection camera vs replay camera (§5.3) | camera pose differences | 0.71 px (half-pixel convention) |
| 7 | Link depth vs projected depth (§5.2) | depth scale, near/far, camera misalignment | never negative; pattern as expected |
| 8 | Symmetric grasp model vs MuJoCo finger pads (§5.5) | grasp axis, centre, width model | 3 samples off by 6–11 px, so fingers now come from the pads |
| 9 | Grasped object vs instruction (§5.5) | wrong object | 20/20 |
| 10 | Recompute every store from the records with the eval code | stale or mis-written stores | 100/100 identical |
| 11 | Visual review sheets (`review_gt.py`) | anything a number misses | see `results/libero/review/index.html` |

The visual review draws, per sample, a 3x3 sheet:

1. the dataset frame;
2. the replay render;
3. their difference (×4);
4. instance masks;
5. depth with per-object medians;
6. keypoints and pose;
7. the grasp on its object;
8. the dataset's own frame *at the grasp* with the same finger points (the
   strongest single check, since it tests camera, grasp and object together);
9. a facts panel.

---

## 7. Design decisions

| Decision | Alternatives considered | Reason |
|---|---|---|
| Ground truth from simulator replay | SAM 3 / DA3 / RoboPEPP estimates; hand-picked SAM 3 points | Only the replay gives exact labels for all five tasks. The estimates had documented failures (§2.1). |
| Re-render the conditioning frame at 720 px from the replay | Keep the dataset's 256 px frame, upscaled | Frame and labels then come from one render, with no resampling and no compression. The difference from the dataset frame is ≤ 1.7 px and texture only (§8). |
| Match demos by actions | By episode order; by first frame image | Exact, cheap (kilobytes per file), and unambiguous at 2.9e-8. Order is wrong (e.g. ep1520 = demo_13). |
| Settle steps chosen by measurement | Assume 10 | 10 then needs no assumption: it won on every sample. |
| Two independent replay checks | Trust the replay | The robot track says nothing about objects; the image check is coarse. Together they cover both. |
| Object set from BDDL `obj_of_interest` | Parse instruction text | Resolves "the bowl next to the plate" to one body. It is the task's own definition. |
| Visible-pixel masks | Amodal masks | The model paints what is on screen. |
| DREAM/RoboPEPP keypoint set and frame origins | Visible link centres; all 9 Panda links | Standard, comparable and well-defined. |
| Keep occluded keypoints in view | Drop by depth test | The prompt asks for every joint. This is revisitable (§8). |
| Demonstrated grasp | Grasp-synthesis model | Exact and licence-free. Its limit, one valid grasp, is stated rather than hidden. |
| Finger points from the simulator's pads | Grip site ± half the recorded gap | Exact; the symmetric model was off by 6–11 px on 3 samples. |
| Freeze the 20 samples | Re-select under the new ground truth | Keeps earlier results and analysis comparable. Episode 1497 stays excluded. |

---

## 8. Limitations and threats to validity

1. **The conditioning frame changed.** The world model now sees the 720 px replay
   render, not the dataset frame. They are aligned to ≤ 1.7 px, but some textures
   differ. For example, the microwave window in `ep0145` shows a pattern in the
   dataset that the render lacks, probably a texture or renderer difference
   between OpenVLA's setup and this one. Clips generated from the old frames are
   stale.
2. **Keypoint definition and occlusion.**
   - The ground truth is joint frame origins, which a model reading the image
     cannot always see: `link0` at the base, `link6` inside the wrist.
   - 12 of 80 in-view keypoints are occluded.
   - If the intended task is "dots on the visible joints", the fair ground truth
     would drop occluded keypoints (by the depth test in §5.2) and could move
     points to the visible link surface.
   - This is a choice about what the task asks, not a measurement error. It should
     be fixed in the prompt and ground truth together.
3. **One grasp, not all valid grasps.** A model that paints a different valid grasp
   scores a miss. `grasp_score` is therefore a lower bound on grasp understanding,
   and only samples whose grasp is far from the object's centre (17 of 20 at ≥ 3 cm)
   separate "knows the grasp" from "found the object".
4. **Heading folded to 90°** (§5.4) discards the sign of the direction. This is
   lenient towards a model that points the arrow backwards.
5. **Sample size and selection.** 20 frames, one per task, chosen by rules that
   favour clean single-pick approaches. Results speak to this regime, not to long
   multi-step scenes or contact-rich tasks.
6. **Simulator fidelity.** "Exact" is relative to MuJoCo's geometry, which is what
   was rendered. It does not imply that the simulated scene is physically
   realistic.
7. **Reproducibility.**
   - Versions and revisions are pinned (§3).
   - The 15-step contact window, settle choice and thresholds are fixed in code.
   - The two local environment fixes (§4.2) do not touch physics or rendering.

---

## 9. Relation to RoboTrustBench

RoboTrustBench is real robot video (DROID), with no simulator behind it. It
therefore keeps only the two tasks whose ground truth a model can reasonably
estimate: instance segmentation (SAM 3, scoped to the instruction's objects) and
depth (DA3). The robot and grasp tasks run only on LIBERO.

Results on the two sets differ in kind. LIBERO scores are against exact ground
truth. RoboTrustBench scores are agreement with SAM 3 and DA3. A gap between them
mixes the world model's sim-to-real difference with the estimators' error, and
should be reported that way.

---

## Appendix A: per-sample record

Frame = conditioning frame index; grasp = grasp frame index. Error columns are from
§4.4. Keypoints are those in view. Offset is the grasp's distance from the object
centroid.

| Sample | Instruction | Demo | Frame / grasp | EE err (mm) | Img err | Objects (masks) | Grasped | Keypoints | Depth (m) | Offset (cm) |
|---|---|---|---|---|---|---|---|---|---|---|
| libero_10__ep0145_f052 | put the yellow and white mug in the microwave and close it | demo_25 | 52 / 96 | 0.0 | 5.97 | mug, microwave | mug | 0,6,7,hand | 0.69–2.40 | 3.9 |
| libero_10__ep0206_f023 | put the white mug on the plate and put the chocolate pudding to the right of the plate | demo_45 | 23 / 57 | 0.0 | 3.34 | plate, pudding, white mug | white mug | 0,2,7,hand | 0.55–2.70 | 5.7 |
| libero_10__ep0231_f017 | put the white mug on the left plate and put the yellow and white mug on the right plate | demo_39 | 17 / 64 | 0.0 | 3.32 | 2 plates, 2 mugs | white mug | 0,2,hand | 0.57–2.70 | 4.3 |
| libero_10__ep0304_f040 | put the black bowl in the bottom drawer of the cabinet and close it | demo_30 | 40 / 92 | 0.0 | 7.76 | bowl, cabinet | bowl | 0,6,7,hand | 0.66–2.40 | 4.6 |
| libero_10__ep0374_f054 | pick up the book and place it in the back compartment of the caddy | demo_5 | 54 / 85 | 0.0 | 4.16 | book, caddy | book | 0,2,6,7,hand | 0.57–2.87 | 4.1 |
| libero_goal__ep0546_f011 | put the wine bottle on top of the cabinet | demo_25 | 11 / 39 | 0.0 | 5.83 | bottle, cabinet | bottle | 0,7,hand | 0.70–3.07 | 3.5 |
| libero_goal__ep0657_f012 | put the cream cheese in the bowl | demo_18 | 12 / 49 | 0.0 | 6.09 | cream cheese, bowl | cream cheese | 0,7,hand | 0.70–3.07 | 0.8 |
| libero_goal__ep0674_f012 | put the bowl on top of the cabinet | demo_5 | 12 / 45 | 0.0 | 7.09 | bowl, cabinet | bowl | 0,6,7,hand | 0.70–3.07 | 4.1 |
| libero_goal__ep0751_f014 | put the bowl on the stove | demo_30 | 14 / 55 | 0.0 | 6.25 | bowl, stove | bowl | 0,7,hand | 0.70–3.07 | 4.9 |
| libero_goal__ep0798_f029 | put the wine bottle on the rack | demo_44 | 29 / 91 | 0.0 | 5.24 | bottle, rack | bottle | 0,6,7,hand | 0.69–3.07 | 4.1 |
| libero_object__ep0943_f032 | pick up the bbq sauce and place it in the basket | demo_36 | 32 / 57 | 0.0 | 4.29 | bbq sauce, basket | bbq sauce | 0,2,6,7,hand | 0.74–2.79 | 2.8 |
| libero_object__ep1183_f023 | pick up the butter and place it in the basket | demo_13 | 23 / 54 | 0.0 | 4.29 | butter, basket | butter | 0,2,6,7,hand | 0.74–2.79 | 2.6 |
| libero_object__ep1191_f012 | pick up the cream cheese and place it in the basket | demo_22 | 12 / 51 | 0.0 | 4.43 | cream cheese, basket | cream cheese | 0,2,6,7,hand | 0.74–2.79 | 1.8 |
| libero_object__ep1227_f027 | pick up the orange juice and place it in the basket | demo_0 | 27 / 49 | 0.0 | 4.45 | orange juice, basket | orange juice | 0,2,6,7,hand | 0.73–2.79 | 3.1 |
| libero_object__ep1248_f032 | pick up the milk and place it in the basket | demo_13 | 32 / 54 | 0.0 | 4.40 | milk, basket | milk | 0,2,6,7,hand | 0.74–2.79 | 4.3 |
| libero_spatial__ep1285_f041 | pick up the black bowl next to the plate and place it on the plate | demo_20 | 41 / 71 | 0.0 | 6.02 | bowl, plate | bowl | 0,6,7,hand | 0.63–3.07 | 4.3 |
| libero_spatial__ep1388_f018 | pick up the black bowl on the ramekin and place it on the plate | demo_28 | 18 / 46 | 0.3 | 5.09 | bowl, plate | bowl | 0,6,7,hand | 0.69–3.07 | 3.7 |
| libero_spatial__ep1418_f025 | pick up the black bowl next to the cookie box and place it on the plate | demo_49 | 25 / 86 | 0.01 | 5.03 | bowl, plate | bowl | 0,6,7,hand | 0.60–3.07 | 5.1 |
| libero_spatial__ep1520_f030 | pick up the black bowl on the wooden cabinet and place it on the plate | demo_13 | 30 / 49 | 0.0 | 3.91 | bowl, plate | bowl | 0,hand | 0.58–3.07 | 6.1 |
| libero_spatial__ep1560_f029 | pick up the black bowl between the plate and the ramekin and place it on the plate | demo_26 | 29 / 54 | 0.0 | 5.62 | bowl, plate | bowl | 0,6,7,hand | 0.64–3.07 | 5.7 |

Every sample used 10 settle steps. In the spatial suite each scene has two black
bowls. In every case `obj_of_interest` and the finger contacts pick the same one,
`akita_black_bowl_1`, the bowl the instruction describes.

---

## Appendix B: reproducing every step

```bash
# A: select the 20 samples (main env)
python scripts/fetch_libero.py --out data/libero

# B: the replay environment (once)
venv/bin/pip install uv
venv/bin/uv venv --python 3.10 venv_libero
venv/bin/uv pip install --python venv_libero/bin/python "mujoco==2.3.7" "robosuite==1.4.1" \
    "bddl==1.0.1" "numpy==1.26.4" "opencv-python-headless==4.10.0.84" h5py fsspec aiohttp \
    requests matplotlib easydict pyyaml future gym==0.25.2 pyarrow pandas "imageio[ffmpeg]" cloudpickle
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
git -C third_party/LIBERO checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
echo "$PWD/third_party/LIBERO" > venv_libero/lib/python3.10/site-packages/libero_repo.pth
#   then: fix the OpenGL path in venv_libero/.../mujoco/cgl/cgl.py to
#   /System/Library/Frameworks/OpenGL.framework/OpenGL, and write
#   venv_libero/.libero/config.yaml pointing at third_party/LIBERO/libero/libero/{bddl_files,init_files,assets}

# C: match demos
venv_libero/bin/python scripts/libero_replay/match_demos.py

# D + E: replay, validate, render, record
LIBERO_CONFIG_PATH=venv_libero/.libero MUJOCO_GL=cgl \
    venv_libero/bin/python scripts/libero_replay/render_gt.py

# F: fill the ground-truth stores (any eval run does it), then review by eye
python scripts/run_eval.py --config configs/eval.libero.yaml
venv/bin/python scripts/libero_replay/review_gt.py
open results/libero/review/index.html
```
