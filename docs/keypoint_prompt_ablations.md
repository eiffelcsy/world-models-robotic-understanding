# Robot-keypoint prompt ablations

**Status 2026-11-11: tuning finished; the base is kept** (`pck` 0.575). No arm
beats it, including R2 ("the robot arm stays exactly where it is"). The robot-motion
failure is not fixed by wording. Next: the base alone on the held-out frames, if a
held-out score is wanted (there is no tuned prompt to confirm).

- **Method:** [prompt_framework.md](prompt_framework.md).
- **Setup:** the same as segmentation, depth and grasp: Gemini Omni 1.2 Flash, the
  4 LIBERO tune frames, seed 0, one repeat.
- **Ground truth:** the Panda's link positions from a simulator replay
  ([modules/robot_keypoint_detection/README.md](../modules/robot_keypoint_detection/README.md)).
- **Primary metric:** `pck` on the last frame (within 5% of the frame diagonal,
  about 73 px).

## 1. The published baseline

Wiedemer et al. (2025), Fig. 12, "Keypoint localization" (success 0.58 by eye):

> Add a bright blue dot at the tip of the branch on which the macaw is sitting. The
> macaw's eye turns bright red. Everything else turns pitch black. Static camera
> perspective, no zoom or pan.

`paper` adapts it with the fewest edits that make it the robot task. Their two named
targets, each with its own colour, become one instruction for every joint:

> Add a bright dot of a different color on each joint of the robot arm. Everything
> else turns pitch black. Static camera perspective, no zoom or pan.

"Everything else turns pitch black" and the camera sentence are verbatim. This is our
adaptation, not their prompt for this task.

## 2. What the ground truth looks like on the tune frames

Only **2 to 5 of the 7 keypoints are inside the frame**: the camera crops the arm.
Links listed in the ground truth's `outside_frame` are dropped from scoring.

| frame | keypoints in view | in view |
|---|---|---|
| spatial ep1520 | 2 | link0 (base mount), hand |
| object ep1227 | 5 | link0, link2, link6, link7, hand |
| goal ep0546 | 3 | link0, link7, hand |
| libero_10 ep0231 | 3 | link0, link2, hand |

The base mount and the hand are in view on every frame. Extra dots on visible joints
that have no ground truth (link1, link5) lower precision. The 73 px tolerance is
lenient, so dot size matters less than dot count and placement.

## 3. Bridging path: paper → base (6 runs)

Built the same way as grasp: the task slots (R, T, E) move one at a time, and the
shared core (P, C, H) moves as one block (K1). Segmentation's path already measured
the core paragraphs one at a time.

| arm | change from the previous arm | slot |
|---|---|---|
| `paper` | adapted Fig. 12 prompt, black background | — |
| `R1` | "Everything else turns pitch black." → the base's rest paragraph | R |
| `T1` | "each joint of the robot arm" → the base's joint definition, "sits exactly on its joint", "one dot per joint and nothing else is marked" | T |
| `E1` | "Add a bright dot of a different color" → "One at a time, a small, bright, fully saturated circular dot appears…", "a different vivid color", "a compact circle a few pixels across, not a line, not a ring, not an arrow, not a label" | E |
| `K1` | the base's preservation and camera paragraphs added; its hold paragraph replaces "Static camera perspective, no zoom or pan." The text now equals the base byte for byte (checked) | P + C + H (core) |
| `base` | the base's negatives added (`prompt.yaml`) | N |

- Each arm file (`modules/robot_keypoint_detection/arms/`) carries its word diff
  against the previous arm.
- Arms `paper`–`K1` send no negative.
- **Gate waiver:** `paper` asks for a black background, so the scene-preservation
  gate rejects it by design. The gate is waived for that arm only. The decoder reads
  saturated dots on black without trouble.

## 4. Improvement arms (from the base, one slot each)

Three arms, chosen for how likely they are to raise `pck`. They draw on the
ground-truth layout above and on the lessons from the earlier probes:

- naming what to avoid produces it (segmentation T1 and E4);
- small marks are fragile once the video is compressed;
- concrete visible anchors beat abstract descriptions (grasp T2).

| arm | slot | change | targets |
|---|---|---|---|
| `T2` | T | add "The dots include the round base where the arm is mounted and the gripper at the end of the arm. Joints outside the picture get no dot." | Missing the two keypoints in view on every frame (recall), and crowding the visible arm with extra dots (precision). It gives no count, since the visible count runs from 2 to 5. |
| `E2` | E | "a compact circle a few pixels across" → "about as wide as one gripper finger" | Dots that compression smears below `min_area` (8 px) or desaturates below 0.5. A finger-wide dot (~15–20 px) is well inside `max_area` and costs nothing at a 73 px tolerance. |
| `E3` | E | delete "not a line, not a ring, not an arrow, not a label" | Rings, lines and labels appearing because the prompt names them. The negatives still list them. |

**Rejected while writing:** "There are seven dots: base, shoulder, upper arm, elbow,
wrist, flange, gripper". Most of those joints are out of view on the tune frames,
so naming seven would invite dots on the wrong parts of the visible arm.

## 5. How to run

```bash
scripts/run_ablations.sh robot_keypoint_detection paper R1 T1 E1 K1 base T2 E2 E3   # 36 clips
venv/bin/python scripts/ablation_table.py robot_keypoint_detection pck paper R1 T1 E1 K1 base T2 E2 E3
```

## Results log


**Run 2026-11-11.** 9 arms × 4 frames, seed 0: 36 clips, all generated on the first
attempt.

- **Invalid:** `paper` fails the scene gate on 3 of 4 clips (black background; waived)
  and `E2` on spatial (scene changed 0.184). E2 spatial has not been regenerated yet.
- **Contact sheet:** `results/ablations/robot_keypoint_detection/sheet_paper_R1_T1_E1_K1_base_T2_E2_E3.png`.

### 6. Decoder fix before reading any result (2026-11-11)

The first scoring decoded 16–20 "dots" per clip against 2–5 ground-truth keypoints.
`color_points` took **any saturated blob** as a dot, so red sauce bottles, the
orange-juice carton and wood grain counted.

- **On untouched input frames** (20 LIBERO frames), it found 12.8 dots on average (up
  to 32). Its chance `pck` was 0.000, so `pck` itself was not inflated, but precision
  and `n_pred` meant nothing.
- **Fix:** the same test grasp's `color_mask` already has. A dot pixel must differ
  from the input frame by at least 40 (`min_change: 40.0` in `probe.yaml`).
- **Checks on the 20 frames:**
  - untouched frames now decode **0** dots;
  - dots drawn at the ground truth score `pck` 1.000 and precision 1.000 (before the
    fix: precision 0.285).
- All arms were rescored. The old scores are in `eval_before_fix/`. Base `pck` went
  0.658 → 0.575 (saturated scene objects near the arm had been matching keypoints),
  and precision went 0.145 → 0.382.

### Scores (last frame, fixed decoder)

| step | arm | slot changed | `pck` | per frame (spatial / object / goal / libero_10) | precision | dots decoded | Δ vs previous | decision |
|---|---|---|---|---|---|---|---|---|
| 0 | `paper` | — | 0.358 | .50* / .60* / .33* / .00 | 0.20 | 7.8 | — | gate waived |
| 1 | `R1` | R | 0.492 | .50 / .80 / .33 / .33 | 0.21 | 8.0 | +0.134 | — |
| 2 | `T1` | T | 0.575 | .50 / .80 / 1.00 / .00 | 0.34 | 6.0 | +0.083 | — |
| 3 | `E1` | E | 0.425 | .50 / .20 / .67 / .33 | 0.31 | 4.8 | −0.150 | — |
| 4 | `K1` | P+C+H | 0.525 | .50 / .60 / .67 / .33 | 0.37 | 5.3 | +0.100 | — |
| 5 | `base` | N | **0.575** | .50 / .80 / .67 / .33 | 0.38 | 5.5 | +0.050 | current best |
| — | `T2` | T | 0.525 | .50 / .60 / .67 / .33 | 0.39 | 5.0 | −0.050 vs base | reject |
| — | `E2` | E | 0.475 (3 valid) | .50* / .40 / .67 / .33 | 0.52 | 3.8 | −0.100 vs base | reject |
| — | `E3` | E | 0.400 | .00 / .60 / .67 / .33 | 0.26 | 5.5 | −0.175 vs base | reject |

\* invalid.

**Bridging path:** paper 0.358 → base 0.575 (+0.217). The steps alternate up and down
(±0.15) within the spread seen in the other probes. As a whole the path roughly
doubles precision (0.20 → 0.38), because fewer, more deliberate dots are drawn.

### Diagnosis by slot

1. **The robot arm moves (P/R), the dominant failure.** On the spatial frame, nearly
   every arm swings the robot down into the scene before dotting it; on goal, several
   do. The dots then sit on the new pose, and the ground truth is the input pose.
   Spatial is stuck at 0.50 in every arm: only the base mount and one more point
   match. The scene gate (`max_change 0.15`) did not reject these clips.
2. **E2 (finger-wide dots) raises precision** (0.52) but decodes fewer dots (3.8), so
   recall falls.
3. **E3 (no "not a line…" list)** loses spatial entirely. There was no sign that the
   list was priming lines or rings.
4. **T2 (name the mount and gripper)** is neutral to slightly worse.

### Decision and next arm

**Keep the base.** The next arm targets failure 1 with an R-slot sentence: "The robot
arm stays exactly where it is in the photograph, in the same pose; only the dots
appear." It needs writing and 4 clips, plus regenerating E2 spatial (1 clip).

### Round 3 (2026-11-11)

- **R2** (from base, slot R) adds "The robot arm stays exactly where it is in the
  photograph, in the same pose; only the dots appear." 4 clips, all valid.
- **E2 spatial**, regenerated on seed 1000: valid, `pck` 0.00. So E2 is **0.350**
  over 4 valid clips.

| arm | `pck` | per frame (spatial / object / goal / libero_10) | precision | dots decoded |
|---|---|---|---|---|
| `base` | **0.575** | .50 / .80 / .67 / .33 | 0.38 | 5.5 |
| `R2` | 0.308 | .50 / .40 / .00 / .33 | 0.35 | 4.0 |

**R2 does not stop the motion.** On spatial the arm still swings down into the
scene, and on goal it moves too (`pck` 0). The likely cause shows in the inputs: on
spatial the gripper is partly cut off at the top of the frame, and the model brings
the arm into view so it can mark it. A sentence asking it to hold the pose does not
override that. Robot motion is therefore recorded as **a failure no slot change is
expected to fix** (framework §2, step 5), and tuning stops here.

**Final decision: keep the base.** Contact sheet: `sheet_base_R2.png`.
