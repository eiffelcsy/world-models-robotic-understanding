# Robot-pose prompt ablations

**Status 2026-11-12: finished. T3 was not confirmed on held-out frames;** it is
−0.091 against the base (p 0.21) over all 16 frames, worse on 11, mostly through
heading. The one frame where T3 never drew both dots counts as a failure (0). The
base stays the pose prompt. The final tune numbers, with every regeneration and the
failure rule applied, are in "Final tune scores". See "Held-out confirmation" at the end.

- **Method:** [prompt_framework.md](prompt_framework.md).
- **Setup:** Gemini Omni 1.2 Flash, the 4 LIBERO tune frames, seed 0, one repeat.
- **Ground truth:** the end-effector pose the simulator recorded, projected into the
  image as two points: the grip point, and a point 8 cm ahead along the approach
  axis ([modules/robot_pose_estimation/README.md](../modules/robot_pose_estimation/README.md)).
- **Primary metric:** `pose_score` = `exp(−pos_err / 5% of the diagonal) ×
  (1 − angle_err / 90)`, on the last frame.

## 1. No published pose prompt

**No paper prompts a video generation model to show a robot's pose.** (Search of
2026-11-10.)

- **Wiedemer et al. (2025)** cover pose *editing* ("3D-aware reposing", Fig. 41), not
  estimation.
- **Video-model manipulation work** generates a motion video, then recovers the
  end-effector pose with separate trackers or pose estimators. Examples: TC-IDM
  (arXiv 2601.18323), EmboAlign (2603.05757), NovaFlow (2510.08568), "Dreaming the
  Sound of Contact" (2609.19137). The video model never encodes the pose itself.

The base (`prompt.yaml`) is therefore our own design, and the related work it builds
on is about how to *draw* points and directions so that a model or a reader can
use them:

| source | convention | what the base takes from it |
|---|---|---|
| Wiedemer et al. 2025, Fig. 12 | a "bright blue dot" at a named point, read from the generated frame | a small saturated dot per point |
| MOKA (Liu et al. 2024, arXiv 2403.03174) | coloured dots marked on the image as answers for a VLM to choose | discrete points rather than shapes |
| PIVOT (Nasiriany et al. 2024, arXiv 2402.07872) | arrows drawn from the end-effector show its motion | the end-effector as the anchor, plus a direction |
| object-centric prompt-driven VLA (Li et al., CVPR 2025) | the contact point and the end-effector's axes drawn as coloured cues | a point plus an axis, distinguished by colour |

The base (`prompt.yaml`, our own design) combines these into two dots: a **red** dot on the grip point, and a
**green** dot a short distance ahead along the approach direction. The decoder
(`pose_points`, palette red `#e6194b` and green `#3cb44b`) reads position from the
red dot and heading from the direction red → green. Two dots were chosen over an
arrow because the decoder reads colour blobs: a dot pair gives a direction with no
line fitting, and the same dot decoder serves the keypoint probe.

The base's slots are P, C | T+E, T, T | E+R | H (comment in `prompt.yaml`).

## 1b. A paper-style baseline and bridge (added 2026-11-11)

**Why it was added.** Every other overlay probe starts from a short paper-style
prompt and bridges to its base. Without one here, pose would have no comparable
starting score, and the base's design choices would go unmeasured. So `paper`
adapts Wiedemer et al.'s Fig. 12 keypoint prompt, whose dot-at-a-named-point
convention the base uses. The keypoint probe's `paper` arm is adapted from the same
figure.

> **Fig. 12 (theirs):** Add a bright blue dot at the tip of the branch on which the
> macaw is sitting. The macaw's eye turns bright red. Everything else turns pitch
> black. Static camera perspective, no zoom or pan.
>
> **`paper` (ours, adapted):** Add a bright red dot on the center of the robot's
> gripper and a bright green dot a short distance in front of it, in the direction
> the gripper is pointing. Everything else turns pitch black. Static camera
> perspective, no zoom or pan.

Their two named points, each with its own colour, become the grip point and a point
ahead of it. Red and green are chosen to match the decoder's palette. That is the
one concession to our decoder, and it should be reported. This is our adaptation,
not a published pose prompt.

**Bridge** (built as for grasp and keypoints: the task slots one at a time, the
shared core as one block):

| arm | change from the previous arm | slot |
|---|---|---|
| `paper` | adapted Fig. 12 prompt, black background | — |
| `R1` | "Everything else turns pitch black." → the base's rest sentence | R |
| `T1` | the one-line placement → the base's framing sentence, red-dot sentence and green-dot paragraph | T |
| `E1` | "Add a bright red dot and a bright green dot" → "Two small, bright, fully saturated circular dots appear", plus "Both dots are small compact circles a few pixels across — no lines, no arrows, no rings, no labels, no text." | E |
| `K1` | the base's preservation and camera paragraphs added; its hold paragraph replaces the camera sentence. The text now equals the base byte for byte (checked) | P + C + H (core) |
| `base` | the base's negatives added | N |

- Arms `paper`–`K1` send no negative.
- `paper`'s black background fails the scene gate by design; the gate is waived for
  that arm only.
- T1 brings in the base's green-dot wording as fixed on 2026-11-11 (§3), so the
  path ends at the corrected base.

## 2. What the ground truth looks like

**Heading is nearly free on LIBERO.** The gripper points down at the table, so the
heading runs almost straight down the image:

- Across all 20 LIBERO frames, the median angle from straight down is 5°
  (maximum 59°).
- A fixed guess, "green dot directly below the red one", scores a heading score of
  **0.893**.

So `pose_score` mostly measures **position**. Report the heading score against that
0.893 floor, the way depth is reported against its row-gradient floor.

## 3. Improvement arms (from the base, one slot each)

Three arms, chosen for how likely they are to raise `pose_score`, ranked by expected gain.

| arm | slot | change | targets |
|---|---|---|---|
| `T3` | T | red dot "on the center of the gripper, right at the point the gripper would close on an object" → "between the two fingertips, midway between them, where a held object would be" | **Position**, which dominates the score. The ground-truth point is the grip site between the fingers; "the center of the gripper" can be read as the gripper body or the wrist. |
| `E2` | E | "small compact circles a few pixels across" → "about as wide as one gripper finger" | Dots too small or too pale to decode. Compliance needs both dots in the palette (hue ±22°, saturation ≥ 0.5). The same change as keypoint E2. |
| `E3` | E | delete "— no lines, no arrows, no rings, no labels, no text" | Lines, arrows or rings appearing because the prompt names them (segmentation T1 and E4). The negatives still list them. It is the same change as keypoint E3, so the hypothesis is tested on 8 paired frames across two probes. |

Each arm file (`modules/robot_pose_estimation/arms/`) carries its word diff against
the base.

**Fix to the base (2026-11-11, before any pose clip was made).** The green-dot
sentences said the dot lies "on the line the fingers would travel if the gripper
closed. Imagine the gripper's approach direction as a straight line leaving the
gripper; the green dot lies on that line, a short, fixed distance ahead of the red
dot." But the fingers close sideways, across the approach axis the ground truth
uses. So a model that followed the text would put the dot sideways and lose up to
all of the heading score.

The text was a description error, not a design choice, so it was fixed in
`prompt.yaml` rather than tested as an arm (it was briefly drafted as arm T2). The
base now says: "It sits just beyond the tips of the two gripper fingers, about a
finger's length past them, on the straight line that runs from the back of the
gripper through the middle of the fingertips." A finger's length (~5 cm) puts it
near the ground truth's 8 cm point, and only the direction is scored. Every arm
above was rebuilt on the fixed base and re-diffed.

E3 replaces T2 as the third improvement arm.

## 4. How to run

```bash
scripts/run_ablations.sh robot_pose_estimation paper R1 T1 E1 K1 base T3 E2 E3   # 36 clips
venv/bin/python scripts/ablation_table.py robot_pose_estimation pose_score paper R1 T1 E1 K1 base T3 E2 E3
```

## Results log


**Run 2026-11-11.** 9 arms × 4 frames, seed 0: 36 clips, all generated on the first
attempt.

- **Invalid:**
  - `paper` on 3 of 4 (black background; gate waived);
  - one dot decoded on `base` object, `K1` goal, and `E3` spatial and object.

  None of these has been regenerated yet. Their scores of 0 count in the means
  below.
- **Contact sheet:** `results/ablations/robot_pose_estimation/sheet_paper_R1_T1_E1_K1_base_T3_E2_E3.png`.

### 5. Decoder fix before reading any result (2026-11-11)

`pose_points` took the **largest** red and the largest green saturated blob anywhere
in the frame. A red sauce bottle or a green object could therefore be read as a dot.

- **On untouched input frames**, it found a "dot" on 13 of 20 frames, and both colours
  on 4. So a clip with no marks could pass compliance.
- **With dots drawn exactly at the ground truth:**
  - before the fix: `pose_score` 0.756 on average, with 5 of 20 frames misread (as
    low as 0.004);
  - after the fix: **0.990**, and untouched frames decode **0** dots.
- **Fix:** the same `min_change: 40.0` test against the input frame as the keypoint
  and grasp decoders.
- All arms were rescored. The old scores are in `eval_before_fix/`. Base went 0.086 →
  0.156, and T3 0.234 → 0.447.

### Scores (last frame, fixed decoder)

| step | arm | slot | `pose_score` | per frame (spatial / object / goal / libero_10) | heading score | angle err | `pos_pck` | Δ |
|---|---|---|---|---|---|---|---|---|
| 0 | `paper` | — | 0.293 | .01* / .44* / .34* / .39 | 0.892 | 9.7° | 0.50 | — |
| 1 | `R1` | R | 0.377 | .02 / .43 / .55 / .50 | **0.940** | 5.4° | 0.75 | +0.084 |
| 2 | `T1` | T | 0.198 | .01 / .34 / .44 / .00 | 0.328 | 60.4° | 0.75 | **−0.179** |
| 3 | `E1` | E | 0.130 | .00 / .39 / .10 / .03 | 0.202 | 71.9° | 0.75 | −0.068 |
| 4 | `K1` | P+C+H | 0.328 | .03 / .78 / .00* / .50 | 0.701 | 27.0° | 0.75 | +0.198 |
| 5 | `base` | N | 0.156 | .00 / .00* / .12 / .50 | 0.312 | 62.0° | 0.75 | −0.172 |
| — | `T3` | T | **0.447** | .01 / .76 / .08 / .94 | 0.579 | 37.9° | 0.75 | +0.291 vs base, better 3/4 |
| — | `E2` | E | 0.313 | .02 / .41 / .33 / .50 | 0.847 | 13.7° | 0.50 | +0.157 vs base |
| — | `E3` | E | 0.130 | .00* / .00* / .02 / .50 | 0.256 | 66.9° | 0.75 | −0.026 vs base |

\* invalid. Floor for comparison: a fixed "straight down" heading gets a heading
score of 0.893 (§2).

### Diagnosis by slot

1. **The heading wording (T), the main loss.**
   - From T1 on, the base's green-dot sentence is in the prompt: "just beyond the
     tips of the two gripper fingers … on the straight line that runs from the back
     of the gripper through the middle of the fingertips". This is the wording fixed
     into the base on 2026-11-11.
   - The model puts the green dot *beside* the red one: the two fingertips sit side
     by side in the image. The heading score drops from 0.94 (R1) to 0.33, far below
     the 0.893 straight-down floor.
   - R1 still has the paper's phrase, "a short distance in front of it, in the
     direction the gripper is pointing", and gets the heading nearly right.
   - **So the fix to the base made the heading worse.** The original sentence was
     never run, so it is unknown whether it would have done better. What is clear is
     that the simple directional phrase works.
2. **The robot arm moves (P/R).** On the spatial frame every arm except `paper`
   brings the gripper down into the scene, so spatial scores about 0 everywhere. This
   is the same failure as the keypoint probe's.
3. **T3 (red dot between the fingertips) is the best arm.** It has the best position
   score, and libero_10 reaches 0.94. Its heading still suffers from failure 1.
4. **E2 (finger-wide dots) helps the heading** (0.85), probably because the bigger
   dots decode more reliably. **E3** shows no benefit.

### Decision and next arms

**T3 is adopted** by the rule (0.447 > 0.156, better on 3 of 4 frames). Next arms,
written from T3:

- **T4:** the green-dot sentence → the paper's phrase, "It sits a short distance in
  front of the red dot, in the direction the gripper is pointing." This targets
  failure 1. It partly reverts the 2026-11-11 base fix, and should be reported that
  way.
- **R2:** "The robot arm stays exactly where it is in the photograph, in the same
  pose; only the dots appear." This targets failure 2, and is shared with the
  keypoint probe.

Also regenerate the 4 invalid clips (base object, K1 goal, E3 spatial and object) on
seed 1000.

### Round 3 (2026-11-11)

Two new arms, both built from T3, plus regenerations of the invalid clips on seed 1000:

- **R2** (slot R): adds "The robot arm stays exactly where it is in the photograph,
  in the same pose; only the dots appear."
- **T4** (slot T): green-dot sentence → "It sits a short distance in front of the red
  dot, in the direction the gripper is pointing."
- **Regenerations:**
  - base object: valid, 0.72, so **base is 0.336** over 4 valid clips (was 0.156 with
    the invalid clip as 0);
  - E3 spatial and object: valid, so **E3 is 0.292**;
  - K1 goal: **invalid again** (one dot), 2 of 3 attempts used; K1 stays 0.328 with
    that cell as 0.

| arm | from | `pose_score` | per frame (spatial / object / goal / libero_10) | heading score | angle err | valid |
|---|---|---|---|---|---|---|
| `T3` | base | **0.447** | .01 / .76 / .08 / .94 | 0.579 | 37.9° | 4/4 |
| `R2` | T3 | 0.095 | .00 / .38 / .00* / .00* | 0.228 | 69.5° | 2/4 |
| `T4` | T3 | 0.304 | .02 / .46 / .20 / .54 | 0.673 | 29.4° | 4/4 |
| `base` (regenerated) | — | 0.336 | .00 / .72 / .12 / .50 | — | — | 4/4 |

- **R2 makes things worse.** The robot still moves on spatial, and on goal and
  libero_10 only one dot is drawn (invalid; not regenerated, since R2 is rejected
  either way). Robot motion is recorded as a failure no slot change is expected to
  fix, as for keypoints.
- **T4 raises the heading score a little** (0.58 → 0.67) but loses position. The
  green dot still often lands beside or above the red one, so the paper's phrase does
  not recover R1's heading (0.94) once the rest of the base wording is in place.
  Heading remains below the 0.893 straight-down floor in every arm from T1 on, and
  this should be reported.

**Final decision: T3.** It is better than the regenerated base on 3 of 4 frames
(object 0.76 vs 0.72, libero_10 0.94 vs 0.50, spatial 0.01 vs 0.00; goal 0.08 vs
0.12). Contact sheet: `sheet_T3_R2_T4.png`.

## Held-out confirmation (2026-11-12)

**Setup:** base vs T3 on the 16 test frames, seed 1, paired by frame
(`configs/ablations/confirm.{gen,eval}.robot_pose_estimation.yaml`, fixed decoder).

**Validity:** 10 of 16 frames were valid in both arms on the first attempt; on the 6
others one arm drew only one dot or changed the scene. Both arms on those frames
were regenerated on seed 1001, which fixed 5.

On the sixth frame, `libero_goal__ep0674_f012`, T3 drew only the red dot on all three
seeds (1, 1001, 1002), while base was valid on seed 1002. Under the failure rule (§6.1
of the segmentation doc: a frame still invalid after 3 attempts is a failure of that
arm), **T3 scores 0 there**, paired with base's seed-1002 clip (0.034). The frame is
not dropped; dropping it would remove T3's worst case. **16 pairs**, from 48 clips in
all.

| | base | T3 | T3 − base | 95% CI | Wilcoxon p | d_z | T3 better/worse |
|---|---|---|---|---|---|---|---|
| **`pose_score`** | **0.387** | 0.296 | **−0.091** | [−0.245, +0.066] | 0.21 | −0.27 | 5/11 |
| best-frame `pose_score` | 0.391 | 0.312 | −0.080 | [−0.238, +0.075] | 0.32 | | 5/11 |
| `pos_err` (lower is better) | 0.028 | 0.027 | −0.001 | [−0.009, +0.009] | 0.30 | | 10/6 |
| heading score | 0.550 | 0.385 | −0.166 | [−0.363, +0.022] | 0.08 | | 4/12 |

The failed clip keeps its own measured `pos_err` and angle; only the headline counts
as 0. Dropping the frame instead gives base 0.410, T3 0.316, −0.094 (p 0.25): the same
conclusion.

**T3 is not confirmed.** The tune-frame gain (0.447 vs 0.336) came from 4 frames on
one seed. On held-out frames:

- position is unchanged;
- the heading is worse in T3 on 12 of 16 frames. Both arms carry the same green-dot
  sentence, so this is sample-to-sample variation in where the green dot lands;
- both arms' heading (0.55 and 0.39) stays below the 0.893 a fixed straight-down guess
  gets (§2);
- spatial stays near 0 in both (robot motion, round 3).

Per suite, base → T3: libero_10 0.615 → 0.364, goal 0.414 → 0.296 (with the failure),
object 0.475 → 0.401, spatial 0.043 → 0.123.

**The base stays the pose prompt.** Its held-out `pose_score` is 0.387 (16 frames). Stats:
`results/confirmation/robot_pose_estimation/libero/confirm_stats.txt`.

### Visual check of the held-out clips (2026-11-12)

The last frames of every held-out clip and every attempt were checked by eye
(`sheet_all.png`, `sheet_retried.png` in `results/confirmation/robot_pose_estimation/libero/`):

- **Every invalid clip ("1 point(s) < 2") has a red dot but no green one.** The green
  dot is missing or merged into the red. On goal ep0674, T3 drew only the red dot on
  all three seeds. That frame is scored as T3's failure.
- **The scene gate missed a camera tripod** that appeared on the table (base,
  libero_10 ep0304, seed 1001, counted valid at 0.32). This is the same literal reading
  of the core sentence "locked off on a tripod" as on tune frames.
- **Heading:** where both dots are drawn, the green dot is often beside the red one
  rather than below it, in both arms. That matches T3's lower heading score.

## Final tune scores (2026-11-12)

Every tune clip that failed the gate was regenerated, up to 3 attempts. A frame still
invalid after that scores 0 as a failure. `paper` is exempt: its black background
fails the gate by design, so its clips keep their raw scores. Recompute with
`venv/bin/python results/ablations/merged_means.py robot_pose_estimation pose_score <arms>`.

| arm | slot | `pose_score` | per frame (spatial / object / goal / libero_10) | regenerations and failures |
|---|---|---|---|---|
| `paper` | — | 0.293 | .01 / .44 / .34 / .39 | gate waived |
| `R1` | R | 0.377 | .02 / .43 / .55 / .50 | — |
| `T1` | T | 0.198 | .01 / .34 / .44 / .00 | — |
| `E1` | E | 0.130 | .00 / .39 / .10 / .03 | — |
| `K1` | P+C+H | 0.439 | .03 / .78 / .45 / .50 | goal valid on the 3rd attempt (seed 2000) |
| `base` | N | 0.336 | .00 / .72 / .12 / .50 | object regenerated (seed 1000) |
| `T3` | T | **0.447** | .01 / .76 / .08 / .94 | — |
| `E2` | E | 0.313 | .02 / .41 / .33 / .50 | — |
| `E3` | E | 0.292 | .00 / .64 / .02 / .50 | spatial and object regenerated |
| `R2` | R | 0.140 | .00 / .38 / .18 / **0 (failure)** | goal valid on the 3rd attempt; libero_10 invalid on all 3 (one dot) |
| `T4` | T | 0.304 | .02 / .46 / .20 / .54 | — |

**Bridging path with final numbers:** paper 0.293 → R1 0.377 (+0.084) → T1 0.198
(−0.179) → E1 0.130 (−0.068) → K1 0.439 (+0.309) → base 0.336 (−0.103). The earlier
round tables show values from before these regenerations. Where they differ (K1,
base, E3, R2), this table is the one to cite.

**Decisions are unchanged.** T3 remains the best improvement arm and was tested on
held-out frames (not confirmed). K1 (0.439) is a bridging step, not a candidate, as
on every probe's path.
