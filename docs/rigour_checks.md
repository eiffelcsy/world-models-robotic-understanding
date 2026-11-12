# Rigour checks on the existing clips

**Run 2026-11-11.** Four re-analyses of clips and ground truth already on disk.
No clip was generated, and no prompt, decoder setting, metric or ground-truth
store was changed. Everything here is **post hoc**: it was designed after the
held-out results were known, so it is reported as sensitivity and description,
not as a new confirmatory test. The pre-specified results in the per-probe
documents stand as they are.

| item | question | script | output |
|---|---|---|---|
| 1 | Where in the chain is each score lost? | `scripts/rigour/funnel.py` | `results/rigour/funnel.md` |
| 2 | How much does the segmentation result depend on the decoder's thresholds? | `scripts/rigour/decoder_sweep.py` | `results/rigour/decoder_sweep.md` |
| 3 | What do a guess, a perfect answer and a specialist score on the same frames? | `scripts/rigour/specialists.py`, `references.py` | `results/rigour/references.md` |
| 4 | Does a perfect answer survive video compression, the readout and the gates? | `scripts/rigour/ceiling.py` | `results/rigour/ceiling.md` |

All four run from the repo root with `venv/bin/python scripts/rigour/<name>.py`
and rebuild their tables from the clips. The scripts reproduce the documented
numbers before changing anything: at the settings in `probe.yaml` they give
held-out segmentation 0.511 (base) and 0.552 (H1), 19 and 21 of 33 objects, and
tune scores 0.481 / 0.713 / 0.863 (base / E1 / H1), as in
[segmentation_prompt_ablations.md](segmentation_prompt_ablations.md).

## Summary

| # | Finding | Consequence |
|---|---|---|
| 1 | On held-out segmentation, H1 paints 32 of 33 objects (base: 25) and 28 are recoverable when the decoder is told where the object is (base: 22). The decoder then loses 7 of H1's 28. | The reported 0.552 is mostly an encoding loss, not a failure to find objects. Report the funnel beside `mean_iou`. |
| 2 | The 0.80 saturation threshold costs 0.12–0.17 `mean_iou`. From 0.3 to 0.5 the score is flat (base 0.63, H1 0.72) and the decoder's floor and ceiling checks are unchanged. | The headline number depends on a threshold the two checks do not pin down. Report 0.80 (pre-specified) with the 0.50 value as sensitivity. |
| 2 | On the tune frames, base scores 0.848 at 0.50, against E1's 0.872. | E1's adopted gain (+0.232 at 0.80) was almost all the threshold, not the prompt. |
| 3 | The mean depth map of the *other* frames in the same suite scores δ1 0.841 on held-out frames, above Omni (0.668) and Depth Anything 3 (0.568). | "Beats the row gradient" is not enough. Omni beats this prior only on ordering (monotone δ1 +0.089, CI [+0.048, +0.126]), not on δ1. |
| 3 | Keypoints and pose on the tune frames are below a mean-position guess (PCK 0.575 vs 0.725; pose score 0.156 vs 0.307; n = 4). | No evidence yet that these two probes read the robot from the image. |
| 3 | For grasp, under the old `grasp_score`, a dot centred on the demonstrated grasp scored 0.75 and "paint the top quarter of the object" 0.67. | **Fixed 2026-11-11:** the headline is now `grasp_loc` (a perfect mark 1.00, whole object 0.00). Guesses that know only the object still score about 0.5, a dataset prior that every grasp result must beat. Under `grasp_loc` no tune arm beats them. |
| 3 | SAM 3 given the full referring phrase does no better than with the noun (0.549 vs 0.559). | The prompt difference does not explain SAM 3's misses. |
| 4 | A perfect answer survives H.264 at the generator's bit rate, with and without added grain, for every probe (segmentation 0.990 → 0.977) as long as dots have a radius of 4 px or more. At 2 px, the pose answer passes the gates on 2 of 20 frames. | Compression is not what limits the scores. Dot size is a real constraint: the keypoint and pose prompts ask for dots "a few pixels across". |

## 1. Score funnel

Each headline score is the product of three things: the model located the
property, it drew it in the requested form, and the decoder could read it. The
funnel reports them apart.

### Segmentation (16 held-out frames, 33 objects, last frame, seed 1)

- **painted**: at least half of the object's pixels differ from the conditioning frame.
- **readable**: a decoder that is *given the object's true mask* recovers it at IoU ≥ 0.5.
  It takes the modal hue of the changed pixels inside the object, at any
  saturation ≥ 0.2, and keeps the connected pieces of that hue that touch the
  object. Spill connected to the object and under-painting still cost IoU. It is
  an upper bound on any colour decoder. On untouched frames it scores 0.000; on
  the true masks painted flat it scores 0.993.
- **matched**: the probe's own decoder matches it at IoU ≥ 0.5 (the reported result).

| arm | painted | readable | matched | mean oracle IoU per frame [95% CI] | mean decoder IoU per object |
|---|---|---|---|---|---|
| base | 25 (76%) | 22 (67%) | 19 (58%) | 0.656 [0.508, 0.789] | 0.530 |
| H1 | 32 (97%) | 28 (85%) | 21 (64%) | 0.805 [0.731, 0.874] | 0.575 |

Where the 33 objects are lost:

| | not painted | painted, not readable (extent, merge, spill) | readable, lost by the decoder (encoding) | matched |
|---|---|---|---|---|
| base | 8 | 3 | 3 | 19 |
| H1 | 1 | 4 | 7 | 21 |

- **H1 − base on the oracle score: +0.149, 95% CI [+0.022, +0.291]**, paired by
  frame. This is exploratory: the oracle score was defined after the held-out
  result was known. The pre-specified test on `mean_iou` (+0.041, p = 0.94) is
  still the confirmatory result. What this adds is the reason: H1 changed *what
  is painted*, and the decoder hid it.
- The oracle score does not charge for extra masks, so it is not a replacement
  for `mean_iou`.

### Depth (16 held-out frames)

The analogue of the oracle decoder is the **best monotone map** from brightness to
depth (isotonic fit in log depth, either direction). It removes any penalty for
how brightness encodes distance and keeps only the ordering.

| prediction | δ1, affine in depth | δ1, affine in disparity | δ1, best monotone map | \|Spearman ρ\| |
|---|---|---|---|---|
| Omni base | 0.668 | 0.788 | 0.905 | 0.820 |
| Omni N2 | 0.694 | 0.754 | 0.892 | 0.807 |
| row gradient (no image) | 0.575 | 0.654 | 0.676 | 0.650 |
| mean depth of the same suite's other frames | 0.841 | 0.735 | 0.817 | 0.753 |
| Depth Anything 3 | 0.568 | 0.756 | 0.861 | 0.917 |

- **Encoding loss is large.** Base goes from 0.668 to 0.905 once the encoding is
  free. Its brightness orders the scene far better than the affine score says.
- **Against the row gradient**, base is above on 16 of 16 frames under the
  monotone map (+0.229 [+0.195, +0.264]) and under Spearman.
- **Against the suite-mean prior** (§3), base is *below* on δ1 (−0.173
  [−0.288, −0.066]), level in disparity (+0.054 [−0.009, +0.112]) and above under
  the monotone map (+0.089 [+0.048, +0.126], 12 of 16 frames).
- **The ranking depends on the metric.** By affine δ1: suite mean > Omni > DA3.
  By Spearman: DA3 > Omni > suite mean. A depth claim should name the metric and
  show both.
- The monotone fit has many free parameters and flatters every row a little; read
  the differences between rows, not the absolute values.

### Grasp (tune frames, seed 0; each stage also requires the ones above)

| stage | all arms but `paper` (40 clips) | base | T2 |
|---|---|---|---|
| valid | 39 | 4/4 | 4/4 |
| something painted | 39 | 4/4 | 4/4 |
| paint on the target object | 32 | 4/4 | 4/4 |
| paint reaches the grasp | 28 | 4/4 | 3/4 |
| a part, not the whole object (coverage < 0.5) | 3 | 0/4 | 0/4 |
| paint concentrated at the grasp (focus ≥ 0.5) | 1 | 0/4 | 0/4 |

The model finds the object (32 of 40) and almost never selects a part of it (3 of
40). That is object localisation, not grasp knowledge.

### Keypoints (tune frames, seed 0, decoder as in `probe.yaml` today)

Across the nine arms, 15–32 dots are decoded for 13 keypoints in view; 5–8 of 13
keypoints are hit at 5% of the diagonal (73 px) and 1–7 at 2.5%. Precision is
0.16–0.40. Full table: `results/rigour/funnel.md`. Pose has too few clips for a
funnel.

## 2. Decoder sensitivity (segmentation)

Same clips, rescored with `min_saturation` swept; `min_change` 40. Floor: frame 0
of the 32 held-out clips. Ceiling: true masks painted flat, 20 frames.

| `min_saturation` | test base | test H1 | H1 − base [95% CI] | matched of 33 (base / H1) | tune base | tune E1 | tune H1 | floor | ceiling |
|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 0.626 | 0.724 | +0.097 [−0.046, +0.256] | 22 / 28 | 0.873 | 0.874 | 0.962 | 0 | 0.988 |
| 0.4 | 0.627 | 0.724 | +0.097 [−0.046, +0.256] | 22 / 28 | 0.872 | 0.875 | 0.963 | 0 | 0.988 |
| 0.5 | 0.627 | 0.722 | +0.095 [−0.047, +0.253] | 22 / 28 | 0.848 | 0.872 | 0.961 | 0 | 0.989 |
| 0.6 | 0.623 | 0.687 | +0.064 [−0.092, +0.232] | 22 / 28 | 0.705 | 0.863 | 0.956 | 0 | 0.990 |
| 0.7 | 0.584 | 0.642 | +0.059 [−0.127, +0.258] | 20 / 26 | 0.576 | 0.847 | 0.938 | 0 | 0.990 |
| **0.8 (probe.yaml)** | 0.511 | 0.552 | +0.041 [−0.145, +0.245] | 19 / 21 | 0.481 | 0.713 | 0.863 | 0 | 0.990 |
| 0.9 | 0.257 | 0.448 | +0.190 [+0.028, +0.353] | 9 / 17 | 0.268 | 0.363 | 0.477 | 0 | 0.991 |

`min_change` 20 / 40 / 60 at saturation 0.80: test base 0.511 at all three, H1
0.545 / 0.552 / 0.552. The change threshold does not matter.

- **The score is flat from 0.3 to 0.5 and falls from there.** 0.80 sits on the
  slope, so it measures how saturated the paint is as much as where it is.
- **The floor and ceiling do not choose a threshold.** Both pass at every value.
  The saturation test was the only guard against scene colours when it was set;
  `min_change` now does that job.
- **H1 − base is not significant at any threshold up to 0.8**, and is at 0.9. The
  comparison's verdict is stable where the decoder is sensible, which supports
  reporting the pre-specified result.
- **E1's tune gain was the threshold.** At 0.80 E1 beats base by +0.232; at 0.50
  by +0.024. E1 asked for saturated paint, which is what 0.80 demands.
- **Against SAM 3** (§3): at 0.80 H1 is level with SAM 3 (−0.007); at 0.50 H1 is
  above it by 0.163 [0.077, 0.250]. The comparison with the specialist flips
  with this one parameter.

**Suggested handling.** Keep 0.80 as the primary, pre-specified result. Add the
0.50 value as a labelled post hoc sensitivity row, with this table in an
appendix. For any decoder not yet used in a confirmation, choose thresholds on
the tune frames and synthetic checks only, and say so.

## 3. Reference table

Means on the frames shown. "Prior" rows use no knowledge of the image being
scored. Skill = (model − best prior) / (ceiling − best prior). Full tables,
including tune and all-20 columns: `results/rigour/references.md`.

| probe (metric) | null | best prior | ceiling | specialist | Omni | frames | skill |
|---|---|---|---|---|---|---|---|
| segmentation (`mean_iou`) | 0.000 | — | 0.988 | SAM 3 0.559 (noun), 0.549 (phrase) | base 0.511, H1 0.552 | 16 test | — |
| depth (`delta1`) | 0.350 | 0.841 suite mean | 1.000 | DA3 0.568 | base 0.668, N2 0.694 | 16 test | −1.09, −0.93 |
| depth (`disp_delta1`) | 0.502 | 0.735 suite mean | 1.000 | DA3 0.756 | base 0.788, N2 0.754 | 16 test | 0.20, 0.07 |
| keypoints (`pck`) | 0.000 | 0.725 mean position | 1.000 | — | base 0.575 | 4 tune | −0.55 |
| keypoints (`f1`) | 0.000 | 0.592 mean position, same suite | 1.000 | — | base 0.438 | 4 tune | −0.38 |
| pose (`pose_score`) | 0.000 | 0.307 mean grip point, heading down | 0.986 | — | base 0.156 | 4 tune | −0.22 |
| grasp (`grasp_loc`, from 2026-11-11) | 0.000 | 0.705 dot on the object point nearest the gripper* | 1.000 | — | base 0.057, T4 0.162, E3 0.292 | 4 tune | −2.20, −1.84, −1.40 |
| grasp (old `grasp_score`) | 0.000 | 0.736 top quarter of the object* | 0.728 (2 cm dot), 0.795 (1 cm dot) | — | base 0.034, T4 0.168, E3 0.072 | 4 tune | undefined |

\* given the true object mask (and, for the first, the gripper's image position); knows nothing about the grasp.

**Segmentation.**
- SAM 3 with the referring phrase: 24 of 33 matched, against 25 with the noun.
  The phrase differs only on the 5 spatial frames; it fixes one (0.65 → 0.97) and
  breaks one (0.98 → 0.49).
- SAM 3 − H1: +0.007 [−0.068, +0.082] (noun), −0.003 [−0.082, +0.074] (phrase).
- There is no image-free prior for segmentation, so no skill score.

**Depth.**
- **The strongest prior is the dataset itself.** LIBERO frames within a suite
  share a table and a camera, so the mean depth of the other frames in the suite
  scores δ1 0.841 (0.778 using all 19 other frames). The row gradient (0.575) is
  a much weaker floor.
- Depth Anything 3 against simulator depth: δ1 0.568, disparity δ1 0.756,
  Spearman 0.917. It orders the scene best of all and fits an affine map to depth
  worst. Its per-frame δ1 ranges from 0.01 to 0.98. For RoboTrustBench, where DA3
  *is* the ground truth, this is the pseudo-ground-truth error to quote.
- DA3 − Omni base: δ1 −0.101 [−0.203, −0.011]; disparity δ1 −0.032 [−0.063, +0.002].

**Keypoints and pose** (tune frames only, n = 4, CIs include 0).
- The arm is in much the same place in every frame, so a mean-position guess
  reaches PCK 0.52–0.73 at the 5% (73 px) tolerance.
- Pose: Omni's position is better than the guess (`pos_pck` 0.75 vs 0.25–0.50),
  but its heading score is 0.312 against 0.936 for "straight down". The pose
  score is lost on the heading dot.

**Grasp** (as found, under the old `grasp_score`; fixed the same day, below).
- **A mark centred on the grasp scored below 1:** 0.748 at 2 cm radius, over 20
  frames. Rim grasps sit at the object's edge, so part of the mark is off the
  object, and `on_object` charged for it. Clipped to the object, the mark scores
  1.0, but on 4 frames it is then a sliver of 27–62 px.
- **A finger-wide mark could be too small to decode.** At 1 cm radius it is
  149–197 px on 5 of the 20 frames, and after the decoder's blur and cleanup it
  fell under the 184 px floor.
- **"Top quarter of the object" scored 0.665**, close to the ideal answer.
  LIBERO's grasps are top-down on rims and lids, and the camera looks down.

**Fix (2026-11-11).** The headline is now `grasp_loc`: one minus the paint's mean
distance from the grasp beyond the 2 cm tolerance, over the object's. It was chosen
on synthetic answers only, and the decoder floor was halved to 92 px. Details,
checks and the rescored tune arms are in
[grasp_prompt_ablations.md](grasp_prompt_ablations.md), "Metric revision".
- A mark on the grasp now scores 1.000 (1 or 2 cm radius, through the decoder and
  through video); the whole object 0.003.
- Guesses that know the object but not the grasp score 0.35 (centroid), 0.49 (top
  quarter) and 0.53 (nearest the gripper). This is a dataset prior, not a metric
  fault, and it is the floor a grasp result must beat.
- On the tune frames no arm beats it. The best, E3, scores 0.292; the
  nearest-gripper guess scores 0.705 on the same 4 frames.
- The rescore changes the tune ranking: E3 first (0.292), then T4 (0.162).
  Under the old metric T4 was first and E3 sixth. Which arm to confirm is open.

## 4. Ceilings through video

The ideal answer for each probe, drawn from the ground truth on all 20 frames,
was made into a clip shaped like a generated one (1280×720, 24 fps, 6 s, H.264 at
2200 kb/s, the generator's average): the conditioning frame holds, cross-fades to
the answer, and holds. It was then scored by the unchanged pipeline (last-frame
readout, decoder, metric, gates). A static clip is easy to encode, so a second
variant adds fresh grain (sd 4 of 255) to every frame.

| probe | ideal answer | metric | still frame | through video | passes gates | with grain | passes gates |
|---|---|---|---|---|---|---|---|
| segmentation | distinct colours | `mean_iou` | 0.990 | 0.977 | 20/20 | 0.977 | 20/20 |
| segmentation | all one colour | `mean_iou` | 0.990 | 0.976 | 20/20 | 0.975 | 20/20 |
| depth | linear in depth | `delta1` | 1.000 | 1.000 | 20/20 | 1.000 | 20/20 |
| depth | linear in inverse depth | `delta1` | 0.766 | 0.766 | 20/20 | 0.765 | 20/20 |
| grasp | dot on the grasp, 1 cm radius | old `grasp_score` | 0.606 | 0.807 | 19/20 | 0.806 | 19/20 |
| grasp | dot on the grasp, 2 cm radius | old `grasp_score` | 0.748 | 0.750 | 20/20 | 0.749 | 20/20 |
| grasp | dot on the grasp, 1 cm radius | `grasp_loc`, floor 92 px | 1.000 | 1.000 | 20/20 | 1.000 | 20/20 |
| grasp | dot on the grasp, 2 cm radius | `grasp_loc`, floor 92 px | 1.000 | 0.999 | 20/20 | 0.999 | 20/20 |
| keypoints | dots, radius 2 px | `pck` | 0.267 | 1.000 | 20/20 | 1.000 | 20/20 |
| keypoints | dots, radius 4 px / 8 px | `pck` | 1.000 | 1.000 | 20/20 | 1.000 | 20/20 |
| pose | dots, radius 2 px | `pose_score` | 0.049 | 0.098 | 2/20 | 0.098 | 2/20 |
| pose | dots, radius 4 px / 8 px | `pose_score` | 0.990 | 0.990 | 20/20 | 0.990 | 20/20 |

- **Compression costs almost nothing.** Segmentation loses 0.013; depth, and dots
  of 4 px radius or more, lose nothing. No ideal answer fails the scene or
  stability gates. The two checks the segmentation document lists as "not run"
  (compression, same colour) both pass.
- **Same colour for every object costs nothing here** (0.976), because the target
  objects in these 20 frames do not touch.
- **Very small dots are not a reliable encoding.** At 2 px radius the pose
  decoder finds both dots on 2 of 20 frames, and the keypoint decoder reads a
  still frame at PCK 0.267. (Through video the keypoint dots do decode, because
  chroma subsampling spreads their colour.) Both prompts ask for dots "a few
  pixels across"; the finger-wide arms (keypoint E2, pose E2) ask for the size
  the decoder needs.
- **Grasp:** under the old `grasp_score` the ceilings stayed at 0.75–0.81
  through video, as in §3: the limit was the metric, not the codec. Under
  `grasp_loc` both dots score 0.999–1.000 and pass every gate.
- The synthetic clip has no motion, so this is an upper bound on what
  compression leaves. The grain variant is the closer case and changes nothing.

## What this changes in the write-up

1. Report the funnel beside every headline score, with the oracle bound labelled
   as an upper bound.
2. State the segmentation result at 0.80 and at 0.50, and say E1's tune gain was
   a threshold effect.
3. Replace "beats the row gradient" with the three-prior comparison for depth,
   and say on which metric Omni clears the suite-mean prior.
4. Cite §4 for "compression does not limit the scores", and raise the dot size
   in the keypoint and pose prompts to at least 4 px radius.
5. Do not present keypoint, pose or grasp numbers as evidence of understanding
   until a held-out run beats the priors in §3. For grasp, use `grasp_loc` and
   report the difference from the object-only guesses.
6. Quote SAM 3 (0.559) and DA3 (δ1 0.568, Spearman 0.917) against simulator truth
   as the measured error of the RoboTrustBench pseudo ground truth.

## Limits of these checks

- All post hoc, on clips whose headline results were already known.
- Keypoint, pose and grasp rows use the 4 tune frames and one seed.
- Bootstrap CIs resample frames as if independent; frames within a suite share a
  scene, so the intervals are somewhat too narrow.
- The oracle decoder and the monotone fit are upper bounds, not scores a blind
  readout could reach.
