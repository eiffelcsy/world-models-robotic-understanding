# Grasp-region prompt ablations

**Status 2026-11-12: finished.** **E3 (the dot) is the final grasp prompt.** It was
chosen on 2026-11-11 under the new `grasp_loc` headline, before any held-out E3 clip
was made (option 1 below).

- **Held-out confirmation:** E3 beats the base on all 16 test frames: `grasp_loc`
  0.456 vs 0.026, +0.430, CI [+0.28, +0.58], p 0.001.
- **But it is level with guesses that know only the object** (top quarter 0.451,
  nearest the gripper 0.484). So the gain shows that the model finds and marks the
  object part, not that it knows the grasp. See "Held-out confirmation" at the end.

- **Method:** [prompt_framework.md](prompt_framework.md).
- **Setup:** the same as segmentation and depth: Gemini Omni 1.2 Flash, the 4 LIBERO
  tune frames, seed 0, one repeat.
- **Ground truth:** the grasp recorded in each demonstration, on the simulator's
  exact object mask ([modules/grasp_region_seg/README.md](../modules/grasp_region_seg/README.md)).
- **Primary metric:** `grasp_loc` on the last frame, from 2026-11-11. Every table
  before "Metric revision" uses the old `grasp_score`, as it was at the time.

## 1. No published baseline

We found no paper that asks a video generation model to mark where an object
should be grasped. The search was run 2026-11-10: about ten web searches, Wiedemer
et al.'s project page and PDF, and the full text of the closest papers.

| work | what it does with a video model | usable as a grasp baseline? |
|---|---|---|
| Wiedemer et al. 2025, Fig. 45 "Affordance recognition" | "The robot hands mounted on robot arms pick up the hammer, naturally like how a human would." Scored by eye (0.5) | no: asks for the action, nothing to decode |
| Grasp as You Dream (arXiv 2604.07517) | Veo / Kling generate a human hand grasping; the template is "A human hand grasps the 𝒫 of 𝒪 to 𝒯". The grasp is recovered by hand-pose estimation (HaMeR) | no: the part is chosen by a VLM beforehand, and nothing is marked |
| MME-CoF (2510.26802), VGI-Bench (2608.19583) | benchmark video models on reasoning, including an "embodied"/"affordance" category or tag | no: judged by a VLM from camera behaviour or the final scene, no mask |
| RoboWM-Bench, Veo-Act, TC-IDM | video models as world models or planners for manipulation | no: the output is an action or trajectory |
| VidBot, Robo-ABC, FSAG, RAGNet, VLAff | affordance or grasp maps from real videos, image diffusion features or VLMs | no: not a video generation model |

**Related convention: answering by marking the frame.** Two lines of work ask video
or image generators to answer by drawing on the input image, scored on the last
frame. Neither has a grasp task:

- Thinking with Video / VideoThinkBench (2511.04570): Sora-2 is asked, e.g., to
  "draw a black line tangent to the circle at the highlighted point";
- BabyVision-Gen (2601.06521): models are told to "annotate the original input
  image with minimal overlays (e.g., circles, lines, arrows)".

This supports the general setup (mark the answer, read the last frame). It is also
a possible later E arm: a point or circle at the grasp instead of a region mask.
It is not a grasp prompt to use as a baseline.

So the baseline, `paper`, is **Wiedemer et al.'s segmentation prompt (App. B.2)
with the fewest edits that make it a grasp task**. Against the instance_seg
`paper` arm, it changes only:

- "instance segmentation" → "grasp region segmentation";
- "each distinct entity is overlaid in a different flat color" → "the part of
  {grasp_object} that a robot would grasp is overlaid in a flat color";
- the four one-entity-at-a-time bullets → one bullet for the single target, keeping
  "covered by a flat color, perfectly preserving its silhouette".

Everything else is verbatim, including the green background and the Camera
bullets. This is our adaptation, not a published prompt, and should be reported
that way.

## 2. The base prompt, by slot

| paragraph | slot |
|---|---|
| "This photograph stays exactly as it is…" | P (core) |
| "The camera is locked off on a tripod…" | C (core) |
| "An opaque colored mask appears on top of the graspable part of {grasp_object} — the region … a robot gripper would close around…" | T (what) + E (opaque mask) |
| "The mask covers only that graspable region, not the whole object, and sits precisely on its outline." | T (extent) |
| "It is one flat, fully saturated color, with no shading…" | E |
| "Once it has appeared it stays unchanged…" | H |
| "Everything not covered by the mask remains the untouched original photograph…" | R |
| "The mask is in place by {settle_s} seconds…" | H + C |
| `negative:` "camera motion, … pale colors, … transparency" | N |

## 3. Bridging path: paper → base (6 runs)

Fewer arms than segmentation's path, in two ways:

- **The task slots (R, T, E) move one at a time.**
- **The core (P, C, H) moves as one block (K1).** These paragraphs are the wording
  shared by all the overlay probes, and segmentation's path already measured them
  one at a time (P1, C1, H2 there).

| arm | change from the previous arm | slot |
|---|---|---|
| `paper` | adapted segmentation prompt, green background | — |
| `R1` | "the background fades to green" → the base's rest paragraph | R |
| `T1` | "the part … that a robot would grasp" → the base's target definition and "covers only that graspable region, not the whole object" | T |
| `E1` | "a flat color … perfectly preserving its silhouette" → opaque mask, "sits precisely on its outline", "one flat, fully saturated color…" | E |
| `K1` | framing, scene bullets and Camera list → the base's P, C and H paragraphs, in prose. The text now equals the base byte for byte (checked) | P + C + H (core) |
| `base` | the base's negatives added (`prompt.yaml`) | N |

- Each arm file (`modules/grasp_region_seg/arms/`) carries its word diff against
  the previous arm. Every diff was checked.
- Arms `paper`–`K1` send no negative.
- **Gate waiver:** `paper` asks for a green background, so the scene-preservation
  gate rejects it by design. As in segmentation, the gate is waived for that arm
  only. From R1 on, the background must stay, and the gate applies.

## 4. Improvement arms (from the base, one slot each)

These were written before the base was run, so they come from evidence already in
hand rather than from a base diagnosis:

- the segmentation lessons: say what you want, not what to avoid; "mask" beats
  "paint";
- what the metric punishes (README): the whole object scores 0, and a disk on the
  object's centre scores 0.34 on frames whose grasp is off-centre;
- the demonstrations themselves: the finger gap at the grasp is 5–15 mm for bowls,
  mugs and bottles, so the grasp is on a rim, neck or wall, not the middle.

| arm | slot | change | expected failure it targets |
|---|---|---|---|
| `T2` | T (extent) | add "It is a small patch, about the width of a gripper finger, where the fingers would touch the object." | whole object or a large blob painted; `tightness` and `focus` collapse |
| `T3` | T | add "This is usually a thin part the fingers can pinch: a section of the rim of a bowl or mug, the neck of a bottle, or the upper sides of a box or carton." | the mark sits on the object's middle, away from the demonstrated grasp |
| `E2` | E | "one flat, fully saturated color" → "… magenta" | the mark comes out in the object's own hue or pale, and does not decode |

- **E2's contract with the decoder:** `color_mask` with `hue: null` keeps any
  saturated hue that differs from the input. Magenta passes unchanged, so the
  decoder needs no change.
- **T3 is a prior about LIBERO's grasps.** It names typical parts by object type,
  never by frame, but it may still fit this dataset's demonstrations more than
  grasping in general. Report that caveat with any gain.

## 5. How to run

```bash
python scripts/run_gen.py  --config configs/gen.libero.yaml --arm paper   # and R1, T1, E1, K1, base, T2, T3, E2
python scripts/run_eval.py --config configs/eval.libero.yaml --arm paper
```

Restrict `modules.include` to `grasp_region_seg` to keep a run to this probe.
**9 runs × 4 frames = 36 clips.**

## Results log

**Run 2026-11-10.** 9 arms × 4 tune frames, seed 0. 35 of 36 clips were made.

- **Missing: E2 on libero_10.** It was refused twice with HTTP 429, "Your project has
  exceeded its monthly spending cap". It was not generated, and no further clip can
  be made until the cap is raised (ai.studio/spend).
- **Invalid, not regenerated: R1 on spatial.** It marked nothing. The regeneration on
  seed 1000 (§6.1 of the segmentation doc) is blocked by the same cap.
- **paper:** 3 of 4 clips fail the scene gate (green background), as expected. The
  gate is waived for this arm.
- **Means** count the invalid R1 clip as 0. R1's mean over its 3 valid clips is
  0.205.
- **Contact sheet:** `results/ablations/grasp_region_seg/sheet.png`. The first row
  is the ground truth: the object mask plus the demonstrated grasp, with a 2 cm
  circle.

### Scores (last frame, mean of 4 frames)

| arm | slot | `grasp_score` | `hit` | `focus` | `on_object` | `coverage` | `tightness` | Δ vs previous | valid |
|---|---|---|---|---|---|---|---|---|---|
| `paper` | — | 0.009 | 1.00 | 0.19 | 0.52 | 0.86 | 0.27 | — | 1/4 (waived) |
| `R1` | R | 0.153 | 0.75 | 0.29 | 0.75 | 0.46 | 0.68 | +0.144 | 3/4 |
| `T1` | T | 0.134 | 0.75 | 0.24 | 0.65 | 0.46 | 0.78 | −0.019 | 4/4 |
| `E1` | E | 0.019 | 1.00 | 0.27 | 0.66 | 0.85 | 0.29 | **−0.116** | 4/4 |
| `K1` | P+C+H | 0.026 | 1.00 | 0.39 | 0.97 | 0.97 | 0.06 | +0.007 | 4/4 |
| `base` | N | 0.034 | 1.00 | 0.37 | 0.91 | 0.91 | 0.17 | +0.008 | 4/4 |
| `T2` | T (extent) | **0.157** | 0.75 | 0.39 | 0.98 | 0.62 | 0.59 | **+0.123 vs base**, better 3/4 | 4/4 |
| `T3` | T | 0.063 | 1.00 | 0.37 | 0.87 | 0.90 | 0.21 | +0.030 vs base, better 3/4 | 4/4 |
| `E2` | E | 0.077 | 1.00 | 0.40 | 0.81 | 0.92 | 0.15 | +0.063 vs base on 3 frames, better 2/3 | 3/3 |

Per frame `grasp_score` (spatial / object / goal / libero_10):

| paper | R1 | T1 | E1 | K1 | base | T2 | T3 | E2 |
|---|---|---|---|---|---|---|---|---|
| .01/.00/.02/.00 | .00*/.19/.08/.34 | .25/.00/.19/.09 | .00/.03/.04/.00 | .00/.02/.08/.00 | .00/.02/.02/.09 | .14/.43/.07/.00 | .13/.05/.07/.00 | .00/.21/.02/— |

\* invalid (nothing marked).

### Diagnosis by slot (from the contact sheet)

1. **The whole object is painted (T extent). This is the dominant failure.**
   - From E1 on, nearly every clip fills the entire bowl, carton, bottle or mug.
     Coverage is 0.85–0.97, so `tightness` (zero once the whole object is painted)
     drops to 0.06–0.29. Every grasp is then a `hit` but scores almost nothing.
   - The ground truth shows why that is wrong. The demonstrated grasps sit on the
     bowl's rim, the carton's top, the bottle's upper part and the mug's rim.
2. **E1 is where it starts (−0.116), and its wording is the likely cause.**
   - E1 replaces "perfectly preserving its silhouette" with "The mask … sits
     precisely on its outline". "Its outline" can be read as the *object's*
     outline, which asks for the whole object.
   - Before E1, R1 and T1 paint parts more often: R1 the mug's handle (0.34), T1
     part of the bowl rim (0.25).
   - This is a T-extent problem caused by E wording, and it is the next arm to run
     (below).
3. **T2's size anchor partly works.** "A small patch, about the width of a gripper
   finger" cuts coverage from 0.91 to 0.62. On the object frame it gives a small mark
   on the carton's top (0.43), the best clip of the run. But it still paints the
   whole bowl on spatial, and on libero_10 the mark misses the rim (hit 0).
4. **T3's part prior barely helps** (+0.030). The model still fills the whole object.
   On the object frame it draws a gripper-like outline rather than a patch.
5. **E2 (magenta)** decodes fine but fills the whole object, like base. The colour
   was not the problem: no clip in any arm failed to decode for colour.
6. **Other failures:**
   - **paper:** the green background restyles the scene, and it adds
     detection-style corner brackets (also in T1 and E1 on libero_10), probably
     from the "segmentation" framing.
   - **base, spatial:** a camera **tripod** appears on the table, likely the core
     sentence "locked off on a tripod" read literally. It happened once; note it for
     the core.
   - **T1, object:** marks the wrong object (hit 0).

### Decision

By the framework's rule, **T2 is adopted**. It is the highest improvement arm (0.157
against base 0.034) and better on 3 of 4 frames. Later arms are written from T2.

Caveats:

- The scores are near zero, so a few clips drive the means. T2's gain is mostly the
  object frame (0.02 → 0.43). Segmentation's held-out run showed per-frame swings of
  this size between prompts by chance.
- On the bridging path, the measurable step is E1's drop. P, C and H as a block (K1)
  and the negatives change almost nothing.

### Round 2 arms (written from T2, 2026-11-10; not yet run)

Both arms are in `modules/grasp_region_seg/arms/`, word-diffed against T2 and
dry-run.

| arm | slot | change from T2 | targets |
|---|---|---|---|
| `T4` | T (extent) | "and sits precisely on [-its outline.-]{+that part of the object's surface.+}" | E1's wording, which ties the mask to the object's outline (finding 2) |
| `E3` | E | "mask" → "dot" throughout, and "[-An opaque colored mask-]{+A solid, opaque, round colored dot+}", "a small [-patch,-]{+dot,+} about the width of a gripper finger" | Whole-object fills: a dot has no outline to snap to the object's silhouette. This is the answering-by-marking convention (§1). |

**E3's contract with the decoder is unchanged.** At the grasp, LIBERO's scale is
730–1090 px/m. A finger-wide (~2 cm) dot is therefore about 250–1500 px, above
`color_mask`'s `min_area_frac` of 0.0002 (~184 px). The README's synthetic tests
score a small region on the grasp at about 0.99.

**Pending clips:** E2 on libero_10 (1 clip), R1 on spatial on seed 1000 (1 clip),
and T4 and E3 (8 clips). All of them are in `scripts/run_pending_ablations.sh`. To
score:

```bash
venv/bin/python scripts/ablation_table.py grasp_region_seg grasp_score base T2 T3 E2 T4 E3
```

R1's seed-1000 clip is scored under `eval/R1/r1000/` and replaces the invalid seed-0
clip in R1's spatial cell.

### Round 2 results (2026-11-11)

All clips are valid, scored as above. The cap-blocked E2 clip and R1's seed-1000
regeneration are now included:

- **E2:** the libero_10 clip scores 0.03, so E2 is 0.065 over 4 frames (was 0.077 over 3).
- **R1:** the seed-1000 spatial clip is valid but scores 0.001, so R1 is **0.154** over
  4 valid clips.

| arm | from | `grasp_score` | per frame (spatial / object / goal / libero_10) | `coverage` | `focus` | `on_object` | `hit` |
|---|---|---|---|---|---|---|---|
| `T2` (current best before) | base | 0.157 | .14 / .43 / .07 / .00 | 0.62 | 0.39 | 0.98 | 0.75 |
| `T4` | T2 | **0.168** | .13 / .53 / .01 / .00 | 0.19 | 0.43 | 0.53 | 0.75 |
| `E3` | T2 | 0.072 | .00 / .00 / .29 / .00 | 0.10 | 0.36 | 0.71 | 0.25 |

- **T4 fixes the extent.** Coverage drops from 0.62 to 0.19: the bowl gets an arc on its
  rim, the carton a patch on its top (0.53, the best clip so far). On goal the mark
  lands on the robot's fingers rather than the bottle (`on_object` 0.04).
- **E3's dots are small and on plausible grasp spots,** but 2–3.4 cm from the
  demonstrated grasp, just outside the 2 cm tolerance (`hit` 0.25). They land on the
  bowl's rim, the carton's middle, the bottle's neck (0.29) and beside the mug's
  handle. The README's caveat applies: a demonstration is one valid grasp, so this is
  a lower bound.
- **Decision:** by the rule, **T4 is adopted** (0.168 > 0.157). The margin (+0.011) is
  far inside the arm-to-arm spread, so T2 and T4 should be treated as tied. Confirm
  T4 against the base on the held-out frames.

Contact sheets: `results/ablations/grasp_region_seg/sheet.png` (all arms, with the
ground truth) and `sheet_T2_T4_E3.png`.

## Metric revision (2026-11-11)

### Why

The rigour checks ([rigour_checks.md](rigour_checks.md) §3) scored reference answers
under `grasp_score` on all 20 LIBERO frames:

| answer | `grasp_score` |
|---|---|
| dot (2 cm radius) on the demonstrated grasp | 0.748 |
| top quarter of the object (knows the object, not the grasp) | 0.665 |
| whole object | 0.004 |

The metric could not tell knowing the grasp from finding the object and painting
its top. Two parts of it caused this:

- **`on_object` penalises a correct mark.** Demonstrated grasps sit on rims and
  edges, so a patch centred on the grasp lies partly off the object. Clipped to
  the object, the same patch is a thin sliver: 27–62 px on 4 frames.
- **`hit` is all or nothing at 2 cm.** A mark 2.5 cm away scores 0, the same as a
  mark on the wrong object. E3's dots, 2–3.4 cm from the grasp, scored 0 on 3 of 4
  frames.

A third problem was in the decoder. At `min_area_frac` 0.0002 (184 px), a
finger-wide (1 cm radius) mark drawn exactly on the grasp was dropped on 5 of 20
frames after the median blur and cleanup.

### The new headline: `grasp_loc`

For each pixel, take its distance from the grasp centre beyond the existing 2 cm
tolerance (`tolerance_m`, the same one `hit` uses; zero inside it). Then

    grasp_loc = 1 − (mean of that distance over the painted pixels)
                  / (mean of that distance over the object's pixels)

clipped to [0, 1]. Nothing painted scores 0.

- **1:** all the paint is within 2 cm of the grasp.
- **0:** the paint is on average no nearer the grasp than the object as a whole,
  which is what painting the whole object gives. So the score is *relative to
  having found the object*.
- **Partial credit:** a mark 3 cm off earns some; a mark on the far side of the
  object earns none.
- **Paint off the object:** not penalised within 2 cm. Beyond that it counts by
  its distance.
- **One parameter**, the existing 2 cm tolerance. `focus_m` and `free_coverage`
  no longer affect the headline.

`object_excess_m`, the denominator in metres, says how much a frame can separate
grasp knowledge from object finding. It is 0.6–1.2 cm on 3 of the 20 frames (small
objects, nearly all within 2 cm of the grasp) and 2.0–4.9 cm on the rest.

`grasp_score` and its parts are still computed and reported, so earlier tables
stay comparable.

### How it was chosen

Only on synthetic answers drawn from the ground truth (20 LIBERO frames), before
looking at any model clip under the new score:

| answer | `grasp_loc`, tolerance 1 / **2** / 3 cm | through the decoder, floor 184 px → 92 px (frames decoded) | old `grasp_score` |
|---|---|---|---|
| dot, 1 cm radius, on the grasp | 1.000 / **1.000** / 1.000 | 0.750 (15/20) → 1.000 (20/20) | 0.821 |
| dot, 2 cm radius, on the grasp | 0.881 / **1.000** / 1.000 | 1.000 → 1.000 | 0.746 |
| whole object | 0.000 / **0.000** / 0.000 | 0.003 → 0.003 | 0.000 |
| dot, 1 cm, on the object's centroid | 0.239 / **0.351** / 0.531 | 0.283 (16/20) → 0.352 (20/20) | 0.150 |
| top quarter of the object | 0.394 / **0.482** / 0.562 | 0.486 → 0.486 | 0.665 |
| dot, 1 cm, on the object point nearest the gripper | 0.406 / **0.525** / 0.675 | 0.479 (16/20) → 0.528 (20/20) | 0.206 |

- **The tolerance stays 2 cm**, the value `hit` already used; it was not tuned.
  At 1 cm a 2 cm-radius mark on the grasp would lose credit.
- **The decoder floor is halved** to 0.0001 of the frame (92 px), and the
  compliance gate with it, so a mark the decoder reads is never "nothing marked".
  At 92 px no paint is found on the 20 untouched frames, or on frame 0 of any of
  the 45 grasp clips. No existing clip's `grasp_score` changed.
- **Through video** (H.264 at the generator's bit rate, with and without grain),
  both ideal dots score 0.999–1.000 and pass every gate on 20 of 20 frames.

**What the new metric cannot remove.** On LIBERO the demonstrated grasp is usually
near the top of the object and near the gripper. A guess that knows the object and
the gripper's position scores about 0.5 (top quarter 0.49, nearest the gripper 0.53, on all 20 frames). That is a
property of the dataset, not of the metric. It is the floor a model has to beat,
and it is reported beside every grasp result (`results/rigour/references.md`).

### The tune arms rescored

Same clips, same frames, last frame; the eval was rerun for every arm. R1 uses its
valid seed-1000 clip on spatial.

| arm | old `grasp_score` | `grasp_loc` | per frame `grasp_loc` (spatial / object / goal / libero_10) | best-frame `grasp_loc` |
|---|---|---|---|---|
| `paper` | 0.009 | 0.014 | .00 / .00 / .06 / .00 | 0.039 |
| `R1` | 0.154 | 0.050 | .01 / .00 / .00 / .19 | 0.395 |
| `T1` | 0.134 | 0.050 | .20 / .00 / .00 / .00 | 0.278 |
| `E1` | 0.019 | 0.000 | .00 / .00 / .00 / .00 | 0.372 |
| `K1` | 0.026 | 0.036 | .10 / .02 / .03 / .00 | 0.378 |
| `base` | 0.034 | 0.057 | .19 / .02 / .02 / .00 | 0.060 |
| `T2` | 0.157 | 0.149 | .07 / .52 / .01 / .00 | 0.256 |
| `T3` | 0.063 | 0.042 | .17 / .00 / .00 / .00 | 0.234 |
| `E2` | 0.065 | 0.086 | .01 / .31 / .00 / .03 | 0.405 |
| `T4` | 0.168 | 0.162 | .28 / .37 / .00 / .00 | 0.332 |
| `E3` | 0.072 | **0.292** | .39 / .20 / .48 / .10 | 0.299 |

On the same 4 frames, the object-only guesses score 0.32 (centroid dot), 0.62 (top
quarter) and 0.71 (nearest the gripper).

- **E3 moves from sixth to first.** Its dots land 2–3.4 cm from the grasp; the old
  `hit` gave them nothing, `grasp_loc` gives partial credit. It is the only arm that
  scores on every frame (0.10–0.48).
- **No arm beats a guess that knows only the object.** The best, E3 (0.292), is
  below the centroid dot (0.32) and well below the top-quarter and nearest-gripper
  guesses. On these frames there is no evidence that Omni marks the grasp better
  than it would by finding the object.
- **Many arms paint near the grasp mid-clip and then spread.** Best-frame scores
  run to 0.37–0.41 for R1, E1, K1 and E2, against 0.00–0.09 on the last frame.

### What this means for the decision

**This is post hoc.** The metric was chosen on synthetic answers only, but after
the round-2 tune results under the old metric were known. The adoption rule
picks T4 under `grasp_score` and E3 under `grasp_loc`. With 4 frames, one seed and
arm-to-arm swings of this size, neither ranking is reliable, so the choice of arm
to confirm should be made openly, before any held-out grasp clip:

1. **E3 against base** under `grasp_loc` (32 clips): follows the rule under the
   frozen metric.
2. **E3 and T4 against base** (48 clips), with a Holm correction for two
   comparisons: does not depend on which metric picked the arm.

Either way, the held-out report should add each arm's difference from the
nearest-gripper and top-quarter guesses on the same frames. Without that, a gain
over base cannot be read as grasp knowledge.

Scoring:

```bash
venv/bin/python scripts/ablation_table.py grasp_region_seg grasp_loc base T2 T4 E3
```

## Held-out confirmation (2026-11-12)

**The comparison was fixed before any held-out clip.** On 2026-11-11 the user chose
E3, under the frozen `grasp_loc` headline (option 1 above). A held-out T4 run had
been started and was stopped after 6 of 16 clips. Those 6 clips were deleted unscored
on the user's instruction. The 16 base clips made in that run were kept.

**Setup:**

- base vs E3 on the 16 frames in `configs/libero_test_ids.txt`, seed 1, paired by
  frame;
- `configs/ablations/confirm.{gen,eval}.grasp_region_seg.yaml`;
- the current decoder (`min_area_frac` 0.0001, `min_change` 40).

**Validity:** 10 of 16 frames were valid in both arms on the first attempt. On the 6
others, one arm changed the scene too much (base on 2, E3 on 4). Both arms on those
frames were regenerated on a shared seed, 1001, and all 12 were valid, giving 16/16
pairs. 44 clips in all.

**Primary result (last frame, 16 pairs):**

| | base | E3 | E3 − base | 95% bootstrap CI | Wilcoxon p | d_z | E3 better/worse |
|---|---|---|---|---|---|---|---|
| **`grasp_loc`** | 0.026 | **0.456** | **+0.430** | [+0.278, +0.584] | **0.001** | 1.32 | 14/0 |
| best-frame `grasp_loc` | 0.171 | 0.459 | +0.287 | [+0.073, +0.492] | 0.016 | 0.65 | 12/2 |
| `grasp_score` (old headline) | 0.032 | 0.341 | +0.309 | [+0.198, +0.421] | 0.001 | 1.32 | 14/1 |
| `coverage` (lower is tighter) | 0.659 | 0.161 | −0.498 | [−0.699, −0.273] | 0.003 | | 12/3 |
| `hit` | 0.750 | 0.875 | +0.125 | [−0.125, +0.375] | 0.317 | | 3/1 |

**E3 is confirmed against the base.** Per suite (4 frames each): libero_10 0.011 →
0.447, goal 0.053 → 0.601, object 0.024 → 0.332, spatial 0.016 → 0.442. This is the
first held-out confirmation in the project. Segmentation's H1 and depth's N2 both
failed theirs.

**Against guesses on the same 16 frames** (`grasp_loc`; each guess is given the true
object mask, and the nearest-gripper guess also the gripper's image position; the
E3 clips are those used in the paired test):

| guess | `grasp_loc` | E3 − guess | 95% CI | p | E3 better/worse |
|---|---|---|---|---|---|
| whole object painted | 0.003 | +0.453 | [+0.294, +0.616] | <0.001 | 14/2 |
| dot on the object's centroid | 0.360 | +0.096 | [−0.147, +0.328] | 0.43 | 8/8 |
| top quarter of the object | 0.451 | +0.005 | [−0.206, +0.227] | 0.78 | 6/10 |
| dot on the object point nearest the gripper | 0.484 | −0.028 | [−0.214, +0.161] | 0.83 | 8/6 |

**Reading:**

- **What the gain over base comes from.** The base paints the whole object, which
  `grasp_loc` scores near 0. E3 marks a small part of it.
- **What it does not show.** E3 does not beat the guesses that know the object but
  not the grasp. So these frames give no evidence that the model knows *where* the
  demonstrated grasp is, beyond picking a part near the top or near the gripper.
- **Caveat:** the demonstration is one valid grasp, so a mark on another valid grasp
  scores low (README). The guess comparison is the honest bound on what the prompt
  adds.

Files in `results/confirmation/grasp_region_seg/libero/`: `confirm_stats.txt`,
`vs_guesses.txt` and `vs_guesses.py`. Recompute with
`venv/bin/python scripts/confirm_stats.py grasp_region_seg E3 grasp_loc …`.

### Visual check of the held-out clips (2026-11-12)

The last frames of every held-out clip, including all attempts, were checked by eye
(`sheet_all.png`, `sheet_retried.png` in `results/confirmation/grasp_region_seg/libero/`):

- **The scores match what is visible.** Base paints the whole object (mugs, bowls,
  bottles, cartons) on almost every frame. E3 draws a small round dot on the named
  object, usually on its rim or top.
- **The scene gate fails in both directions:**
  - **Missed:** some clips it passed are visibly broken. A camera tripod appears on
    the table (base, goal ep0674, seed 1001). The bowl is replaced by a green face
    mask (base, spatial ep1285, seed 1001). A second robot arm appears at the top
    left (E3, spatial ep1285, seed 1001). These are local additions under the 15%
    threshold.
  - **Over-strict:** some clips it rejected look correct at this size. For example,
    E3's seed-1 clips on goal ep0751 and object ep1191 show a clean dot and an
    unchanged scene; they changed 0.19 and 0.24 across the frame, probably a global
    shift in brightness or colour.
- **Consequence:** validity partly reflects the gate rather than visible failure. The
  regeneration protocol treats both arms alike, so the paired comparison stays
  fair, but "invalid" should not be read as "visibly broken". The tripod and face
  mask are the same literal readings seen on tune frames ("on a tripod" in the core
  camera sentence; "mask").
