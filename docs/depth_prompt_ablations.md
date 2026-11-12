# Depth prompt ablations

**Status 2026-11-10: finished.** 19 arms run on the 4 tune frames (76 clips, seed
0) and scored in both spaces. N2 was the best arm on the tune frames but was **not
confirmed** on the 16 held-out frames: δ1 +0.025, p 0.98; disparity δ1 −0.035, p
0.43. Both base and N2 beat a no-image gradient on held-out frames (p ≤ 0.02). The
results log is at the end. The method is [prompt_framework.md](prompt_framework.md), and the setup (Gemini
Omni 1.2 Flash, 4 LIBERO tune frames, seed 0, exact simulator depth) is the same as
for segmentation ([segmentation_prompt_ablations.md](segmentation_prompt_ablations.md)).

## 1. The published baseline

Depth is one of Wiedemer et al. (2025)'s failure cases (App. D, Fig. 62). There is
no quantitative result, only a qualitative one: Veo 3 produces little more than a
binary foreground/background split, and struggles to use a colour map it is given.
Their prompt:

> The image transitions to a depth-map of the scene: Darker colors represent pixels
> further from the camera, lighter colors represent pixels closer to the camera. The
> exact color map to use is provided on the right side of the image. Static scene, no
> pan, no zoom, no dolly.

**The legend.** The colour map is a visual prompt: a vertical magma bar pasted on
the input image's right edge, light (near) at the top and black (far) at the bottom.

- `scripts/make_depth_legend_frames.py` draws the same bar into the right-hand black
  bar of the LIBERO frames (x 1200–1239), covering no scene pixel. Those pixels are
  NaN in the depth ground truth and never scored.
- The output is `data/libero/frames_depth_legend/`.
- The `paper` arm runs on those frames (`configs/gen.libero.depth_legend.yaml`);
  every other arm uses the plain frames.
- Magma's brightness rises steadily from far to near, so the grayscale decoder reads
  it unchanged.

## 2. The base prompt, by slot

| paragraph | slot |
|---|---|
| "This photograph becomes a single grayscale depth map of exactly the same scene…" | P (transform family) |
| "Depth is shown only as brightness… near white… grading smoothly to black…" | E |
| "The image is completely grayscale. There is no color anywhere, … no colormap…" | E (rules) |
| "The camera is locked off on a tripod…" | C |
| "The depth map is fully formed by {settle_s} seconds and then holds…" | H |
| `negative:` "color, colored, tint, colormap, rainbow, …" | N |

## 3. Bridging path: paper → base

The base prompt was adapted from Wiedemer et al.'s prompt. This path measures the
effect of each change by applying them one at a time, each step building on the
previous one. It measures changes that were designed beforehand; it does not
search for them. If a step lowers the score, the path still continues to the base,
and the drop is reported.

| arm | change from the previous arm | slot |
|---|---|---|
| `paper` | their prompt, verbatim, with the magma legend | — |
| `V1` | legend removed, with its sentence | V (visual cue, not a slot) |
| `E1` | their colour sentence → the base's brightness paragraph | E |
| `P1` | "The image transitions to a depth-map of the scene." → the base's preservation paragraph | P |
| `E2` | the base's grayscale-rules paragraph added | E |
| `C1` | "Static scene, no pan, no zoom, no dolly." → the base's camera sentence | C |
| `H1` | the base's timing and hold sentences added; the text now equals the base byte for byte | H |
| `base` | the base's negatives added (`prompt.yaml`) | N |

Each arm file carries its word diff against the arm before it, and each diff was
checked. Every step before `base` sets `negative: ""`. **8 runs × 4 frames = 32
clips.**

## 4. Improvement arms (from the base)

These are hypotheses. Per the framework, run the base first, diagnose its failures
by slot, then run the arms that match the commonest failure. Each changes one slot
of the base and carries its word diff.

| arm | slot | change | hypothesis |
|---|---|---|---|
| `E3` | E | delete "There is no color anywhere, no red or green or blue tint, no colormap, no rainbow, no red-to-blue scale." | Naming what to avoid produced it in segmentation (T1, E4). |
| `N2` | N | drop `colormap, rainbow, red blue gradient, jet, turbo, viridis` from the negatives | The same; Omni appends the negatives to the prompt text. |
| `E4` | E | add "The table surface itself grades smoothly from bright along the bottom edge of the image to dark at the back." | Targets the paper's binary foreground/background failure. |
| `E5` | E | brightness paragraph restated as a lamp at the camera whose light fades with distance | The paper's "Realism" advice (Sec. C); light falloff is a real depth cue. |
| `E6` | E | prefix "First, every surface turns into plain matte grey clay, with no color or texture. Then depth…" | Separates removing texture from encoding depth. |
| `H2` | H | "The photograph becomes the depth map within the first second, then holds…" | The paper's "Speed control" advice: instant changes for image-to-image tasks. |
| `E7` | E | add "Brightness falls off with distance: a surface twice as far away is half as bright." | States disparity encoding; should raise `disp_delta1` if followed. |
| `E8` | E | add "Brightness falls evenly with distance: equal steps in distance are equal steps in brightness." | States linear-depth encoding; should raise `delta1` if followed. |

The base says only "near white… grading smoothly to black", so it leaves the
encoding open; E7 and E8 each state one. Together they test whether the model can
follow an encoding instruction at all. Compare each against the base in both spaces:
following E7 should move `disp_delta1` up relative to `delta1`, and E8 the reverse.

## 5. Two alignments, both reported (decided 2026-11-09, before any clip was scored)

**What the decoder does.** `gray_depth` reads brightness as depth (depth = 1 −
luminance). The metric fits a scale and shift and then computes δ1, AbsRel and the
rest.

**What changed.** That fit is now done in two spaces, and both are reported:

| scores | fit | convention |
|---|---|---|
| `delta1`, `abs_rel`, … | prediction fitted to ground-truth **depth** | the original setting |
| `disp_delta1`, `disp_abs_rel`, … | prediction fitted to ground-truth **inverse depth** (1/depth), then inverted for the errors | the relative-depth standard (Ranftl et al., MiDaS; Depth Anything) |

- The fit is affine, so the decoder's brightness direction does not matter and the
  decoder is unchanged. Only the space of the fit differs.
- In disparity space the aligned disparity is clipped to [1/max_depth, 1/min_depth]
  before inverting.
- The best frame is still chosen by `delta1`. Its `best_disp_*` scores are taken at
  that frame.
- `modules/depth_perception/metrics.py` gained `space` and `prefix`.
  `probe.yaml` lists the metric twice.
- **Fix, 2026-11-10 (found when scoring these clips, before any result was read).**
  In depth space the affine fit can place some pixels at or below zero depth. Those
  made `rmse_log` and `silog` NaN, which crashed the report, and their negative
  ratio was counted as a δ1 hit. The aligned depth is now clipped to
  [min_depth, max_depth], as the disparity branch already did. All 16 arms were
  rescored with the fix. Earlier depth-space scores (the old RoboTrustBench runs)
  can differ slightly where the fit went negative.

**Check, on simulator depth from two LIBERO frames.** A map drawn linear in depth
and one drawn linear in inverse depth each score perfectly in their own space and
are penalised in the other:

| frame | drawn as | `delta1` | `disp_delta1` |
|---|---|---|---|
| spatial ep1520 (0.58–3.07 m) | linear in depth | 1.000 | 0.808 |
| | linear in inverse depth | 0.625 | 1.000 |
| libero_10 ep0231 (0.57–2.70 m) | linear in depth | 1.000 | 0.798 |
| | linear in inverse depth | 0.759 | 1.000 |

So under the depth-only metric, a perfect disparity map would have scored as low as
δ1 0.63. The eval was also run end to end on two old RoboTrustBench clips, and both
score sets appear.

**Which is primary.** The prompt does not say which encoding it wants (§4: E7, E8),
so neither space is privileged. Report both. An arm that wins in only one space
has changed the encoding, not the quality.

## 6. How to run

```bash
python scripts/run_gen.py  --config configs/gen.libero.depth_legend.yaml --arm paper
python scripts/run_gen.py  --config configs/gen.libero.yaml --arm V1     # and E1, P1, E2, C1, H1, base
python scripts/run_eval.py --config configs/eval.libero.yaml --arm V1
```

To keep a run to depth only, restrict `modules.include` to `depth_perception`.

## Results log

**Run 2026-11-09/10.** 16 arms × 4 tune frames, seed 0, one repeat: 64 clips.

- Five clips failed on the server (job status `failed`; no clip, not charged): E1 goal,
  P1 object and goal, E2 goal and libero_10. They were resent with the same request
  and seed, and all succeeded.
- All 64 clips are valid.
- Scores are on the last frame. Means are over the 4 frames.
- The contact sheets (input, GT, then one row per arm) are
  `results/ablations/depth_perception/sheet_bridging.png` and `sheet_improvement.png`.
  The scripts that made them are in the same folder.

### Reference points

What a prediction scores without using the image, on the same 4 frames:

| prediction | `delta1` | `disp_delta1` |
|---|---|---|
| constant depth | 0.230 | 0.402 |
| input photo decoded as depth (no change) | 0.311 | 0.469 |
| **row gradient** (far at the top, near at the bottom; no image used) | **0.585** | **0.666** |

The row gradient is the honest floor. The camera looks down at a table, so "lower in
the image is nearer" already gets most of the score. An arm only shows depth
understanding to the extent it beats 0.585 / 0.666.

### Bridging path: paper → base

| step | arm | slot changed | `delta1` | `disp_delta1` | Δ vs previous | colour share |
|---|---|---|---|---|---|---|
| 0 | `paper` | — | 0.594 | 0.729 | — | 0.99 |
| 1 | `V1` | V (legend removed) | 0.688 | 0.878 | +0.094 / +0.149 | 0.00 |
| 2 | `E1` | E (brightness paragraph) | 0.622 | 0.799 | −0.066 / −0.079 | 0.04 |
| 3 | `P1` | P (preservation paragraph) | 0.611 | 0.680 | −0.011 / −0.118 | 0.01 |
| 4 | `E2` | E (grayscale rules) | 0.757 | 0.901 | +0.146 / +0.221 | 0.02 |
| 5 | `C1` | C (camera sentence) | 0.678 | 0.752 | −0.079 / −0.149 | 0.00 |
| 6 | `H1` | H (timing and hold) | 0.700 | 0.854 | +0.022 / +0.103 | 0.01 |
| 7 | `base` | N (negatives) | 0.685 | 0.788 | −0.015 / −0.067 | 0.01 |

Paper → base overall: **+0.091 / +0.059**.

### Improvement arms (each from base)

| arm | slot | `delta1` | `disp_delta1` | Δ vs base | frames better (δ1, disp) | decision |
|---|---|---|---|---|---|---|
| `base` | — | 0.685 | 0.788 | — | — | — |
| `E3` | E | 0.682 | 0.813 | −0.003 / +0.025 | 3/4, 3/4 | reject (one space) |
| `N2` | N | 0.718 | 0.855 | **+0.033 / +0.067** | 3/4, 2/4 | **adopt (rule); within noise** |
| `E4` | E | 0.630 | 0.849 | −0.055 / +0.061 | 1/4, 3/4 | reject (one space) |
| `E5` | E | 0.588 | 0.661 | −0.096 / −0.126 | 1/4, 2/4 | reject |
| `E6` | E | 0.741 | 0.768 | +0.056 / −0.020 | 3/4, 2/4 | reject (one space) |
| `H2` | H | 0.694 | 0.852 | +0.009 / +0.065 | 2/4, 3/4 | reject (below N2) |
| `E7` | E | 0.573 | 0.714 | −0.111 / −0.074 | 0/4, 2/4 | reject |
| `E8` | E | 0.643 | 0.789 | −0.041 / +0.001 | 1/4, 2/4 | reject |

**Decision.** By §5, an arm counts only if it improves both spaces. Two arms do: N2,
and H2 with +0.009 in δ1. N2 is the larger, so the framework's rule adopts N2. The
next paragraph shows this gain is inside the arm-to-arm spread, so N2 is a screening
pick, not a finding.

### How large a difference is noise

A different prompt gives a different sample, even with the same seed. The 15
grayscale arms give an estimate of how much the score moves from that alone:

- the 4-frame mean has an SD of 0.054 (δ1) and 0.072 (disp) across arms;
- the difference between two random arms exceeds 0.127 (δ1) and 0.174 (disp) 10% of
  the time.

No improvement arm differs from base by more than that. On the bridging path, only
P1 → E2 comes close (+0.146 / +0.221), and the next step, C1, gives about half of
it back. With one seed on 4 frames, **none of the wording changes is shown to matter
for depth.** The one exception is the legend: it changes the output qualitatively
(next section).

### Diagnosis by slot (from the contact sheets)

1. **Omni does not show the paper's failure; it shows a different one.**
   - Veo 3's failure was a binary foreground/background split. Omni's output is not
     binary: about 11 grey levels hold over 2% of pixels each, and the two largest
     hold only about 0.33 of the image.
   - Instead, Omni produces a **grey clay render**. It desaturates the photo and
     keeps its shading, so brightness follows surface tone and lighting, not
     distance. The dark cabinet stays dark although it is the nearest object, and
     the back wall goes black.
   - The rank correlation with true depth is 0.77–0.88. That is largely the
     row-gradient effect above, plus the darkened background.
   - This is an **E failure that wording does not fix**. E5 (lamp falloff) and E6
     (grey clay first) were written for it, and neither helped. E6 makes the clay
     render explicit and scores much like base.
2. **The colour-map legend (paper arm).** The model does not encode depth with the
   magma bar. It **tints the whole photo magma**: colour share 0.99, against ≤0.04
   in every grayscale arm. This matches the paper's report that the model
   "struggles" with a given colour map. Removing the legend (V1) is the largest
   clean step on the path.
3. **Added objects near the camera (P).**
   - In 24 of the 60 grayscale clips, a new object appears at the bottom edge: a
     large white plate, a slab, a bottle or a bowl. E2 has none.
   - These are most likely a literal reading of "near is white".
   - They do not lower the score consistently. In clips with an added object, the
     score changes by −0.13 to +0.05 δ1 against clips of the same frame without
     one; only frame 3, where a slab covers the near table, drops clearly.
   - They are still a preservation failure, and a P-slot arm (e.g. "nothing new
     enters the scene") would target them.
4. **Encoding instruction is not followed (E7, E8).**
   - If E7 (disparity) were followed, `disp_delta1 − delta1` would rise; if E8
     (linear) were followed, it would fall.
   - Base's gap is 0.103, E7's is 0.141 and E8's is 0.146. Both moved the same way,
     and both lowered δ1.
   - Every arm scores higher in disparity space, and so does the row gradient
     (0.666 vs 0.585). Part of the gap therefore comes from the scene geometry, not
     from the encoding.
5. **C and H.** No camera motion or late transitions were seen. Every clip has
   settled by the last frame.

### Round 2: arms written from the diagnosis (2026-11-10)

The diagnosis named the clay render as the commonest failure (every clip). So three
arms target it, each changing one slot of the base, run on the same 4 frames with
seed 0. That is 12 clips, all valid. The contact sheet is `sheet_round2.png`.

| arm | slot | change | `delta1` | `disp_delta1` | Δ vs base | frames better | Δ vs N2 (current best) | decision |
|---|---|---|---|---|---|---|---|---|
| `P2` | P | "a single grayscale depth map of" → "the grayscale image of a depth camera, like a Kinect sensor, filming" | 0.708 | 0.838 | +0.023 / +0.051 | 3/4, 4/4 | −0.010 / −0.017 | reject |
| `E9` | E | add "A dark object and a light object at the same distance get the same shade: the brightness of the photograph plays no part." | 0.683 | 0.809 | −0.001 / +0.021 | 2/4, 2/4 | −0.035 / −0.046 | reject |
| `E10` | E | near and far anchored to scene surfaces: "the surface along the bottom edge of the image… is the brightest", "black at the wall at the back" | 0.663 | 0.779 | −0.021 / −0.008 | 1/4, 2/4 | −0.055 / −0.076 | reject |

- **No arm beats N2**, so nothing is adopted. All three changes are well inside the
  arm-to-arm spread (0.127 / 0.174).
- **P2** is the only one better than base on every frame in disparity space (4/4).
  It is the best candidate if repeats are run. It still adds objects near the
  camera, in 3 of 4 clips.
- **The clay render remains in all 12 clips.**
  - E9 lightens the dark cabinet in frame 1, but the table is still not graded.
  - E10 follows its anchor literally: in frame 3 it adds a bright strip along the
    bottom edge instead of grading the table.
- So the failure stays an E failure that single-slot wording does not fix with this
  model.

### Stage 3: confirmation (2026-11-10)

Base against N2, once, on frames neither was tuned on. This follows the
segmentation confirmation exactly:

- the 16 frames in `configs/libero_test_ids.txt`, one clip per arm (32 clips);
- **seed 1** (`--repeat 1`), shared by both arms, so each frame is a paired
  comparison;
- LIBERO only.

All 32 clips were generated on the first attempt, and all are valid. Results, the
stats scripts (`confirm_stats.py`, `vs_prior.py`) and the clips' scores are in
`results/confirmation/depth_perception/libero/`.

**Primary result (last frame, 16 frames, paired):**

| | base | N2 | N2 − base | 95% bootstrap CI | Wilcoxon p | d_z | N2 better |
|---|---|---|---|---|---|---|---|
| `delta1` | 0.668 | 0.694 | **+0.025** | [−0.053, +0.116] | 0.98 | 0.14 | 8/16 |
| `disp_delta1` | 0.788 | 0.754 | **−0.035** | [−0.091, +0.020] | 0.43 | −0.29 | 7/16 |
| `abs_rel` | 0.219 | 0.217 | −0.002 | [−0.045, +0.034] | 0.53 | | 6/16 |
| `disp_abs_rel` | 0.151 | 0.154 | +0.003 | [−0.018, +0.024] | 0.82 | | 7/16 |
| best-frame `delta1` | 0.693 | 0.716 | +0.023 | [−0.045, +0.098] | 0.86 | | 7/16 |
| best-frame `disp_delta1` | 0.743 | 0.748 | +0.005 | [−0.061, +0.064] | 0.82 | | 7/16 |

**N2 is not confirmed.** The tune-frame gain (+0.033 / +0.067) shrinks to +0.025 in
δ1 and reverses in disparity space. Every CI includes 0. As with segmentation's H1,
the tuning gain was selection on 4 frames × 1 seed.

Per suite (4 frames each; descriptive only), base → N2:

| suite | `delta1` | `disp_delta1` |
|---|---|---|
| libero_10 | 0.728 → 0.705 | 0.805 → 0.779 |
| goal | 0.460 → 0.619 | 0.684 → 0.713 |
| object | 0.925 → 0.882 | 0.957 → 0.928 |
| spatial | 0.561 → 0.570 | 0.707 → 0.595 |

**Secondary: does Omni encode depth at all?** Each prompt is compared, paired by
frame, against the row-gradient floor (§ Reference points), which uses no image.
This comparison was defined after the tune-frame analysis but before these test
clips were made.

| | score | row-gradient floor | difference | 95% CI | Wilcoxon p | above floor |
|---|---|---|---|---|---|---|
| base `delta1` | 0.668 | 0.575 | +0.093 | [+0.025, +0.156] | 0.018 | 12/16 |
| base `disp_delta1` | 0.788 | 0.654 | +0.135 | [+0.087, +0.180] | 0.0004 | 13/16 |
| N2 `delta1` | 0.694 | 0.575 | +0.119 | [+0.051, +0.181] | 0.005 | 13/16 |
| N2 `disp_delta1` | 0.754 | 0.654 | +0.100 | [+0.047, +0.153] | 0.005 | 13/16 |

On held-out frames, both prompts beat the no-image floor in both spaces. Omni's
output carries real depth information beyond "lower is nearer", even though it
looks like a grey clay render, not a depth map. The margin is modest, about 0.1 δ1.

### Summary

- On LIBERO tune frames, Omni's depth beats a no-image row gradient by about 0.1 δ1
  and 0.1–0.2 disparity δ1. It does this by rendering grey clay, not a depth map.
- The paper's colour-map legend makes the model tint the image instead of encoding
  depth.
- Within one seed, no single-slot wording change separates from sample noise. The
  rule adopts N2 as a screening pick.
- Held-out confirmation: N2 is not better than base. Both beat the no-image floor
  by about 0.1 δ1 / 0.1–0.14 disparity δ1. That is the reportable depth finding:
  Omni encodes some depth, and wording does not measurably change how much.
- The base prompt stays the depth prompt. To claim any prompt effect for depth, you
  would need more seeds per frame on the tune set, not more wording arms.
