# Segmentation prompt ablations

Goal: find the best prompt for the segmentation probe, show that it beats the current
one on frames it was not tuned on, and describe the framework the prompt is built from.

Status 2026-11-09: tuning is finished and **H1 is frozen as the segmentation
prompt** (`modules/instance_seg/arms/H1.yaml`). Six prompts were run on the 4 LIBERO
tune frames with Gemini Omni 1.2 Flash. H1 scored mean last-frame IoU 0.863 with
10 of 10 objects matched, against the base's 0.481 (Results log, below).
**On the 16 held-out frames H1 did not confirm** (Stage 3, below): base 0.511, H1
0.552, a difference of +0.041 with 95% CI [−0.145, +0.245] and p = 0.94. The
gain seen on the tune frames did not carry over. The
midterm numbers in §1 come from RoboTrustBench, made with Veo before the changes in §3.

## 1. Starting point

| measure (RoboTrustBench midterm run, 17 of 20 scored) | value |
|---|---|
| `mean_iou`, last frame | 0.114 (median 0.037) |
| `mean_iou`, best frame | 0.193 |
| Invalid clips | 6 of 17: 4 changed the scene too much, 2 marked nothing |
| Samples with no ground truth | 3: SAM 3 got "it" twice and "black hoodie" once |

The median is far below the mean, so a few good clips carry the result. Most clips
fail outright rather than being slightly wrong. That makes the first question "does
the model paint something decodable at all?", not "how sharp are the edges?".

These numbers predate the fixes in §3 (the decoder's change check, the revised
RoboTrustBench set), so they are not a baseline for the ablations; Stage 0 is.

## 2. The prompt framework

The method is in [prompt_framework.md](prompt_framework.md). In short:

- Each prompt is made of slots. P, C, H and the camera negatives are the core
  shared by all five probes; T, E and R are the probe's own.
- An arm is the base text with one slot reworded. Its header states the slot and
  the exact edit, and a word diff against the base confirms nothing else changed.
- Each arm targets the most frequent failure from the last diagnosis.

Segmentation is the pilot, so its core-slot results are carried over to the other
four probes.

| slot | segmentation base wording |
|---|---|
| **P** | "This photograph stays exactly as it is..." |
| **C** | "The camera is locked off on a tripod..." / "Static locked-off camera, one continuous shot..." |
| **H** | "Once a mask has appeared, it stays unchanged..." / "All masks are in place by {settle_s} seconds..." |
| **N** | the `negative:` list, appended to the prompt (Omni has no negative field) |
| **T** | "...on top of each of these objects — {relevant_objects} — ... covering it exactly. Each mask spreads outward..." |
| **E** | "an opaque colored mask" |
| **R** | "Everything not covered by a mask remains the untouched original photograph..." |

The encoding slot has to match the decoder, so an arm that changes it must state any
decoder setting it needs before its clips are generated.

## Results log

| step | arm | slot changed | targets failure | last-frame mean IoU | Δ vs current best | objects matched | decision |
|---|---|---|---|---|---|---|---|
| 0 | base | — | — | 0.481 | — | 4/10 | base |
| 1 | E1 | E | shaded recolouring, 6 of 10 objects | 0.713 | +0.232 | 7/10 | adopted |
| 2a | T1 (from E1) | T | spill onto the support surface (spatial) | 0.467 | −0.246 | 6/10 | not adopted |
| 2b | H1 (from E1) | H | paint fading at the end (goal) | 0.863 | +0.150 | 10/10 | adopted; current best |
| 3 | E4 (from H1) | E | uneven saturation inside masks (shading, pale rims) | 0.682* | −0.181 | 7/10 | not adopted; H1 stays current best |
| 4 | E5 (from H1) | E (term) | "mask" read as an object (face mask in E4) | 0.581* | −0.282 | 5/10 | not adopted; H1 stays current best |
| held-out | H1 vs base, 16 test frames, seed 1 | — | — | 0.552 vs 0.511 | +0.041 [−0.145, +0.245], p = 0.94 | 21/33 vs 19/33 | **not confirmed** |

\* After regenerating the one invalid clip (seed 1000), 2026-11-12. First scored with that clip included: E4 0.604, E5 0.441. Neither approaches H1 (0.863), so neither decision changes.

**Frozen: H1** (2026-11-09). Tuning stopped after E4 and E5. Both scored below H1,
and both lost on frames their edits did not address, so on 4 frames and one seed
further edits could no longer be told apart from chance. H1 is the base prompt
with two edits: E1's encoding sentence and H1's hold sentence. No core slot
(preservation, camera, hold-timing, shot, shared negatives) was changed, so the
core carried to the other four probes is the current wording, unchanged.

All scores use the decoder as fixed on 2026-11-09 (see "Decoder fix" below). All 16
clips were rescored with it, and no clips were regenerated.

**Decoder fix, 2026-11-09.** On the goal frame the decoder dropped the thin wine
bottle whenever it sat next to the large cabinet, even though its paint was flat,
saturated and above the size minimum. The cause:

- A colour was kept only if its hue peak held 0.5 % of **all** painted pixels
  (`hue_min_frac`), so one large object raised the bar for every other.
- Next to the cabinet's ~68,000 painted pixels, the bottle's ~1,700 sat right at
  that floor and flickered out over the last second.

The floor is now absolute: a colour needs at least `min_area_frac` of the frame
(0.1 %, 922 px) within its hue window, the same minimum a mask must meet
(`modules/instance_seg/extractor.py`). The fix was made for a measurement reason,
not to favour an arm. It applies to every clip, and both decoder checks are
unchanged:

- untouched frames decoding to any instance: 0/20;
- ground-truth masks painted in distinct colours: mean IoU 0.990, minimum 0.938
  (before: 0.990 / 0.939).

Only goal-frame scores moved: E1 0.490 → 0.722, H1 0.494 → 0.814. Base and T1 are
unchanged.

**E1** (4 clips, seed 0, all valid on the first attempt):

| frame | base | E1 | Δ | objects matched (base → E1) | E1 best frame |
|---|---|---|---|---|---|
| spatial | 0.587 | 0.567 | −0.020 | 1/2 → 1/2 | 0.568 |
| object | 0.465 | 0.934 | +0.469 | 1/2 → 2/2 | 0.952 |
| goal | 0.828 | 0.722 | −0.106 | 2/2 → 1/2 | 0.863 |
| libero_10 | 0.043 | 0.629 | +0.586 | 0/4 → 3/4 | 0.720 |
| **mean** | **0.481** | **0.713** | **+0.232** | 4/10 → 7/10 | 0.776 |

What E1 did:

- **It fixed the failure it targeted.** Paint is now flat and saturated on every
  frame: median saturation is 0.82–0.94 (base: 0.53–0.77 on the recoloured
  objects). Every object was painted in a distinct colour.
- **Spatial is unchanged in effect.** The bowl's mask now covers the whole
  cabinet top, a worse spill than the base's (T).
- **Goal lost at the end.** At frame 90 it scored 0.863. The thin wine bottle's
  paint sits at the decoder's 0.80 saturation threshold (median 0.82) and fades
  slightly in the last second. It is still decoded on the last frame, but its mask
  is too small to match (H).
- **libero_10:** all 4 objects were decoded (base: 2), and 3 matched. The mugs were
  redrawn as plain mugs (P).

E1 was adopted. Its losses came from slots it did not change: the spill on spatial
(T) and the bottle fading at the end on goal (H), which became the next targets.
The base prompt (`prompt.yaml`) stays unchanged, as the reference for Stage 3.
Later arms are written from the current best's text, in `arms/`.

**T1 and H1** (both built from E1, one slot each, confirmed by word diff; 8 clips,
seed 0, all valid on the first attempt):

| frame | E1 | T1 | H1 | objects matched (E1 / T1 / H1) |
|---|---|---|---|---|
| spatial | 0.567 | 0.321 | 0.931 | 1/2 · 1/2 · 2/2 |
| object | 0.934 | 0.471 | 0.955 | 2/2 · 1/2 · 2/2 |
| goal | 0.722 | 0.422 | 0.814 | 1/2 · 1/2 · 2/2 |
| libero_10 | 0.629 | 0.655 | 0.750 | 3/4 · 3/4 · 4/4 |
| **mean** | **0.713** | **0.467** | **0.863** | 7/10 · 6/10 · 10/10 |

- **T1 made the spill worse.** Naming "the table, shelf or other surface" seems to
  have drawn the model's attention to the surface.
  - On spatial, the whole cabinet top was painted, and so was a second bowl the
    prompt did not name.
  - On object, the basket went back to shaded recolouring.

  A negative instruction that names the thing to avoid is a known risk with
  generative models. The next T arm should say where the mask goes, not where it
  doesn't.
- **H1 is the best prompt so far,** with all 10 objects matched.
  - On goal, the bottle holds its colour to the end and matches.
  - On spatial it had no spill. H1's edit says nothing about extent, though, so
    that gain is not attributable to the H slot. With one seed, any change to the
    prompt produces a different clip, so per-frame changes outside the edited slot
    are partly chance. Read the mean, not single frames.

**Diagnosis of H1** (last frame, per object, against the simulator masks):

| frame | object | IoU | painted (changed) | painted at sat ≥ 0.80 | decoded | spill |
|---|---|---|---|---|---|---|
| spatial | bowl | 0.953 | 0.99 | 0.96 | 0.97 | 0.01× |
| spatial | plate | 0.910 | 0.99 | 0.91 | 0.91 | 0 |
| object | orange juice | 0.942 | 1.00 | 0.94 | 0.94 | 0 |
| object | basket | 0.967 | 1.00 | 0.98 | 0.98 | 0.01× |
| goal | wine bottle | 0.641 | 0.96 | 0.68 | 0.64 | 0 |
| goal | cabinet | 0.988 | 1.00 | 0.99 | 0.99 | 0 |
| libero_10 | left plate | 0.506 | 0.94 | 0.54 | 0.51 | 0 |
| libero_10 | right plate | 0.879 | 0.97 | 0.88 | 0.88 | 0 |
| libero_10 | yellow-and-white mug | 0.706 | 0.93 | 0.72 | 0.71 | 0 |
| libero_10 | white mug | 0.910 | 0.97 | 0.92 | 0.91 | 0 |

"Painted" and "decoded" are fractions of the object's true pixels. "Spill" is
painted area outside the object, as a multiple of its size.

- **Every remaining loss is under-coverage, and the cause is encoding (E).**
  Every object is painted almost entirely (93–100 % of its pixels changed) and
  nothing spills.
- **The loss is paint below the decoder's 0.80 saturation:** the pink left plate
  (only 54 % of it saturated, median 0.80), the bottle (68 %) and the shaded mug
  (72 %; median brightness 0.52, on its shadowed side and handle).
- **The decoder loses almost nothing beyond that:** "decoded" ≈ "painted at sat ≥
  0.80" on every object.
- **No losses left for T, H, R or P.** Within the score there is no spill, no
  missing or wrong object, and no fading. The redrawn mug shapes do not cost IoU
  here.
- **Next arm, E4:** the same slot again, with one sentence added: "Every pixel of a
  mask is exactly the same color and brightness, even where the object is in
  shadow." Lowering the 0.80 threshold instead would fit the measurement to these
  clips, so it is not done.

**E4** (from H1, one sentence added to E; 4 clips, seed 0). *Correction
(2026-11-12):* this section first said all four clips were valid. In fact the spatial
clip failed the scene gate (changed 0.177). It was regenerated on seed 1000 (valid,
0.84), so **E4's mean is 0.682**, not 0.604. The table below shows the first scoring:

| frame | H1 | E4 | objects matched (H1 → E4) | what happened |
|---|---|---|---|---|
| spatial | 0.931 | 0.530 | 2/2 → 1/2 | bowl mask spilled over the whole cabinet top again (3.4× the bowl's area) |
| object | 0.955 | 0.470 | 2/2 → 1/2 | basket painted magenta but with its weave shading kept: 2 % of it saturated enough to decode |
| goal | 0.814 | 0.579 | 2/2 → 1/2 | the bottle was replaced by a green **face mask**, a new object |
| libero_10 | 0.750 | 0.835 | 4/4 → 4/4 | better: both mugs flat and fully saturated (0.94, 0.91) |
| **mean** | **0.863** | **0.604** | 10/10 → 7/10 | |

- **E4 is not adopted; H1 stays the current best.** E4 fixed the frame it was
  aimed at (libero_10, where the shaded mug went from 0.706 to 0.936) and broke
  three others.
- **Mentioning what to avoid primed it, again.** E4 says "even where the object
  is in shadow", and the basket came back with its shading. T1 named "the table,
  shelf or other surface", and the surface was painted. Two of two "avoid X" sentences
  produced X. Future arms should state only the wanted result.
- **The word "mask" is ambiguous.** E4 repeats "mask" once more ("Every pixel of a
  mask"), and on goal the model drew a literal face mask in place of the bottle.
  Every prompt so far calls the paint a "mask". A wording arm that replaces it
  (for example "a flat layer of paint") targets this directly.
- **Single-seed swings are large.** The spatial spill disappeared in H1 and
  returned in E4, though neither edit is about extent. Per-frame differences of
  this size between neighbouring arms are partly chance. That matters more now
  that the remaining gains are small.

**E5** (from H1, "mask" → "paint" in 5 places, nothing else changed; 4 clips, seed
0). *Correction (2026-11-12):* the goal clip failed the scene gate (changed 0.193), not
"all valid" as first written. It was regenerated on seed 1000 (valid, 0.81), so **E5's
mean is 0.581**, not 0.441. The table below shows the first scoring:

| frame | H1 | E5 | objects matched (H1 → E5) | what happened |
|---|---|---|---|---|
| spatial | 0.931 | 0.319 | 2/2 → 1/2 | plate painted a pale, glossy cyan: none of it saturated enough to decode |
| object | 0.955 | 0.843 | 2/2 → 2/2 | basket recoloured with its weave still showing (71 % decodable) |
| goal | 0.814 | 0.249 | 2/2 → 0/2 | cabinet repainted a shaded magenta (0 % decodable); bottle recoloured as green glass |
| libero_10 | 0.750 | 0.353 | 4/4 → 2/4 | mugs and a plate recoloured with shading and texture; one plate only partly painted |
| **mean** | **0.863** | **0.441** | 10/10 → 5/10 | |

- **E5 is not adopted; H1 stays the current best.**
- **"Paint" is read as physical paint.** The objects were recoloured as if painted
  for real, keeping their lighting, gloss, shading and texture: the failure E1 fixed.
- **"Mask" carries the meaning the decoder needs:** a flat graphic overlay. The
  face mask in E4 was a cost of that word, but the alternative costs far more.
  "Mask" stays.

Diagnosis of the base (Stage 0, 10 objects on 4 frames), counted by object:

| failure | slot | objects |
|---|---|---|
| shaded or textured recolouring, below the decoder's saturation | E | 6 |
| mask spills onto the support surface | T | 1 |
| mask changes after appearing, or appears late | H | 2 |
| object redrawn | P | 1 |
| target not marked or wrong object | T | 0 |

## Bridging path: Wiedemer et al. (2025) → base

**Origin of the base prompt.** It was adapted from the Veo segmentation prompt of
Wiedemer et al. (2025, *Video models are zero-shot learners and reasoners*, App.
B.2), rewritten in one pass for this setting.

**What this path does.** It measures, after the fact, what each of those design
changes contributes. Starting from the published prompt, the changes are applied
one at a time:

| arm | change from the previous arm | slot |
|---|---|---|
| paper | the published Veo prompt, verbatim, green background | none (reference) |
| F1 | bullets and headings → prose, same sentences | form |
| T2 | "each distinct entity" → the named objects | T |
| R1 | "the background fades to green" → everything else stays untouched | R |
| E6 | the paper's description of the answer → our opaque-mask and outline sentences | E |
| H2 | our hold and timing sentences | H |
| P1 | "starts from the provided, unaltered photograph" → our preservation paragraph | P |
| C1 | the paper's static-scene and camera sentences → our camera paragraph and shot sentence | C |
| N1 | our negatives; identical to the base prompt (same request hash) | N |

Each arm file (`modules/instance_seg/arms/`) carries its word diff against the
previous arm. Arms paper–C1 send no negative. N1 sends exactly the base's request,
and on Omni the same request and seed give the same clip, so the base's seed-0
clips stand for N1 and it was not regenerated.

**Results** (4 tune frames, seed 0, last-frame mean IoU, our decoder and metric):

| arm | spatial | object | goal | libero_10 | mean | Δ previous | objects matched | valid |
|---|---|---|---|---|---|---|---|---|
| paper | 0.044 | 0.087 | 0.204 | 0.163 | **0.124** | — | 4/10 | 1/4* |
| F1 | 0.032 | 0.216 | 0.179 | 0.000 | **0.107** | −0.018 | 2/10 | 0/4* |
| T2 | 0.246 | 0.158 | 0.329 | 0.044 | **0.194** | +0.088 | 3/10 | 3/4* |
| R1 | 0.299 | 0.143 | 0.000 | 0.288 | **0.183** | −0.012 | 3/10 | 4/4 |
| E6 | 0.088 | 0.457 | 0.662 | 0.141 | **0.337** | +0.154 | 2/10 | 4/4 |
| H2 | 0.164 | 0.000 | 0.544 | 0.198 | **0.227** | −0.110 | 2/10 | 4/4 |
| P1 | 0.534 | 0.898 | 0.040 | 0.555 | **0.507** | +0.280 | 6/10 | 4/4 |
| C1 | 0.531 | 0.470 | 0.467 | 0.088 | **0.389** | −0.118 | 3/10 | 4/4 |
| N1 = base | 0.587 | 0.465 | 0.828 | 0.043 | **0.481** | +0.092 | 4/10 | 4/4 |

\* **Green-background arms.** paper, F1 and T2 ask for the background to turn
green. The validity gate rejects clips whose scene changes outside the marked
objects, so it rejects these arms' clips by design. For these three arms the gate
is waived and all 4 clips are scored. For every later arm the background is meant
to stay, and the gate applies as usual.

**Regenerations and failed jobs:**

- **Invalid clips, regenerated on a new seed (§6.1):**
  - H2 spatial: scene changed on seed 0, marked nothing on seed 1000, valid on
    seed 2000;
  - H2 libero_10: marked nothing on seed 0, valid on seed 1000;
  - C1 spatial: scene changed on seed 0, valid on seed 1000.
- **Server failures:** 5 jobs failed on Google's side (`DEADLINE_EXCEEDED`, not
  charged) and one returned a transient HTTP 403. These are not model outputs, so
  they were resent with the same request and seed.
- **Clips:** 32, plus 4 regenerations.

**What the path shows:**

- **The published prompt does poorly in this setting:** 0.124 against the base's
  0.481. Much of that is by construction:
  - it asks for every entity, and our metric charges for every extra mask (28 and 22
    masks against 10 targets for paper and F1);
  - its green background is only partly tolerated by our decoder.

  This compares prompts within our pipeline, not with Wiedemer et al.'s reported
  0.74, which is best frame, best of 10 samples, on LVIS, with Veo 3 and their
  matching.
- **Largest steps up:** P1, our preservation paragraph (+0.280), and E6, our
  opaque-mask and outline wording (+0.154). Naming the targets (T2, +0.088) helps
  too.
- **Steps down:** H2 (hold and timing, −0.110) and C1 (camera and shot, −0.118).
- **Read single steps with care.** Each step is 4 frames on one seed. The held-out
  run (Stage 3) showed that differences of ±0.1 to ±0.5 per frame arise between
  prompts by chance, so single steps of about 0.1 are within that range. The
  robust reading is the whole path: from 0.124 to about 0.4–0.5 over the
  adaptation, with the gain concentrated in the encoding and preservation changes.
- **The path is cumulative,** so each Δ is the effect of that change given all the
  earlier ones, not its effect alone.

## 3. Fixes before spending on clips (status 2026-11-09)

| # | Issue | Decision | Status |
|---|---|---|---|
| 1 | The prompt asks for "an opaque colored mask"; the decoder needs saturation ≥ 0.80 and a different hue per object | Run as an ablation arm, not an edit: **E1** (`modules/instance_seg/arms/E1.yaml`) changes only that sentence | **done**: E1 run and adopted (0.481 → 0.713) |
| 2 | The decoder had no "changed from the input" check | `color_instances` gained `min_change` (40, as the grasp decoder): a pixel counts as paint only if it also differs from the conditioning frame | **done.** Untouched LIBERO frames decoding to ≥ 1 instance: 10/20 → 0/20. Ground-truth masks painted in distinct colors: mean IoU 0.82 → 0.99 (min 0.94) |
| 3 | Failed samples drop out of the mean | Not needed: clips are generated until valid (§6.1) | dropped |
| 4 | Unusable ground truth | LIBERO: exact simulator masks. RoboTrustBench: the set is now Normal and constrained tasks only (`configs/pilot_ids.txt`) | **done**, one sample still open (below) |
| 5 | Clips cached by probe and sample only | Every arm and repeat has its own folder: `clips/<set>/<probe>/<arm>/r<repeat>/`; reports and manifests are suffixed the same way (`wmcore/arms.py`, `--arm`, `--repeat`) | **done** |
| 6 | No seed sent to the generator | Sent: `seed = seed_base + repeat`, the same for every arm. See below: the generator is now Gemini Omni 1.2 Flash | **done**, unverified until one paid clip |
| 7 | Readout docs drifted | Every probe reports the **last frame** (the readout, `last_frame`) and the **best frame** (the frame the ground truth scores highest, an upper bound). Depth moved from `most_static` to `last_frame` | **done** |

**Generator: Gemini Omni 1.2 Flash.** `backends/omni12` calls
`gemini-omni-1.2-flash` through the Gemini API's Interactions API (`GEMINI_API_KEY`;
the image goes inline, where Vertex's Omni examples need a Cloud Storage upload).
What follows from the docs (checked 2026-11-09):

- **Seed.** `generation_config.seed`, an integer, "for reproducibility". Unlike
  Veo's docs, Omni's give no range and do not promise the same video for the same
  seed. Two same-seed clips on one tune frame measure how close they come; Stage 0
  then measures the spread between seeds.
- **First frame.** `video_config.task: image_to_video` makes the conditioning
  frame the first frame. The backend always sets it.
- **Negative prompt.** Omni has none. The probe's `negative:` text is appended
  to the prompt after one fixed sentence, "The clip must not contain any of the
  following:" (`NEGATIVE_LEAD`, `wmgen/backends.py`), the same for every arm. This
  makes the N slot part of the prompt text, so a with/without-negative arm now
  also changes prompt length.
- **Audio** is always generated and is ignored. **Duration** is 3–15 s; the probe
  asks for 6 s.
- **Model drift.** `gemini-omni-1.1-flash` is routed to 1.2 on 2026-12-03, and the
  docs don't say whether 1.2 is preview or GA. Every clip's sidecar records the
  model id; check it hasn't changed between stages.

The previous generator, Veo 3.1 (the Gemini API preview was shut down on
2026-10-22; `backends/veo31_vertex` is the Vertex GA version, kept but unused), is
not mixed in: every existing clip was made with Veo, so Stage 0 starts from new
Omni clips.

**Backend verified 2026-11-09** with one live clip (base, r0, seed 0, on
`libero_spatial__ep1520_f030`): 6 s, 1280×720, 24 fps, about 42 s to generate. The
first frame matches the conditioning frame to within compression noise (mean
absolute difference 2.0/255), and the camera held still. Both objects were painted
and held: the plate (magenta) matched; the bowl's cyan mask spilled onto the cabinet
top and did not match. Last frame: mean IoU 0.587, 1 of 2 objects matched (best
frame: 0.588). Only the status string on failure is still unseen. The model also
writes a short "thought" step before the video, a likely source of variation
between repeats.

**Same-seed test (2026-11-09).** A second clip with the same frame, prompt and seed 0
came back practically identical: different files, but a mean frame difference of
0.43–0.61/255 (encoder noise; no pixel off by more than 37/255), the painted regions
overlapping at IoU 0.99, and the same score (mean IoU 0.587, 1 of 2 matched). The
"thought" step does not add randomness the seed doesn't control. Consequences:

- Repeats must use different seeds (they do: `seed_base + repeat`). Tuning
  uses one seed (Stage 0), so seed noise is not measured there.
- Shared seeds across arms work as intended: two arms on the same seed differ
  only by their prompt.
- **Regenerating an invalid clip with the same seed returns the same invalid
  clip.** The regenerate-until-valid loop (§6.1) must change the seed on each
  retry, by a fixed rule recorded in the sidecar (for example
  `seed_base + repeat + 1000 × attempt`), so a retry never collides with
  another repeat's seed.

**RoboTrustBench set (fix 4).** The 9 counterfactual, adversarial and
underspecified samples were replaced: 15 A_Normal and 5 constrained samples, one
per non-vague B subcategory, drawn with a fixed seed (the file's header says how).
SAM 3 produced ground truth for all 9 new samples, but it misses a named object on
4 (rubber bands, crumpled napkin, rail, milk carton), splits the cabinet into 3 on
one and returns two towels on another; check these by eye before Stage 3. The kept "black hoodie" sample
(`IPRL+5085c3ce+2023-10-21-17h-45m-44s__A_Normal`), which had no SAM 3 ground
truth, was replaced by the next seeded A_Normal draw with one clean instance per
named object (5 draws skipped, listed in the file; the result is "pick up the black
mug from the desk"). The 10 new samples have no clips yet.

## 4. Measurement checks, no generation needed

Run these once the fixes are in, and stop if one fails.

| check | clip | must score | status (LIBERO, 20 frames) |
|---|---|---|---|
| Decoder floor | input frame held still | `mean_iou` of 0, nothing decoded | **pass** on single frames: 0/20 decode anything |
| Decoder ceiling | ground-truth masks painted in distinct colors | `mean_iou` above 0.9 | **pass** on single frames: 0.99 mean, 0.94 min |
| Same-color case | ground-truth masks all one color | record the score; shows what touching objects cost | not run |
| Compression | ceiling clip re-encoded at generator-like quality | within 0.05 of the ceiling | not run |

The floor and ceiling were run on still frames through the decoder and metric; the
clip versions (through the readout and gates) are not built yet.

## 5. Data split

Fixed 2026-11-09, before any ablation clip, with `random.Random(0)`:

- **Tune, 4 frames** (`configs/libero_tune_ids.txt`), one per suite:
  `libero_spatial__ep1520_f030`, `libero_object__ep1227_f027`,
  `libero_goal__ep0546_f011`, `libero_10__ep0231_f017` (2, 2, 2 and 4 objects).
  Every prompt choice is made here. `configs/gen.libero.yaml` uses this file.
- **Test, 16 frames** (`configs/libero_test_ids.txt`): used once, in Stage 3.
- **Transfer:** the 20 RoboTrustBench samples, Stage 3 only.

Why 4 and not 2. Prompt-engineering work tunes on a small development set and
reports on a larger held-out set that no prompt choice has seen; with few items the
usual split puts about a fifth to a third into development. Here the frame is the
unit (repeats of one frame are not independent evidence), so the tune set has to
span the variation the test set holds: the four suites differ in scene, camera
height and object count. Two frames cannot cover four suites, and a prompt that
wins on one scene type would be chosen blind to the other two. Four frames, one per
suite, are the smallest set that covers them, and leave 16 for the test, which is
what the statistics need (below).

What 4 frames can and cannot do. With 4 paired frames no significance test can
reach p < 0.05 (the smallest two-sided Wilcoxon p for n = 4 is 0.125). So on the
tune set a prompt is chosen by its mean gain, read together with the per-frame
results (§7), never by a p-value, and the claim that it is better is tested once, on the 16 test frames.

## 6. Stages

Clip counts are per arm × frame × repeat, on the 4 tune frames unless stated.

### 6.1 Valid clips only

A clip a gate rejects (scene changed, nothing marked, never settles) is
regenerated with the next seed until it passes, up to 3 attempts per frame and
repeat. Not dropping invalid clips silently has a price: an arm that needs more
attempts is worse, so every arm reports its **attempts per valid clip** and its
first-attempt invalid rate beside its score. A frame still invalid after 3 attempts
is reported as a failure of that arm. (The regeneration loop is not automated yet:
generate, evaluate, then regenerate the invalid ones with the next seed.)

### Stage 0: base prompt (4 clips) — done 2026-11-09
Base prompt, 4 tune frames, **one repeat, seed 0**. Decided 2026-11-09: seed noise
is ignored during tuning. Every arm uses the same seed 0, so arms are paired by frame
and seed and differ only by their prompt (on Omni the same seed and prompt give the
same clip). The cost: a difference between arms mixes the prompt's effect with how
that prompt happens to interact with seed 0, so no tuning result is a claim; Stage 3
is.

| frame | objects | last-frame mean IoU | matched | best frame |
|---|---|---|---|---|
| libero_spatial__ep1520_f030 | bowl, plate | 0.587 | 1/2 | 0.588 |
| libero_object__ep1227_f027 | orange juice, basket | 0.465 | 1/2 | 0.470 |
| libero_goal__ep0546_f011 | wine bottle, cabinet | 0.828 | 2/2 | 0.844 |
| libero_10__ep0231_f017 | 2 mugs, 2 plates | 0.043 | 0/4 | 0.216 |
| **mean** | | **0.481** | 4/10 | 0.530 |

All 4 clips were valid (first frame matches, camera still) and needed one attempt.
By eye, the model painted **every** target object in all 4 clips, each in a
different colour (cyan, magenta, purple, yellow, green), and touched no
non-target object. The losses come from how it paints:

1. **Recolouring instead of flat paint (the main loss).** On 6 of the 10 objects
   the model re-rendered the object in a new colour and kept its shading and
   texture (the basket, both mugs, both plates in libero_10, the cabinet). That
   paint has a median saturation of 0.53–0.77, under the decoder's 0.80, so those
   objects decode partly or not at all. Where the paint was flat (the bowl,
   plate and juice), saturation was 0.87–0.98 and the objects matched or nearly so.
2. **Spill onto the support surface.** The bowl's mask spread over the cabinet top
   around it.
3. **Masks that change after appearing.** The cabinet's mask started as an ellipse
   on its top and later covered the whole cabinet; the last plate in libero_10
   appeared after the requested settle time.
4. **Objects redrawn.** The yellow-and-white mug became a plain yellow mug.

Failure 1 is exactly what E1 asks for ("one flat, fully saturated color"), so E1
stays the first Stage 1 arm. The decoder's 0.80 threshold is not lowered to fit
these clips: that would tune the measurement to the tune set.

### Stage 1: encoding (4 clips per arm)
Encoding comes first because every other slot is unreadable if the paint can't be
decoded. Each arm: the 4 tune frames, seed 0, as in Stage 0, so arm and base are
paired by frame and seed.

| arm | encoding text | decoder | status |
|---|---|---|---|
| base (E0) | "an opaque colored mask" | current | run: 0.481 |
| E1 | "...one flat, fully saturated color... a different color for each object, no mask white, black or grey" | current | run: 0.713, adopted |
| E2 | fixed palette by object order: "{object 1} is pure magenta, {object 2} is pure cyan..." | hue pinned per object | needs a per-object prompt slot and a pinned-hue decoder |
| E3 | E1, plus "everything else turns solid black" | brightness threshold, no saturation test | needs a decoder mode |

E2 lets the decoder tell which object each mask is meant to be, so it can score
wrong-object paint. E3 gives up scene preservation, so `scene_preservation` doesn't
apply; report it as a separate track if it wins.

### Stage 2: other slots — not run
The planned half-fraction factorial on the core slots (preservation, timing, settle
time, target format) was not run. Tuning stopped at H1 (see the results log), so
the core slots keep their base wording for all five probes.

### Stage 3: confirmation (32 clips)
Base against H1, once, on frames neither was tuned on:

- **LIBERO test set:** the 16 frames in `configs/libero_test_ids.txt`, base and H1,
  one clip each (32 clips).
- **RoboTrustBench transfer: not run** (decided 2026-11-09). The confirmation is on
  LIBERO only, where the ground truth is exact.
- **Seed 1, not 0** (`--repeat 1`). H1 was chosen on seed 0, so confirming on a
  fresh seed keeps the result from depending on that one seed. Base and H1 share
  the seed, so each frame is still a paired comparison.
- No prompt changes after this stage.

**Clips:** 25 for tuning and the same-seed test, 32 for Stage 3, and 4
regenerations: 61 in all.

#### Stage 3 result (2026-11-09)

All 16 frame pairs are valid.

- **Regenerations:** 2 of the first 32 clips were invalid. Base on
  `libero_spatial__ep1388_f018` marked nothing, and H1 on `libero_goal__ep0674_f012`
  changed the scene 0.151 > 0.15 outside the marked region. Both arms on those two
  frames were regenerated on a new shared seed (1001), so each frame stays a paired
  comparison. All 4 regenerations were valid.
- **Attempts per valid clip:** 17/16 for each arm.

**Primary result (last-frame mean IoU, 16 frames):**

| | base | H1 | H1 − base |
|---|---|---|---|
| mean | 0.511 | 0.552 | **+0.041**, 95% bootstrap CI [−0.145, +0.245] |
| median | 0.484 | 0.484 | |
| frames better / worse | | | 7 / 9 |
| Wilcoxon signed-rank (two-sided) | | | W = 66, **p = 0.94** |
| paired effect size d_z | | | 0.10 |

**H1 is not shown to be better than the base on held-out frames.** The effect is
far below the roughly 0.75 SD this test can detect, so the result is
inconclusive, not evidence of no effect.

The sensitivity check agrees: using only the 14 frames whose first attempts were
both valid, the difference is +0.021 [−0.153, +0.208], p = 0.81.

**Secondary metrics** (none differs reliably; all 95% CIs include 0):

| metric | base | H1 |
|---|---|---|
| best-frame mean IoU | 0.552 | 0.591 |
| matched IoU | 0.680 | 0.732 |
| recall / precision | 0.583 / 0.667 | 0.635 / 0.677 |
| foreground IoU | 0.557 | 0.523 |
| count error | 0.312 | 0.250 |
| objects matched | 19/33 | 21/33 |
| last-to-best gap | 0.041 | 0.039 |

**Per suite** (4 frames each; descriptive only): spatial 0.366 → 0.744, goal 0.474
→ 0.572, object 0.602 → 0.420, libero_10 0.602 → 0.472. Per-frame differences
range from −0.505 to +0.857.

**What changed and what did not.** Per object, against the simulator masks, on
the last frame:

| | base | H1 |
|---|---|---|
| objects mostly unpainted (< 50 % of pixels changed) | 8/33 | 1/33 |
| mean fraction of each object painted | 0.76 | 0.92 |
| mean fraction saturated enough to decode | 0.58 | 0.69 |
| decodable / painted | 0.77 | 0.75 |
| saturated paint outside the targets (× target area) | 0.40 | 0.11 |

- **What transferred:** H1 paints more of the targets and puts less paint
  elsewhere.
- **What did not:** the encoding fix, the reason E1 was adopted. The share of
  paint saturated enough to decode is unchanged (0.77 vs 0.75). On the tune frames
  E1 had raised it sharply.
- **Net:** more paint, but no more of it readable, so IoU barely moved.

**Why the tune result did not hold.** H1 was picked as the best of six prompts on
4 frames with one seed. Per-frame differences between prompts on the test set are
as large as ±0.5, so a prompt that wins on 4 frames and one seed can win largely
by chance. Choosing the best of several such results inflates the winner's tune
score; this is the selection effect a held-out set exists to catch. The tune score
of 0.863 is not an estimate of H1's real performance; 0.552 is.

**For the other four probes:**

- Tune on more frames, more than one seed, or both, before choosing.
- Prefer edits whose mechanism can be checked on held-out frames (as above).
- Do not rely on mean IoU over 4 frames alone.

## 7. Analysis, fixed in advance

**Primary metric:** `mean_iou` on the last frame, valid clips only (§6.1). The
best-frame score is chosen by the ground truth, so it is reported as an upper bound
and never used to choose a prompt.

**Secondary metrics:** `foreground_iou` (did paint land on objects?), `matched_iou`
(outline quality), recall, precision, count error, attempts per valid clip, first-attempt invalid rate by gate, and the
gap from last frame to best frame (does the answer hold until the end?).

**Unit of analysis:** the frame. Average repeats within a frame first, so that repeats
don't count as extra samples.

**Comparison:** on the tune set, the paired difference per frame (one seed, so
there is no noise floor; 4 frames cannot support a test, §5). In Stage 3, H1 minus base per
frame: the mean, a bootstrap 95% CI over frames, and a two-sided Wilcoxon
signed-rank test, on the 16 LIBERO test frames. This is the only comparison, so no
multiple-comparison correction is needed.

**Choosing on the tune set:** an arm is adopted if it raises the mean last-frame
IoU over the current best, reported with its per-frame results and diagnosis. Only
Stage 3 claims that a prompt is better.

**What 16 test frames can show:** with 16 paired frames, only fairly large effects
are detectable (roughly 0.75 standard deviations of the paired difference at 80%
power). Anything smaller should be reported as inconclusive, not as no effect.

## 8. What to report

### Result

The segmentation probe's prompt was tuned on 4 LIBERO frames, with one seed and 6
prompts. The selected prompt, **H1**, was then compared once with the original prompt
on 16 held-out frames, using a new seed.

- **H1 is not shown to be better.**
- Last-frame mean IoU: 0.552 for H1 against 0.511 for the original.
- Difference: +0.041, 95% bootstrap CI [−0.145, +0.245].
- Wilcoxon signed-rank test: p = 0.94, n = 16, d_z = 0.10.

H1's tune-set score (0.863) reflects selection on 4 frames and one seed, and is not
an estimate of its performance.

### SAM 3 on the same frames (reference)

**Setup:** SAM 3 was run on the same LIBERO conditioning frames, with the same
settings that make the RoboTrustBench pseudo ground truth (`probe.yaml`
`ground_truth`) and each frame's object names (`gt_prompt`). It was scored against
the same simulator masks with the same metric. SAM 3 segments the photograph
directly, while the world model has to paint its answer into a video, so SAM 3 is
a reference for what a dedicated segmenter achieves on these frames, not a
competitor on equal terms.

| 16 test frames | SAM 3 | base (seed 1) | H1 (seed 1) |
|---|---|---|---|
| last-frame mean IoU | **0.559** | 0.511 | 0.552 |
| objects matched (of 33) | **25** | 19 | 21 |
| IoU of matched objects | **0.935** | 0.680 | 0.732 |
| recall / precision | 0.760 / 0.833 | 0.583 / 0.667 | 0.635 / 0.677 |
| count error | 1.125 | 0.312 | **0.250** |

**Paired differences, test frames:**

- SAM 3 − H1: +0.007, 95% CI [−0.068, +0.082], p = 0.94.
- SAM 3 − base: +0.048, 95% CI [−0.113, +0.212], p = 0.46.

On all 20 frames the means are SAM 3 0.615, H1 0.614 and base 0.505.

**What this shows:**

- **On mean IoU, H1 is level with SAM 3.** Neither difference from SAM 3 is
  distinguishable from zero.
- **The two fail in opposite ways.**
  - SAM 3's masks are near-exact when it finds the object (0.935), but it misses
    or over-counts.
  - The world model rarely miscounts, but its masks are rougher.
- **SAM 3 misses LIBERO's packaged objects:** the microwave, caddy, wine rack,
  butter, milk and cream cheese (twice) were not grounded.
- **SAM 3 over-counts in two ways:**
  - "Black bowl" matches both bowls in the spatial scenes. The world model is
    given "the black bowl on the wooden cabinet"; SAM 3 gets only the noun.
  - It splits a cabinet into 5 parts.
- **For RoboTrustBench, where SAM 3 is the ground truth:** on LIBERO it outlines
  the objects it finds almost exactly, but finds about three quarters of them and
  adds extra masks. RoboTrustBench segmentation scores are therefore partly
  limited by the pseudo ground truth, not only by the world model.

The masks are in the session scratchpad. They can be regenerated by running
`Sam3Instances` with `probe.yaml`'s params on `data/libero/frames`.

### Contents

| item | where |
|---|---|
| Prompt text of every arm | `modules/instance_seg/prompt.yaml` (base); `modules/instance_seg/arms/{E1,T1,H1,E4,E5}.yaml`, each with its word diff in the header |
| The split | 4 tune frames (`configs/libero_tune_ids.txt`), 16 test frames (`configs/libero_test_ids.txt`), §5 |
| Tuning results | Results log (§2), one row per arm, with per-frame tables and diagnoses |
| Held-out results, CIs, invalid rate, clip counts | Stage 3 result (§6) |
| Final prompt, by slot | `arms/H1.yaml`; slots per paragraph in docs/prompt_framework.md |

**Example clips** (H1, held-out, chosen by last-frame mean IoU, not by eye), in
`clips/libero/instance_seg/H1/r1/`:

| | frame | mean IoU |
|---|---|---|
| best | `libero_spatial__ep1285_f041` | 0.866 |
| median | `libero_goal__ep0798_f029` / `libero_object__ep1191_f012` | 0.482 / 0.486 |
| worst | `libero_object__ep1183_f023` | 0.259 |

**Failure breakdown, final prompt (H1), held-out:**

- **Clips:** of 17 attempts, 16 were valid. The one invalid clip changed the scene
  outside the marked region (0.151 > 0.15). No clip marked nothing.
- **Objects:** of the 33 target objects on the 16 valid clips:

| outcome | objects |
|---|---|
| matched (IoU ≥ 0.5) | 21 |
| painted, but less than half saturated enough to decode | 8 |
| decoded, but merged with another, wrongly split, or a poor outline | 3 |
| not painted (< 50 % of its pixels changed) | 1 |

- **Answer gone by the readout:** none to speak of. The last-frame score is 0.039
  below the best frame's.

The dominant remaining failure is the encoding: paint present but not flat and
saturated enough to decode. That is the failure E1 targeted and fixed on the tune
frames but not on the held-out ones.
