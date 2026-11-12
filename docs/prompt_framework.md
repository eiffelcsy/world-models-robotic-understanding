# Prompt framework

This document describes a method for writing and tuning the five probe prompts. It
needs no code: the prompts stay plain text in `modules/<probe>/prompt.yaml`, and
an ablation arm is a copy of that text with one slot reworded.

## 1. Anatomy: what every prompt must say

Every probe prompt asks the world model one kind of question: keep this
photograph, show the answer in a form the decoder can read, then hold still. Each
part of that request is a **slot**. Every slot exists to prevent one failure the
score would punish.

| slot | says | prevents | shared? |
|---|---|---|---|
| **P** Preservation | the scene stays as it is (or, for depth, keeps its geometry) | a repainted or restyled scene | core |
| **C** Camera | the camera is locked off; one shot, no cut | an answer misaligned with the ground truth | core |
| **H** Hold | the answer is complete by `settle_s`, then nothing changes | the answer missing or changing at the readout frame | core |
| **N** Negatives | known failure modes, listed | shading, text, camera motion | core list + task list |
| **T** Target | what to mark, and how exactly the mark sits on it | wrong, missing or spilling marks | task |
| **E** Encoding | what the answer looks like, matched to the decoder | an answer the decoder can't read | task |
| **R** Rest | what must stay untouched | non-targets changed | task |

**Core** slots carry the same words in every prompt. Today the four overlay
probes (segmentation, grasp, keypoints, pose) share P, C, H and the N camera terms
word for word. Depth replaces the photograph instead of drawing on it, so its P
is a "transform" variant. Its C and H are worded differently, which is a known
inconsistency to fix when depth is tuned (§4).

**Task** slots are each probe's own. **E is a contract with the decoder**
(`probe.yaml`): any change to E must state the decoder setting it relies on
before clips are generated.

Where each slot sits in each prompt (also noted in a comment in each `prompt.yaml`):

| probe | paragraph 1 | 2 | 3 | 4 | 5+ |
|---|---|---|---|---|---|
| instance_seg | P | C | T+E | R | H |
| grasp_region_seg | P | C | T+E | R | H |
| robot_keypoint_detection | P | C | T+E | R | H |
| robot_pose_estimation | P | C | T+E, T, T | E+R | H |
| depth_perception | P (transform) | E | E | C+H | — |

## 2. Method: tuning one probe

Each step produces a number or a decision, and every result goes in the probe's
results log (§5).

1. **Base.** Run the current prompt on the 4 tune frames (seed 0, one repeat).
   Score it on the last frame.
2. **Diagnose.** Look at every clip and assign each lost object or point to the
   slot whose failure explains it (the table in §3). Count how often each failure
   occurs. The most frequent failure picks the next slot to change.
3. **Ablate one slot.** Write an arm: the base text with only that slot's words
   changed.
   - The arm's header states the slot, the exact old and new wording, and the
     failure it targets.
   - Check that nothing else changed by diffing the two texts word by word,
     e.g. `git diff --no-index --word-diff`.
   - Run the arm on the same frames and seed as the base, so the pair differs
     only in the prompt.
4. **Decide.** Adopt the arm if it raises the mean last-frame IoU (or the probe's
   primary metric) over the current best prompt; later arms are then written from
   it. Report the per-frame results and the diagnosis with every decision. With 4
   frames this is a screening step, not a significance test.
5. **Repeat** steps 2–4 until the remaining failures are ones no slot change is
   expected to fix, or the budget runs out.
5b. **Invalid clips.** A clip the gates reject is regenerated on a new seed, up to 3
   attempts (on held-out frames, both arms of the pair together). A frame still
   invalid after 3 attempts is **scored 0 as a failure of that arm**, never dropped;
   only the headline is set to 0, and the clip's other measurements are kept. The
   `paper` arms, whose backgrounds fail the scene gate by design, are exempt and keep
   their raw scores.
6. **Confirm once** on the 16 held-out test frames: base versus final, paired by
   frame, with a bootstrap CI and a Wilcoxon test. This is the only claim that a
   prompt is better.

## 3. Failure → slot

| failure seen in a clip | slot to change |
|---|---|
| paint shaded or textured, or too pale to decode | E |
| two objects in the same color | E |
| a target not marked, or the wrong object marked | T |
| mark spills past the object (onto a table or support) | T (extent) |
| a non-target object or the robot changed | R |
| scene restyled, objects redrawn | P |
| camera moves, zooms or cuts | C |
| mark appears late, fades or changes after appearing | H |

## 4. Across the five probes

- **Segmentation is the pilot.** Its tuning includes the core slots (P, C, H,
  N-core), because those are tested there first.
- **Freezing the core.** When segmentation is finished, its final core wording is
  copied word for word into the other four prompts. It is then frozen.
- **The other probes**, in order: grasp, keypoints, pose, then depth. Each runs §2
  on its own tune frames and changes only its own slots (T, E, R, N-task). A core
  slot is reopened only if a probe's diagnosis points at it, and the result is
  then reported as a deviation.
- **Final result:** five prompts that share one core and differ only in their
  task slots. Each is confirmed once on held-out frames. Segmentation and depth are
  also run on RoboTrustBench to test transfer.
- **Caveat to report:** the core is tuned on segmentation and carried over to the
  other probes, not tuned on them.

## 5. Results log

Each probe keeps one table in its ablation document. For segmentation that is
[segmentation_prompt_ablations.md](segmentation_prompt_ablations.md):

| step | arm | slot changed | targets failure | last-frame mean IoU | Δ vs current best | objects matched | decision |
|---|---|---|---|---|---|---|---|

Per-probe documents (status 2026-11-10):

| probe | document | status |
|---|---|---|
| instance_seg | [segmentation_prompt_ablations.md](segmentation_prompt_ablations.md) | finished; H1 frozen, not confirmed on held-out frames |
| depth_perception | [depth_prompt_ablations.md](depth_prompt_ablations.md) | finished; N2 not confirmed, base kept |
| grasp_region_seg | [grasp_prompt_ablations.md](grasp_prompt_ablations.md) | finished; E3 final, **confirmed** vs base on held-out frames (grasp_loc 0.456 vs 0.026, p 0.001) but level with object-only guesses |
| robot_keypoint_detection | [keypoint_prompt_ablations.md](keypoint_prompt_ablations.md) | finished; base kept (pck 0.575); robot motion not fixable by wording |
| robot_pose_estimation | [pose_prompt_ablations.md](pose_prompt_ablations.md) | finished; T3 not confirmed on held-out frames (−0.091, p 0.21, 16 frames incl. 1 failure); base kept |

Pending runs: `scripts/run_ablations.sh <probe> <arm> ...` for one probe, or
`scripts/run_pending_ablations.sh` for everything prepared. Summarise with
`scripts/ablation_table.py <probe> <headline> <arm> ...`.
