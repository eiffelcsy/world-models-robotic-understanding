# grasp_region_seg

Ask the world model to paint the graspable region of the object to pick up, then
score it against the grasp the robot actually made in the demonstration.

## The two halves

| file | pipeline | says |
|---|---|---|
| [prompt.yaml](prompt.yaml) | wmgen | mask the graspable part of `{grasp_object}` in one flat saturated color, then hold still |
| [probe.yaml](probe.yaml) | wmeval | decode the painted region and score it against the demonstrated grasp |

## The ground-truth chain

LIBERO ships no grasp labels, but every demonstration contains a grasp that worked.

```
gripper command + finger gap ──▶ the frame the fingers settle on the object
replay at that frame ──▶ the two finger pads' inner faces ──world_to_image──▶ finger points + centre
MuJoCo finger contacts after the close ──▶ the grasped object ──▶ its exact mask
```

`scripts/fetch_libero.py` finds the grasp and writes it to the sample's meta. A close
counts as held if the fingers stop at least 4 mm apart and the gripper then lifts at
least 3 cm and travels at least 5 cm. Episodes are used only if their first close
held, because a failed attempt may have moved the object. Conditioning frames come
from before that close, with the arm clear of the grasp point. An object pushed
before the close is caught because the grasp spot then looks the same at the start
and end of the episode.

`scripts/libero_replay/render_gt.py` then replays the demonstration, reads which
object the finger geometries touch in the 15 steps after the close
(`sim_grasp_object`), records where the two finger pads are at the grasp frame
(the fingers are driven separately, so an object can push them off the grip
site's centre), and records the segmentation render the mask is cut from. A
sample fails if the grasp sits more than `max_object_gap_m` from that mask.

This needs a demonstration and a replay, so the probe runs on LIBERO only. See
[docs/libero_ground_truth.md](../../docs/libero_ground_truth.md).

### What it can't tell you

A demonstration is one grasp that worked, not every grasp that would. A model that
paints a different valid grasp scores low, so `grasp_loc` is a lower bound.
There is no region to take an IoU against. Two-object tasks use the first grasp only.

## Install

```bash
venv/bin/python -m pip install -r modules/grasp_region_seg/requirements.txt
```

The ground truth needs no model.

## The scores

| score | meaning |
|---|---|
| `grasp_loc` | the headline (from 2026-11-11). `1 − E_paint / E_object`, clipped to [0, 1], where `E` is the mean distance of a set of pixels from the grasp centre beyond `tolerance_m` (2 cm). 1: all paint within 2 cm of the grasp; 0: no nearer than the object as a whole (painting the whole object); nothing painted is 0 |
| `object_excess_m` | `E_object` in metres: how much the sample can separate knowing the grasp from finding the object; describes the sample |
| `grasp_score` | `hit * focus * on_object * tightness`, the headline until 2026-11-11, kept for comparison |
| `hit` | the region reaches the grasp, within `tolerance_m` (2 cm) |
| `focus` | share of the painted pixels within `focus_m` (4 cm) of the grasp |
| `on_object` | share of the painted pixels that are on the target object |
| `coverage` | share of the target object that is painted |
| `tightness` | 1 up to `free_coverage` (half the object), falling to 0 at all of it |
| `distance_px`, `distance_m` | from the grasp to the nearest painted pixel; lower is better |
| `grasp_offset_m` | from the object's centre to the grasp; describes the sample |
| `area_pred` | painted area as a fraction of the frame |

Why the headline changed: `grasp_score` gave a guess that paints the top quarter
of the object 0.665 and a mark on the demonstrated grasp 0.748, because `on_object`
penalises a rim grasp's patch for lying partly off the object and `hit` is all or
nothing at 2 cm. The decoder floor was halved at the same time (184 → 92 px), since
a finger-wide mark on the grasp was dropped on 5 of 20 frames. Full account:
[docs/grasp_prompt_ablations.md](../../docs/grasp_prompt_ablations.md), "Metric revision".

Synthetic answers on the 20 LIBERO frames, through the decoder:

| painted region | `grasp_loc` | `grasp_score` |
|---|---|---|
| dot on the grasp, 1 cm or 2 cm radius | 1.00 | 0.83 / 0.75 |
| dot on the object point nearest the gripper | 0.53 | 0.21 |
| top quarter of the object | 0.49 | 0.67 |
| dot on the object's centroid | 0.35 | 0.15 |
| whole object | 0.00 | 0.00 |

The middle three know the object (and, for the first, where the gripper is) but not
the grasp. On LIBERO they score about 0.5, because demonstrated grasps sit near the
top of the object and near the gripper. That is the floor a model has to beat.

`grasp_loc` is a legal `best_frame.rank_by`.

## Inspect

The store (`results/libero/grasp_demonstrated_gt/`) fills itself during an eval
run. To look at every grasp on its frame:

```bash
venv/bin/python scripts/libero_replay/review_gt.py
```
