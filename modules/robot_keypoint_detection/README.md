# robot_keypoint_detection

Ask the world model to dot each robot joint, then score the dot centers against
the exact image positions of the Panda's links from a simulator replay. LIBERO
only: real video has no exact keypoints, and estimated ones were dropped.

## The two halves

| file | pipeline | says |
|---|---|---|
| [prompt.yaml](prompt.yaml) | wmgen | a small bright saturated dot on each robot joint, then hold still |
| [probe.yaml](probe.yaml) | wmeval | find the dots and score them against the simulator's keypoints |

## Ground truth

`sim_keypoints` reads the record `scripts/libero_replay/render_gt.py` writes for
each sample. The keypoints are the seven Panda links RoboPEPP and DREAM use
(link0, 2, 3, 4, 6, 7 and the hand), taken as each body frame's origin in MuJoCo
at the conditioning frame and projected with the render's own camera. A link
behind the arm is kept, since the prompt asks for every joint; a link outside the
rendered view is dropped and named in `outside_frame`. See
[docs/libero_ground_truth.md](../../docs/libero_ground_truth.md).

## The scores

| score | meaning |
|---|---|
| `pck` | fraction of ground-truth keypoints within `threshold_frac` of the frame diagonal — the headline |
| `mean_dist` | mean matched distance, normalized by the diagonal |
| `precision`, `recall`, `f1` | at the same threshold |
| `score` | mean `exp(-d / threshold)`, a smooth version |
| `n_pred`, `n_truth`, `n_matched`, `frame_diagonal` | context |

`pck` is higher-is-better, so it is a legal `best_frame.rank_by`.

## Inspect

```bash
venv/bin/python scripts/libero_replay/review_gt.py      # every ground truth, by eye
venv/bin/python -m modules.robot_keypoint_detection.scan generated.mp4 \
    --gt results/libero/robot_keypoints_sim_gt/<sample_id>_gt.npz
```
