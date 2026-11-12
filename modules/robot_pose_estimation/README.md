# robot_pose_estimation

Ask the world model to mark the robot's end-effector and heading with two
colored dots, then score the decoded position and orientation against the
end-effector pose the simulator recorded. LIBERO only: real video has no
recorded pose, and estimated ones were dropped.

## The two halves

| file | pipeline | says |
|---|---|---|
| [prompt.yaml](prompt.yaml) | wmgen | a red dot on the gripper and a green dot a short distance ahead along its pointing direction |
| [probe.yaml](probe.yaml) | wmeval | find the two dots, then score center distance and heading angle |

## Ground truth

`recorded_pose` projects two world points through the sample's `world_to_image`
matrix: the end-effector position, and a point `heading_distance` (8 cm) ahead of
it along the gripper's approach axis (its local +z). Both come from the replay's
`robot0_eef_pos` and `robot0_eef_quat`, which `scripts/libero_replay/render_gt.py`
writes into the meta and which match the dataset's recorded state to within
0.3 mm. See [docs/libero_ground_truth.md](../../docs/libero_ground_truth.md).

## The scores

| score | meaning |
|---|---|
| `pose_score` | `exp(-pos_err / pos_threshold) * (1 - angle_err / 90)` — the headline |
| `pos_err` | center distance, normalized by the frame diagonal |
| `pos_pck` | 1 when the center is within `pos_threshold_frac` of the diagonal |
| `angle_err` | heading difference, in degrees, folded into [0, 90] |
| `heading_score` | 1 - `angle_err` / 90 |

`pose_score` is higher-is-better, so it is a legal `best_frame.rank_by`.

## Inspect

```bash
venv/bin/python scripts/libero_replay/review_gt.py
venv/bin/python -m modules.robot_pose_estimation.scan generated.mp4 \
    --gt results/libero/robot_pose_recorded_gt/<sample_id>_gt.npz
```
