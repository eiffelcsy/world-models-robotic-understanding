#!/bin/zsh
# Every prompt-ablation run prepared on 2026-11-10 and blocked by the Gemini
# project's monthly spending cap. Run once the cap is raised:
#
#   export GEMINI_API_KEY=...        # or: eval "$(grep -E '^[[:space:]]*export GEMINI_API_KEY=' ~/.zshrc)"
#   scripts/run_pending_ablations.sh > results/ablations/pending.log 2>&1
#
# 82 new clips, if nothing fails on the server: grasp 2 leftovers + 8, keypoints 36,
# pose 36. Existing clips are skipped, so it is safe to rerun after an interruption.
set -u
cd "$(dirname "$0")/.."

# grasp_region_seg: the E2 clip refused by the cap (libero_10), then T4 and E3 (from T2)
scripts/run_ablations.sh grasp_region_seg E2 T4 E3
# grasp_region_seg: R1 on spatial marked nothing; regenerate it on the next seed (1000).
# Spatial is the first tune frame, so --limit 1 makes only that clip.
for step in gen eval; do
  if [[ $step == gen ]]; then
    venv/bin/python scripts/run_gen.py --config configs/ablations/gen.grasp_region_seg.yaml --arm R1 --repeat 1000 --limit 1 2>&1 \
      | sed -E 's/AIza[0-9A-Za-z_-]+/<key>/g' | grep -E "^  grasp_region_seg |rror"
  else
    venv/bin/python scripts/run_eval.py --config configs/ablations/eval.grasp_region_seg.yaml --arm R1 --repeat 1000 2>&1 | grep -E "rror|summary:"
  fi
done

# robot_keypoint_detection: bridging path, then improvement arms (docs/keypoint_prompt_ablations.md)
scripts/run_ablations.sh robot_keypoint_detection paper R1 T1 E1 K1 base T2 E2 E3

# robot_pose_estimation: bridging path, then improvement arms (docs/pose_prompt_ablations.md)
scripts/run_ablations.sh robot_pose_estimation paper R1 T1 E1 K1 base T3 E2 E3
echo ALL_DONE
