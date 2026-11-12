#!/bin/zsh
# Generate and score prompt-ablation arms for one probe on the 4 LIBERO tune frames.
#
#   scripts/run_ablations.sh <probe> <arm> [<arm> ...]
#   scripts/run_ablations.sh robot_keypoint_detection paper R1 T1 E1 K1 base T2 E2 E3
#
# Uses configs/ablations/{gen,eval}.<probe>.yaml (seed 0, one repeat). Clips that
# already exist are skipped, so a rerun only makes what is missing. Generation runs
# twice: the second pass resends jobs that failed on the server (DEADLINE_EXCEEDED,
# transient 403/429), which return no clip. Invalid clips are NOT regenerated here;
# regenerate them with a new seed by hand (--repeat 1000), per the segmentation
# doc's section 6.1.
#
# Needs GEMINI_API_KEY in the environment; this script never prints it.
set -u
cd "$(dirname "$0")/.."
probe=$1; shift
gen=configs/ablations/gen.$probe.yaml; ev=configs/ablations/eval.$probe.yaml
[[ -f $gen && -f $ev ]] || { echo "no configs for $probe in configs/ablations/"; exit 1; }
[[ -n ${GEMINI_API_KEY:-} ]] || { echo "GEMINI_API_KEY is not set"; exit 1; }
for pass in 1 2; do
  for arm in "$@"; do
    echo "### gen $probe $arm (pass $pass)"
    venv/bin/python scripts/run_gen.py --config $gen --arm $arm 2>&1 \
      | sed -E 's/AIza[0-9A-Za-z_-]+/<key>/g' | grep -E "^  $probe |rror|Trace" | cut -c1-240
  done
done
for arm in "$@"; do
  echo "### eval $probe $arm"
  venv/bin/python scripts/run_eval.py --config $ev --arm $arm 2>&1 | grep -E "rror|Trace|summary:"
done
echo DONE
