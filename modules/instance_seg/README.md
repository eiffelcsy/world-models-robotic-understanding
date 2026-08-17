# instance_seg

Ask the world model to repaint the scene so each object carries its own flat
color; score the instances that come back against SAM 3 run on the conditioning
frame.

All four stages are written and `probe.yaml` ships enabled. The extractor's
saturation thresholds are the part still worth calibrating — see
[Calibrating the extractor](#calibrating-the-extractor).

## Install

Torch and transformers are needed by this probe only, and are kept out of the
root `requirements.txt` for that reason:

```bash
venv/Scripts/python.exe -m pip install -r modules/instance_seg/requirements.txt
```

`facebook/sam3` is a gated repo — accept its terms on the Hub and
`huggingface-cli login` once, or point `model_id` at a local checkpoint
directory.

## The ground-truth store

SAM 3 output is not recomputed per run. It lives in a directory, one pair of
files per sample:

```
results/instance_seg_gt/<sample_id>_gt.npz   the masks a metric is scored against
results/instance_seg_gt/<sample_id>_gt.png   the same masks tinted over the frame
```

The pipeline loads the `.npz` when it is there and only loads SAM 3 for the
samples that have none, so a first run fills the directory and every run after it
reads. `<sample_id>` is the conditioning frame's file stem, which is also how the
manifest names the sample — the file for a frame is findable by eye.

The store root is `gt_dir` in `configs/eval.yaml`; the subdirectory name is the
probe's. Delete a file to force that sample to be recomputed, or the directory to
force all of them.

## Regenerating it

Params are not part of the filename, so changing a prompt or a threshold in
`probe.yaml` does **not** invalidate what is stored — the run logs a warning
naming the param that moved and scores against the old masks anyway. Regenerating
is this, and it is also how you tune the prompt in the first place:

```bash
venv/Scripts/python.exe -m modules.instance_seg.preview path/to/frames/
venv/Scripts/python.exe -m modules.instance_seg.preview path/to/frames/ \
    --prompt "objects on the table" "robot arm"
```

Every frame given is re-segmented and its stored answer replaced. Defaults come
from `probe.yaml`, so a run with no flags reproduces exactly what the pipeline
expects; a flag overrides that one param and leaves the rest. A file, a directory
or a glob all work, and the glob is expanded in Python, so it behaves the same in
PowerShell as in bash. Counts and scores go to stdout.

`min_instances` is the one param the preview does not inherit — it is relaxed to
0, so an empty result is shown rather than raised. It only decides whether a thin
result is an error and does not change a mask, so it is not one of the params a
stale-store warning considers.

## Two numbers: the readout frame and the best frame

The eval pipeline reports both, per sample, in `scores.csv` and `summary.md`:

| column | frame | chosen by |
|---|---|---|
| `mean_iou`, `f1`, … | the readout's — `most_static` over the last 2 s | the video alone |
| `best_mean_iou`, `best_f1`, … | whichever frame scores highest | **the ground truth** |

`best_frame_index` says which frame won, and `info["best_frame"]` in
`results.json` records the ranking score, the stride and how many frames were
scored.

This follows [Wiedemer et al., *Video models are zero-shot learners and
reasoners*](https://arxiv.org/abs/2509.20328), which reports the same pair and
explains why: Veo "tends to continue animating a scene even after task
completion, potentially reducing last frame performance."

## Scoring a single generation

```bash
venv/Scripts/python.exe -m modules.instance_seg.scan generated.mp4 \
    --gt results/instance_seg_gt/<sample_id>_gt.npz
```

`--gt` takes a file from the ground-truth store, so SAM 3 is not re-run;
`--frame path/to/input.png --prompt "..."` computes it here instead and leaves
the store alone.

```
generated.mp4: 121 frames @ 24.0 fps, 121 scored  |  gt: 20 instances
  best  frame   87  t= 3.62s  *mean_iou 0.6449  matched_iou 0.8062  ...  16 pred / 20 gt  painted 0.146
  last  frame  120  t= 5.00s  *mean_iou 0.4215  matched_iou 0.4683  ...  18 pred / 20 gt  painted 0.211
```

The same pair as the pipeline's, for one clip:

- **last** is the final frame. Nothing consulted the ground truth to pick it.
- **best** is the frame with the highest `mean_iou` anywhere in the clip, chosen
  *by* the ground truth — the oracle upper bound.

A large gap means the model did understand the scene and then drifted, which is
a readout problem; a low `best` means it never did, which is not. `--rank-by`
picks the score the best frame is chosen on, `--stride N` scores every Nth frame
for a quick pass (the final frame is always scored regardless).

Alongside the two lines it writes `<stem>_best.png` and `<stem>_last.png` —
decoded instances tinted over the generated frame they came from — and
`<stem>_scan.json` with every frame's scores, which is the curve to plot when
the two numbers disagree.

## The metrics

Both sides are unordered mask sets, so scoring is matching first: greedy
one-to-one by descending IoU, then

| score | meaning |
|---|---|
| `mean_iou` | matched IoU summed, over `max(n_pred, n_truth)` — the headline; charged for every unmatched mask on either side |
| `matched_iou` | mean IoU over matched pairs alone: how well the instances it *did* find are outlined |
| `foreground_iou` | union of all predictions vs union of all truth, ignoring how the frame was split up |
| `pq` | panoptic quality at `iou_threshold` |
| `precision` / `recall` / `f1` | at `iou_threshold` |
| `n_pred` / `n_truth` / `n_matched` / `count_error` | raw counts |

## Calibrating the extractor

The decoder reads the colors off the frame's hue histogram rather than matching
a fixed palette, so it does not need to know which colors the model chose. What
it does need is a line between *painted* and *scene*, and that line is
`min_saturation`.

It is not free. Measured on the DROID conditioning frames in this repo, the
fraction of pixels already at S ≥ 0.4 before any repainting runs from 0.04 to
0.90 depending on the lab and the lighting — a scene with a red bin and yellow
tape clears the threshold on its own. Every such region is an instance the model
gets for free, so a generation that ignored the prompt entirely will still score
above zero. Before trusting a run:

```bash
venv/Scripts/python.exe -m modules.instance_seg.scan generated.mp4 \
    --gt results/instance_seg_gt/<sample_id>_gt.npz
```

and compare the `painted` fraction on the reported frames against the same
number for the conditioning frame. If they are close, raise `min_saturation` in
`probe.yaml` until the unpainted frame decodes to near-nothing.

Objects the model paints white, black or grey are invisible to this decoder by
construction — hue is undefined there and the pixels look like unpainted scene.
That is a limit of the convention, not a bug to fix in the extractor; the prompt
has to ask for saturated colors.