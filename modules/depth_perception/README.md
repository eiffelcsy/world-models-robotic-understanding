# depth_perception

Ask the world model to replace the conditioning frame with a grayscale depth
map, then score the depth that comes back against Depth Anything 3 (DA3) run on
the input frame (RoboTrustBench), or against the simulator's metric depth on
LIBERO (`sim_depth`, selected by `configs/eval.libero.yaml`; see
[docs/libero_ground_truth.md](../../docs/libero_ground_truth.md)).

## The two halves

| file | pipeline | says |
|---|---|---|
| [prompt.yaml](prompt.yaml) | wmgen | replace the photograph with a grayscale depth map, near bright, far dark, then hold still |
| [probe.yaml](probe.yaml) | wmeval | decode brightness back into depth and align-score it against DA3 |

They are one convention written twice. The extractor param `near_is_bright`
exists because the prompt asks for bright = close; the prompt's "no colormap"
clause exists because `GrayDepth` reads luminance and nothing else.

Render the prompt without generating anything:

```bash
python scripts/run_gen.py --config configs/gen.yaml --dry-run
```

## Install

```bash
venv/Scripts/python.exe -m pip install -r modules/depth_perception/requirements.txt
```

DA3 is the authors' own package, not a `transformers` class:

```bash
git clone https://github.com/ByteDance-Seed/depth-anything-3
cd depth-anything-3 && pip install -e .
```

Weights come from the Hub (`depth-anything/DA3-LARGE-1.1`, `DA3-BASE`, ...).

## Why the metrics align first

DA3 returns *relative* depth; so does a generative model. Neither shares a scale
or a zero with the other, so `DepthMetrics` fits the prediction to the ground
truth before measuring:

| `align` | fit |
|---|---|
| `none` | no fit; only meaningful if both sides already share a scale |
| `median` | scale by the ratio of medians |
| `scale` | least-squares scale |
| `scale_shift` | least-squares scale and offset — the default |

## The scores

| score | meaning |
|---|---|
| `delta1`/`delta2`/`delta3` | fraction of pixels within 1.25^k of the ground truth — the headline, and the only higher-is-better score |
| `abs_rel`, `sq_rel`, `rmse`, `rmse_log` | the usual monocular-depth errors, after alignment |
| `silog` | scale-invariant log error |
| `n_valid`, `valid_frac` | pixels that survived masking; a score over a handful must not read as a result |

`best_frame.rank_by` is `delta1` because every other score here is
lower-is-better and `best_frame` maximises.

## Regenerating and inspecting the store

```bash
venv/Scripts/python.exe -m modules.depth_perception.preview path/to/frames/
venv/Scripts/python.exe -m modules.depth_perception.scan generated.mp4 \
    --gt results/depth_perception_gt/<sample_id>_gt.npz
```

`preview` refills `<gt_dir>/depth_perception_gt/`; `scan` reports the readout
frame against the best frame and writes the per-frame curve.
