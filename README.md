# world-models-robotic-understanding

`wmeval` — a probe-based evaluation pipeline for video generative world models.

A video model can only answer in pixels, so every question about a scene is posed
as something that happens on screen: "how far away is everything?" becomes "the
photograph is replaced by a depth map." This pipeline scores what came back.

It is model-agnostic. It takes generated videos plus the frames they were
conditioned on, and never touches the generator.

---

## Quickstart

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

1. **Add a probe.** One directory under `modules/`, holding a `probe.yaml` plus
   the code it names. See [modules/\_\_init\_\_.py](modules/__init__.py) for the
   layout and a registration example.

2. **Point at your videos.** Edit `configs/scan.yaml`, then:

   ```bash
   python scripts/make_manifest.py --config configs/scan.yaml
   ```

   It scans `run_dir/<probe>/<sample_id>.mp4` against `frames_dir/<sample_id>.png`
   and writes `configs/manifest.yaml`. Hand-write it instead if your layout
   differs — see [configs/manifest.example.yaml](configs/manifest.example.yaml).

3. **Run.**

   ```bash
   python scripts/run_eval.py --config configs/eval.yaml
   ```

   Results land in `output_dir`: `summary.md`, `summary.json`, `scores.csv`
   (one row per sample and probe), and `results.json` with full records.

`python scripts/run_eval.py --list-components` prints everything your modules
have registered.

---

## The pipeline

For each probe, on each sample:

```
input frame ──▶ ground-truth source ──▶ pseudo GT ────┐
                                                      ├──▶ metrics ──▶ scores
output video ──▶ readout ──▶ output frame ──▶ extractor ──▶ prediction ─┘
```

1. **Ground truth** is derived from the conditioning frame only, by whatever
   methodology the probe defines. It never sees the generated video, so it cannot
   be contaminated by it. Since this is the expensive stage, a source can keep
   its answers in a store — `<gt_dir>/<probe>_gt/<sample_id>_gt.npz` — which the
   pipeline reads when it is there and fills when it is not. Params are not part
   of the filename: regenerating is deleting the files, or running the module's
   own preview tool.
2. **Readout** picks the one frame of the generated video carrying the answer —
   the prompts ask for an overlay to arrive and then hold still, so it is in the
   tail. `last_frame`, `fixed_time` and `most_static` ship with the pipeline.
3. **Extractor** decodes that frame's pixels into probe data: a depth field, a
   region, instance masks, keypoints. The only place that knows the convention.
4. **Metrics** score prediction against ground truth.

A probe that sets `best_frame` in its `probe.yaml` reports a second set of
scores, `best_`-prefixed: the same metrics on whichever frame of the clip scores
highest. That frame is chosen *by* the ground truth, so it is an oracle upper
bound rather than a deployable score, and it is reported next to the readout's
number rather than instead of it — the convention in
[arXiv:2509.20328](https://arxiv.org/abs/2509.20328), where a video model is
observed to keep animating past the point where it had the answer right.

The prediction is resized onto the ground truth's pixel grid before scoring, so a
generation at a different resolution is not an error. The `kind` field on
`ProbeData` is the contract between an extractor and a metric, and is checked
before scoring so a mis-wired `probe.yaml` raises rather than producing a number
that means nothing. A failing sample is recorded and the run continues; set
`fail_fast: true` to stop on the first error.

---

## Layout

```
configs/
  eval.yaml              the run: manifest, output, ground-truth store, which modules
  scan.yaml              input to make_manifest.py
  manifest.example.yaml  what a manifest looks like
modules/                 one directory per probe  <- add and remove probes here
wmeval/
  types.py               Sample, ProbeData, Result, RunContext
  components.py          GroundTruthSource, Extractor, Metric base classes
  registry.py            name -> component lookup
  modules.py             module discovery and import
  config.py              YAML -> config objects
  manifest.py            frames + videos + ground-truth assets
  video.py               readouts
  probe.py               the four stages, per sample
  runner.py              orchestration
  report.py              aggregation -> summary.md / summary.json / scores.csv
scripts/                 run_eval.py, make_manifest.py
```
