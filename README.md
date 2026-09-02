# world-models-robotic-understanding

Two pipelines and a probe library, for asking video generative world models what
they understand about a robot's scene.

A video model can only answer in pixels, so every question about a scene is
posed as something that happens on screen: "how far away is everything?" becomes
"the photograph is replaced by a depth map."

- **`wmgen`** asks the question. Conditioning frames plus a per-probe prompt in,
  generated clips plus a manifest out.
- **`wmeval`** scores what came back. Generated videos plus the frames they were
  conditioned on in, numbers out.

Both are model-agnostic, by opposite routes. `wmeval` never touches a generator
at all. `wmgen` must call one, so a world model there is a
`backends/<name>/backend.yaml` file rather than code — adding Veo, Kling, a
self-hosted checkpoint or a directory of clips made by hand is configuration.

```
conditioning frames ──wmgen──▶ clips + manifest ──wmeval──▶ scores
                        ▲                            ▲
              modules/<probe>/prompt.yaml   modules/<probe>/probe.yaml
```

A probe owns both halves of its convention, in one directory: the prompt that
asks for an appearance change, and the extractor that decodes it. They change
together or the probe measures nothing.

---

## Quickstart

```bash
python -m venv venv
venv/Scripts/python.exe -m pip install -r requirements.txt
```

1. **Add a probe.** One directory under `modules/`, holding a `probe.yaml`, a
   `prompt.yaml` and the code they name. See
   [modules/\_\_init\_\_.py](modules/__init__.py) for the layout and a
   registration example. A directory with only `probe.yaml` is eval-only; only
   `prompt.yaml`, generation-only.

2. **Generate.** Point `configs/gen.yaml` at your frames and pick a backend,
   then:

   ```bash
   python scripts/run_gen.py --config configs/gen.yaml --dry-run
   python scripts/run_gen.py --config configs/gen.yaml
   ```

   The dry run renders every prompt and writes the manifest that would result
   without calling any backend. Generation costs money; do it first. Clips land
   in `clips_dir` and the manifest at `manifest_out`.

   Already have videos? Use the `local` backend to adopt them, or hand-write the
   manifest — see [configs/manifest.example.yaml](configs/manifest.example.yaml)
   and `scripts/make_manifest.py`.

3. **Evaluate.**

   ```bash
   python scripts/run_eval.py --config configs/eval.yaml
   ```

   Results land in `output_dir`: `summary.md`, `summary.json`, `scores.csv`
   (one row per sample and probe), and `results.json` with full records.

`--list-components` on either script prints everything registered.

---

## The evaluation pipeline

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

## The generation pipeline

The mirror image, and it borrows the same mechanisms — a registry of named
components, plugin directories discovered by a marker file, YAML-only configs, a
store that makes reruns free. See [wmgen/README.md](wmgen/README.md); the short
version is that generated clips are cached exactly as ground truth is, because
this is that pipeline's expensive stage, and that a clip already on disk is never
made again.

---

## Layout

```
configs/
  gen.yaml               the generation run: frames, backend, output, which modules
  eval.yaml              the evaluation run: manifest, output, ground-truth store
  scan.yaml              input to make_manifest.py
  manifest.example.yaml  what a manifest looks like
modules/                 one directory per probe   <- add and remove probes here
  <probe>/probe.yaml       how to score it
  <probe>/prompt.yaml      what to ask for
backends/                one directory per world model  <- add models here
wmcore/                  what both pipelines are built from
  registry.py            name -> component lookup
  modules.py             plugin discovery and import
  manifest.py            the schema the two halves agree on
  config.py, utils.py, video.py, components.py
wmgen/
  types.py               Frame, Prompt, Request, Clip, GenContext
  components.py          FrameSource, PromptBuilder, VideoBackend, ClipCheck
  sources.py             where frames come from
  prompt.py              template rendering
  backends.py            http_job, command, local_dir
  checks.py              is the clip usable
  store.py               the clip store: cache, sidecars, prompt drift
  task.py                the stages, per frame
  runner.py, report.py, config.py
wmeval/
  types.py               Sample, ProbeData, Result, RunContext
  components.py          GroundTruthSource, Extractor, Metric base classes
  config.py              YAML -> config objects
  manifest.py            frames + videos + ground-truth assets
  video.py               readouts
  probe.py               the four stages, per sample
  runner.py              orchestration
  report.py              aggregation -> summary.md / summary.json / scores.csv
scripts/                 run_gen.py, run_eval.py, make_manifest.py
```
