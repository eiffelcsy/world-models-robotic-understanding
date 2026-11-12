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

The results reported so far are frozen at git tag `results-freeze-1`; see
[freeze/README.md](freeze/README.md) for generator versions, environments and
checksums of the gitignored clips, results and ground truth.

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

### Prompt arms and repeats

A prompt ablation asks one probe several ways. An arm is a file in the probe's
directory, `modules/<probe>/arms/<arm>.yaml`, holding only what differs from its
`prompt.yaml` (and, if the decoder must change with it, from its `probe.yaml`
`extractor`). `base` is the probe as it stands. Each arm and repeat has its own
clips, reports and manifest, so nothing is reused across arms:

```
clips/<set>/<probe>/<arm>/r<repeat>/<sample_id>.mp4
results/<run>/<arm>/r<repeat>/
configs/manifest.<set>.<arm>.r<repeat>.yaml
```

```bash
python scripts/run_gen.py  --config configs/gen.libero.yaml  --arm E1 --repeat 0
python scripts/run_eval.py --config configs/eval.libero.yaml --arm E1 --repeat 0
```

The backend seed is `seed_base + repeat`, the same for every arm, so arms are
compared on the same seeds. The plan is in
[docs/segmentation_prompt_ablations.md](docs/segmentation_prompt_ablations.md).

### The two eval sets

| task | RoboTrustBench (real video) | LIBERO (simulated) |
|---|---|---|
| `instance_seg` | SAM 3 masks (estimated) | simulator masks (exact) |
| `depth_perception` | Depth Anything 3 (estimated) | simulator depth (exact) |
| `robot_keypoint_detection` | — | simulator link positions (exact) |
| `robot_pose_estimation` | — | recorded end-effector pose (exact) |
| `grasp_region_seg` | — | demonstrated grasp on the simulator's object mask |

RoboTrustBench has no simulator behind it, so it keeps only the two tasks whose
ground truth a model can reasonably estimate. The robot and grasp tasks run only
where exact ground truth exists.

### LIBERO subset

20 simulated Panda episodes from
[lerobot/libero](https://huggingface.co/datasets/lerobot/libero), 5 per suite,
each from a different task. Every probe on this set is scored against the
simulator itself: each sample is replayed in MuJoCo from its original LIBERO demo,
and the conditioning frame, depth, segmentation, robot links, camera and grasped
object are all read from that one replay. [docs/libero_ground_truth.md](docs/libero_ground_truth.md)
has how each ground truth is obtained.

`grasp_region_seg` is scored against the grasp recorded in each demonstration.
Frames come from the approach, before the gripper first closes, so only tasks that
start by picking one named object are used.

The replay runs in its own Python 3.10 environment (`venv_libero`, MuJoCo 2.3.7,
robosuite 1.4.1, LIBERO); setup is in the doc.

```bash
python scripts/fetch_libero.py --out data/libero
venv_libero/bin/python scripts/libero_replay/match_demos.py
LIBERO_CONFIG_PATH=venv_libero/.libero MUJOCO_GL=cgl venv_libero/bin/python scripts/libero_replay/render_gt.py
python scripts/run_gen.py --config configs/gen.libero.yaml --dry-run
python scripts/run_eval.py --config configs/eval.libero.yaml
```

---

## The evaluation pipeline

For each probe, on each sample:

```
input frame ──▶ ground-truth source ──▶ GT ───────────┐
                                                      ├──▶ metrics ──▶ scores
output video ──▶ readout ──▶ output frame ──▶ extractor ──▶ prediction ─┘
```

1. **Ground truth** is derived from the conditioning frame (or, on LIBERO, the
   simulator state behind it) only, by whatever methodology the probe defines. It
   never sees the generated video, so it cannot be contaminated by it. Since this is the expensive stage, a source can keep
   its answers in a store — `<gt_dir>/<probe>_gt/<sample_id>_gt.npz` — which the
   pipeline reads when it is there and fills when it is not. Params are not part
   of the filename: regenerating is deleting the files, or running the module's
   own preview tool where it has one (`instance_seg`, `depth_perception`).
2. **Readout** picks the one frame of the generated video carrying the answer —
   the prompts ask for an overlay to arrive and then hold still, so it is in the
   tail. `last_frame`, `fixed_time` and `most_static` ship with the pipeline.
3. **Extractor** decodes that frame's pixels into probe data: a depth field, a
   region, instance masks, keypoints. The only place that knows the convention.
4. **Metrics** score prediction against ground truth.

A probe that sets `best_frame` in its `probe.yaml` reports a second set of
scores, `best_`-prefixed: the same metrics on whichever frame of the clip scores
highest (the *best frame*, beside the *last frame* the readout takes). That
frame is chosen *by* the ground truth, so it is an upper
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

One optional stage sits between the frames and the prompts. A probe can scope
itself to a task — "mask the objects *this task* involves" — but the only place
those objects are named is the instruction, in free text. `semantics:` turns each
instruction into a small record before any prompt is rendered, and writes two
things into the frame's metadata:

```
instruction ──semantics──▶ relevant_objects   what the prompt asks to be painted
                       └─▶ gt_prompt          what SAM 3 segments to score it
```

Both, because scoping one without the other is how this probe quietly measures
nothing: a model asked to paint the marker and the mug, graded against a ground
truth containing every object on the table, loses points for obeying. It is one
small chat completion per sample against any OpenAI-compatible endpoint, cached
at `<semantics_dir>/<sample_id>.json` and never repeated, with a JSON overrides
file for the ones it gets wrong. A run with no `semantics:` block never calls
anything, which is the right setting for a probe whose prompt names no object.

---

## Layout

```
configs/
  gen.yaml               the generation run: frames, backend, output, which modules
  eval.yaml              the evaluation run: manifest, output, ground-truth store
  {gen,eval}.robotrustbench.yaml, {gen,eval}.libero.yaml   the two eval sets
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
  components.py          FrameSource, SlotSource, PromptBuilder, VideoBackend, ClipCheck
  sources.py             where frames come from
  semantics.py           instruction -> the objects it involves (cached, optional)
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
  simulator.py           reads a sample's simulator replay record (LIBERO)
scripts/                 run_gen.py, run_eval.py, make_manifest.py
  fetch_robotrustbench.py, fetch_libero.py   materialize the two eval sets
  libero_replay/         match_demos.py, render_gt.py, review_gt.py (venv_libero)
docs/libero_ground_truth.md   how every LIBERO ground truth is obtained
```
