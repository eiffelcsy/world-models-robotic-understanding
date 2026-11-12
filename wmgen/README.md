# wmgen

Generating the videos `wmeval` scores. Conditioning frames plus a per-probe
prompt in; clips plus a manifest out.

```bash
python scripts/run_gen.py --config configs/gen.yaml --dry-run   # free, always first
python scripts/run_gen.py --config configs/gen.yaml
python scripts/run_eval.py --config configs/eval.yaml           # reads what it wrote
```

## The pipeline

For each probe, on each frame:

```
frames dir ──▶ source ──▶ Frame ──▶ prompt ──▶ Prompt ──▶ backend ──▶ clip ──▶ checks
                                       ▲                                          │
                            modules/<probe>/prompt.yaml           configs/manifest.yaml
```

1. **Source** yields the conditioning frames. The sample id is the file stem,
   which is also how the manifest, the clip store and wmeval's ground-truth
   store name that sample — so all three are findable from each other by eye.
2. **Prompt** renders `modules/<probe>/prompt.yaml` for one frame, substituting
   `{placeholders}` from the frame's metadata and the video params.
3. **Backend** submits the job, waits, and downloads the clip to
   `<clips_dir>/<probe>/<sample_id>.mp4`.
4. **Checks** decide whether what came back is a video worth scoring. They are
   not scores — grading the generation is wmeval's job, with ground truth.

Then the manifest, which is the actual product: `configs/manifest.yaml`, in the
schema `wmeval` reads, with each frame's `meta` carried through.

## Model-agnostic, the other way round

wmeval is model-agnostic because it never calls a generator. wmgen has to call
one, so it is model-agnostic by making that call configuration rather than code:
**a world model is a `backends/<name>/backend.yaml` file.** Three general
mechanisms ship, and between them they cover how video models are actually
reached:

| type | for |
|---|---|
| `http_job` | hosted APIs — submit, poll, download. Built on stdlib `urllib`, so a new model adds no dependency. |
| `command` | local or self-hosted checkpoints, behind a templated command line. Keeps torch out of wmgen. |
| `local_dir` | clips generated elsewhere, adopted into the run layout. No key, no cost. |

`backends/omni12/backend.yaml` is a worked `http_job`; see
[backends/\_\_init\_\_.py](../backends/__init__.py) for the layout and for the
escape hatch when YAML genuinely cannot express a vendor's call.

Never write a key into a `backend.yaml`. Write `${MY_API_KEY}`; an unset
variable fails before any request is sent, rather than arriving as a 401.

## The clip store

The same caching contract as wmeval's ground-truth store, for the same reason —
this is the expensive stage — except that here it is expensive in money:

```
clips/<probe>/<sample_id>.mp4    the video wmeval will score
clips/<probe>/<sample_id>.json   the prompt, params and job id that produced it
```

A clip that is there is **not** generated again. Nothing about the prompt is in
the filename, so editing `prompt.yaml` does not orphan what you have already
paid for: the run warns, naming the drift, and uses the stored clip anyway.
Regenerating is deleting the file, or `--refresh`.

Two exceptions to "never regenerate". A stored clip that no longer passes its
checks — a truncated download — is dropped and made again, once, rather than
becoming an error the store repeats forever. And a clip that fails its checks on
the way in is deleted rather than stored, so a bad file never becomes permanent.

## Costs, and the dry run

`--dry-run` renders every prompt and writes the manifest that would result,
calling no backend, and leaves `results/gen/prompts.jsonl` to read. Do that
first. It is where a template with a hole in it is meant to fail — a prompt is
the entire instruction to the model, and "over the next  seconds" costs exactly
as much to generate as a good one.

`--limit N` before a paid run; `limit: 1` and one real clip is how a new
`backend.yaml` gets verified, since a dry run cannot check a response path.

## Writing a prompt

`modules/<probe>/prompt.yaml` sits beside `probe.yaml` because what a probe asks
for and what its extractor decodes are one convention. `modules/instance_seg/`
has both, and its prompt is commented clause by clause with which extractor
param each one exists to satisfy — that is the pairing to imitate.

The pipeline is serial and does not resume: interrupting a run abandons whatever
the backend is generating at that moment. If the vendor delivers it anyway, the
`local_dir` backend is how it is adopted rather than bought twice.
