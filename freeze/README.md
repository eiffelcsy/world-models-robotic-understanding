# Results freeze 1 (2026-11-12)

Git tag: `results-freeze-1`, on the commit that adds this directory.

This freeze records the state of the project when the five prompt ablations
and the held-out confirmations were finished. The code is in git at the tag.
`clips/`, `results/`, `data/` and `semantics/` are gitignored because they are
large, so this directory records a SHA-256 checksum for every file in them.
The numbers in the report can then be traced to the exact clips and ground
truth they came from.

## What is in this directory

| file | contents |
|---|---|
| `checksums/clips.sha256` | every generated clip and its sidecar (1,110 files) |
| `checksums/results.sha256` | every eval and ablation output, rigour table and log (1,573 files) |
| `checksums/data.sha256` | conditioning frames and simulator ground truth for LIBERO and RoboTrustBench (2,523 files; `data/backups/` excluded) |
| `checksums/semantics.sha256` | cached instruction-parsing records (1,207 files) |
| `env_main.txt` | `pip freeze` of `venv` (Python 3.14.8): generation, evaluation, SAM 3, DA3 |
| `env_libero.txt` | installed packages in `venv_libero` (Python 3.10.22): MuJoCo 2.3.7, robosuite 1.4.1, LIBERO 0.1.0 |
| `verify.sh` | checks the files on disk against the checksums |

Paths in the checksum files are relative to the repository root. `.DS_Store`
and `__pycache__` are excluded.

## Generators

| clips | model id | backend | generated | count |
|---|---|---|---|---|
| `clips/libero/` (every ablation and held-out result) | `gemini-omni-1.2-flash` | `backends/omni12` | 2026-11-09 to 2026-11-12 | 435 |
| `clips/robotrustbench/` (midterm pilot only) | `veo-3.1-fast-generate-preview` | `backends/veo3` | 2026-09-03 to 2026-10-10 | 120 |

The model id is also written into each clip's sidecar `.json`.

- **Omni 1.2 Flash may change under the same id.** The provider's docs do not
  say whether it is a preview or a GA model, and `gemini-omni-1.1-flash` is
  routed to 1.2 on 2026-12-03. Clips made after this freeze are not guaranteed to
  come from the same model. The clips listed here are the record of what was
  measured.
- **Veo 3.1 preview has been shut down.** The 120 pilot clips cannot be
  regenerated. They also used different data, ground truth and prompts, so they
  are not comparable with the LIBERO results.

## Other models and code versions

| component | version |
|---|---|
| SAM 3 (`facebook/sam3`, Hugging Face) | revision `3c879f39826c281e95690f02c7821c4de09afae7` |
| Depth Anything 3 (`depth-anything/DA3-LARGE-1.1`) | revision `0e109ae307c5982f319a67cf6f9f99ccdc0ec97c` |
| `third_party/depth-anything-3` | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` |
| `third_party/LIBERO` | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| `third_party/RoboPEPP` (used only in the midterm pilot) | `c4a43d96614a243370d102bf1d56b7c645d08e77` |
| instruction parsing (`semantics:`) | `deepseek-chat`, cached in `semantics/`, never re-queried |

## Checking a copy

From the repository root:

```bash
bash freeze/verify.sh
```

It reports any file that is missing or has changed since the freeze. Files
added after the freeze are not checked. A later freeze (`results-freeze-2`)
gets its own directory rather than overwriting this one.

## Keeping the data safe

Git does not hold the clips, results or ground truth, so the checksums only
prove that a copy is unchanged. They do not keep one. Keep an archive of
`clips/`, `results/`, `data/libero/` and `semantics/` (about 1.9 GB) outside this
machine.
