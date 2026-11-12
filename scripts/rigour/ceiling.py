"""Item 4: the decoder ceilings through a video, the readout and the gates.

    venv/bin/python scripts/rigour/ceiling.py [work_dir [probe,probe]]

The ideal answer for each probe (ideal.py) is made into a clip shaped like a
generated one: 1280x720, 24 fps, 6 s, H.264. The conditioning frame holds for half
a second, cross-fades to the answer over a second, then holds. The clip is then
scored by the unchanged pipeline (`Probe.evaluate`: last-frame readout, decoder,
metric, gates). A perfect answer should score near 1 and pass every gate.

Both variants are encoded at 2200 kb/s, which is what the generator's clips
average. A static clip is easy for the encoder, so the second variant adds fresh
Gaussian grain (sd 4 of 255) to every frame, which makes the encoder spend bits
the way a generated clip with motion and texture does. The readout frame then
carries the grain too.

Clips are written to work_dir (default: a temp folder) and deleted afterwards.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio_ffmpeg
import numpy as np

import common as c
import ideal
from wmeval import video as _video  # noqa: F401  (registers the readouts)
from wmeval.config import load_eval_config
from wmeval.modules import import_module
from wmeval.probe import Probe
from wmeval.types import RunContext, Sample

FPS, SECONDS, HOLD_IN, FADE = 24, 6, 12, 24
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else []  # optional: probes to run
BITRATES = (2200, 2201)  # 2201 marks the grain variant: same rate, noise added to every frame
GRAIN_SD = 4.0
VARIANTS = {
    "instance_seg": {"distinct colours": lambda i: ideal.seg(i), "all one colour": lambda i: ideal.seg(i, same_colour=True)},
    "depth_perception": {"linear in depth": lambda i: ideal.depth(i, "linear"),
                         "linear in inverse depth": lambda i: ideal.depth(i, "disparity")},
    "grasp_region_seg": {"dot on the grasp, 1 cm radius": lambda i: ideal.grasp(i, "grasp", 0.01),
                         "dot on the grasp, 2 cm radius": lambda i: ideal.grasp(i, "grasp", 0.02)},
    "robot_keypoint_detection": {f"dots, radius {r} px": (lambda i, r=r: ideal.keypoints(i, r)) for r in (2, 4, 8)},
    "robot_pose_estimation": {f"dots, radius {r} px": (lambda i, r=r: ideal.pose(i, r)) for r in (2, 4, 8)},
}


def encode(reference: np.ndarray, answer: np.ndarray, path: Path, kbps: int) -> None:
    h, w = reference.shape[:2]
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-b:v", f"{kbps}k",
               "-maxrate", f"{kbps}k", "-bufsize", f"{2 * kbps}k", "-pix_fmt", "yuv420p", str(path)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    ref, ans = reference.astype(np.float32), answer.astype(np.float32)
    rng = np.random.default_rng(0)
    for index in range(FPS * SECONDS):
        t = float(np.clip((index - HOLD_IN) / FADE, 0.0, 1.0))
        frame = (1 - t) * ref + t * ans
        if kbps == 2201:  # fresh grain each frame, so the encoder has to spend bits as it does on a generated clip
            frame = frame + rng.normal(0.0, GRAIN_SD, frame.shape).astype(np.float32)
        process.stdin.write(np.clip(frame.round(), 0, 255).astype(np.uint8).tobytes())
    process.stdin.close()
    if process.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


def probes() -> dict[str, Probe]:
    config = load_eval_config(c.ROOT / "configs/eval.libero.yaml")
    built = {}
    for probe_config in config.probes:
        if probe_config.enabled and probe_config.name in VARIANTS and (not ONLY or probe_config.name in ONLY):
            import_module(config.modules_dir, probe_config.module)
            probe = Probe.from_config(probe_config)
            probe.best_frame = None  # the answer is static: the best frame adds nothing but time
            built[probe.name] = probe
    return built


if __name__ == "__main__":
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="ceiling_"))
    work.mkdir(parents=True, exist_ok=True)
    ctx = RunContext(run_dir=work, gt_dir=c.ROOT / "results/libero")  # stores exist, so they are only read
    rows, failures = [], []
    for name, probe in probes().items():
        extractor, metrics = c.build(name)
        key = probe.headline
        for variant, draw in VARIANTS[name].items():
            still, through = [], {k: [] for k in BITRATES}
            valid = {k: 0 for k in BITRATES}
            for i in c.ids("all"):
                reference, answer = c.read_image(c.frame_path(i)), draw(i)
                still.append(c.score(extractor, metrics, answer, i, c.truth(name, i))[1][key])
                for kbps in BITRATES:
                    path = work / f"{name}__{kbps}__{i}.mp4"
                    encode(reference, answer, path, kbps)
                    result = probe.evaluate(Sample(i, c.frame_path(i), videos={name: path}, meta=c.META[i]), ctx)
                    path.unlink()
                    if result.status != "ok":
                        failures.append(f"{name} / {variant} / {kbps} / {i}: {result.error}")
                        through[kbps].append(0.0)
                        continue
                    through[kbps].append(result.scores[key])
                    valid[kbps] += result.valid
                    if not result.valid:
                        failures.append(f"{name} / {variant} / {"grain" if kbps == 2201 else "clean"} / {i}: gate: {result.invalid_reason}")
            cell = lambda v: f"{np.mean(v):.3f} (min {np.min(v):.3f})"
            rows.append([name, variant, key, cell(still)] +
                        [x for k in BITRATES for x in (cell(through[k]), f"{valid[k]}/20")])
            print(rows[-1], flush=True)
    header = ["probe", "ideal answer", "metric", "still frame"] + \
             ["through video, 2200 kb/s", "passes gates", f"through video, 2200 kb/s + grain (sd {GRAIN_SD:g})", "passes gates"]
    text = ["# Ceilings through video (item 4)",
            "The ideal answer on all 20 LIBERO frames, scored directly and after being encoded as a 6 s H.264 clip and "
            "run through the unchanged pipeline (last-frame readout, decoder, metric, gates).",
            c.md_table(header, rows)]
    if failures:
        text.append("## Ideal answers that failed a gate or errored\n\n" + "\n".join(f"- {f}" for f in failures))
    c.write("ceiling.md" if not ONLY else f"ceiling_{'_'.join(ONLY)}.md", "\n\n".join(text))
    if len(sys.argv) <= 1:
        shutil.rmtree(work, ignore_errors=True)
