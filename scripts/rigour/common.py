"""Shared helpers for the rigour checks (docs/rigour_checks.md).

Everything here reads: clips, conditioning frames and the stored LIBERO ground
truth. Nothing is generated and no ground-truth store is written. Outputs go to
results/rigour/.
"""
from __future__ import annotations

import csv
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from wmcore import registry  # noqa: E402
from wmeval.components import load_probe_data  # noqa: E402
from wmeval.probe import _align  # noqa: E402
from wmeval.types import ProbeData, RunContext, Sample  # noqa: E402
from wmeval.video import LastFrame  # noqa: E402
from wmcore.utils import read_image  # noqa: E402

OUT = ROOT / "results" / "rigour"
OUT.mkdir(parents=True, exist_ok=True)

STORES = {
    "instance_seg": "instance_seg_sim_gt",
    "depth_perception": "depth_perception_sim_gt",
    "grasp_region_seg": "grasp_demonstrated_gt",
    "robot_keypoint_detection": "robot_keypoints_sim_gt",
    "robot_pose_estimation": "robot_pose_recorded_gt",
}
META = yaml.safe_load(open("data/libero/meta.yaml"))["meta"]
CTX = RunContext(run_dir=OUT, gt_dir=None)  # gt_dir None: nothing is ever stored


def ids(which: str = "all") -> list[str]:
    read = lambda f: [l.strip() for l in open(f) if l.strip() and not l.startswith("#")]
    tune, test = read("configs/libero_tune_ids.txt"), read("configs/libero_test_ids.txt")
    return {"tune": tune, "test": test, "all": tune + test}[which]


def suite(sample_id: str) -> str:
    return sample_id.split("__")[0]


def frame_path(sample_id: str) -> Path:
    return ROOT / "data/libero/frames" / f"{sample_id}.png"


def sample(sample_id: str) -> Sample:
    return Sample(sample_id=sample_id, input_frame=frame_path(sample_id), meta=META[sample_id])


def truth(probe: str, sample_id: str) -> ProbeData:
    return load_probe_data(ROOT / "results/libero" / STORES[probe] / f"{sample_id}_gt.npz")


def spec(probe: str) -> dict:
    return yaml.safe_load(open(ROOT / "modules" / probe / "probe.yaml"))


def build(probe: str, **extractor_overrides):
    """The probe's own decoder and metrics, as probe.yaml configures them."""
    importlib.import_module(f"modules.{probe}")
    s = spec(probe)
    ex = dict(s["extractor"])
    ex["params"] = {**(ex.get("params") or {}), **extractor_overrides}
    return registry.build("extractor", ex), [registry.build("metric", m) for m in s["metrics"]]


def score(extractor, metrics, frame: np.ndarray, sample_id: str, gt: ProbeData):
    """Decode one RGB frame and score it, exactly as Probe.evaluate does."""
    pred, gt = _align(extractor(frame, sample(sample_id), CTX), gt)
    scores: dict[str, float] = {}
    for metric in metrics:
        scores.update(metric(pred, gt))
    return pred, scores


def clip(probe: str, arm: str, repeat: str, sample_id: str) -> Path:
    return ROOT / "clips/libero" / probe / arm / repeat / f"{sample_id}.mp4"


def final_clip(probe: str, arm: str, sample_id: str, first: str = "r1") -> tuple[Path, str]:
    """The clip that was scored: the regeneration if there is one, else the first attempt."""
    base = int(first[1:])
    for attempt in (2, 1):
        retry = f"r{base + 1000 * attempt}"
        if clip(probe, arm, retry, sample_id).exists():
            return clip(probe, arm, retry, sample_id), retry
    return clip(probe, arm, first, sample_id), first


def last_frame(video: Path) -> np.ndarray:
    return LastFrame(back_off=1)(video)[0]


def frame_at(video: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise IOError(f"no frame {index} in {video}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def scores_csv(path: Path) -> dict[str, dict[str, str]]:
    return {r["sample_id"]: r for r in csv.DictReader(open(path))} if path.exists() else {}


def boot_ci(values, n: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95% CI of the mean, resampling frames."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), (n, len(v)))].mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def md_table(header: list[str], rows: list[list]) -> str:
    fmt = lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(fmt(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text + "\n")
    print(text)
    print(f"\n-> results/rigour/{name}")
