"""A probe = readout + ground-truth source + extractor + metrics.

`Probe.evaluate` is the whole per-sample pipeline, in the order the four stages
were specified:

  1. pseudo ground truth from the *input* frame (loaded from its store, or computed)
  2. an output frame read out of the *generated* video
  3. probe data extracted from that frame's pixels
  4. prediction scored against ground truth

Two numbers, not one
--------------------
A video model keeps animating after it has finished the task, so the frame
carrying the answer is not reliably the frame a readout lands on. Wiedemer et al.
(arXiv:2509.20328) report a pair for exactly that reason, and so does this probe
wherever `best_frame` is configured:

  <metric>        the last frame (the readout). Predetermined and
                  ground-truth-blind, so it is the number a model could
                  actually be deployed against.
  best_<metric>   the best frame: the best any frame of the clip reaches, chosen
                  *by* the ground truth. An upper bound — what the clip
                  contained at its best moment, not what a readout can reach.
                  Never quote it without the label.

The gap between the two is what the readout is losing, and it is the thing to
look at before tuning one.

Failures are returned as `Result(status="error")` rather than raised, so one bad
video does not end a 200-sample run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wmcore import registry
from wmcore.utils import read_image, resize_to
from wmcore.video import iter_frames

from . import gates as _gates  # noqa: F401  (registers the shipped gates)
from .config import ProbeConfig
from .gates import GateContext
from .types import ProbeData, Result, RunContext, Sample


@dataclass
class Probe:
    name: str
    readout: Any
    ground_truth: Any
    extractor: Any
    metrics: list[Any]
    description: str = ""
    best_frame: dict[str, Any] | None = None  # {rank_by, stride}; None skips the best-frame pass
    level: str = ""  # Scene | Interaction | Robot State, for the report only
    headline: str = ""  # which score heads the summary table
    gates: list[Any] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: ProbeConfig) -> "Probe":
        best_frame = dict(config.best_frame)
        if not best_frame.pop("enabled", False):
            best_frame = {}
        elif "rank_by" not in best_frame:
            raise ValueError(f"{config.name}: best_frame needs `rank_by`, the score to maximise")
        return cls(
            name=config.name,
            readout=registry.build("readout", config.readout),
            ground_truth=registry.build("gt", config.ground_truth),
            extractor=registry.build("extractor", config.extractor),
            metrics=[registry.build("metric", spec) for spec in config.metrics],
            description=config.description,
            best_frame=best_frame or None,
            level=config.level,
            headline=config.headline,
            gates=[registry.build("gate", spec) for spec in config.gates],
        )

    def evaluate(self, sample: Sample, ctx: RunContext) -> Result:
        video = sample.video_for(self.name)
        if video is None:
            return Result(sample.sample_id, self.name, "skipped", error="no video for this probe", meta=sample.meta)
        if not video.exists():
            return Result(sample.sample_id, self.name, "skipped", error=f"missing video: {video}", meta=sample.meta)

        info: dict[str, Any] = {"video": str(video)}
        gate_results: list[dict[str, Any]] = []
        valid, invalid_reason = True, None
        try:
            input_frame = read_image(sample.input_frame)
            truth = self.ground_truth(sample, input_frame, ctx)

            output_frame, readout_info = self.readout(video)
            info.update(readout_info)

            prediction = self.extractor(output_frame, sample, ctx)
            prediction, truth = _align(prediction, truth)
            scores = self._score(prediction, truth)

            # Fidelity gates screen the rollout before its score is trusted. A
            # failure flags the row, it does not change the number.
            gate_results = self._run_gates(sample, input_frame, output_frame, prediction, video, info)
            valid = all(g["valid"] for g in gate_results)
            invalid_reason = next((g["reason"] for g in gate_results if not g["valid"]), None)

            if self.best_frame:
                scores.update(self._best_over_frames(video, sample, ctx, truth, scores, info))
        except Exception as exc:  # noqa: BLE001 - one sample must not end the run
            return Result(
                sample.sample_id, self.name, "error", info=info,
                error=f"{type(exc).__name__}: {exc}", meta=sample.meta,
            )

        info["prediction"] = prediction.info
        info["ground_truth"] = truth.info
        info["gates"] = gate_results
        return Result(
            sample.sample_id, self.name, "ok",
            scores=scores, info=info, valid=valid, invalid_reason=invalid_reason, meta=sample.meta,
        )

    def _run_gates(
        self,
        sample: Sample,
        input_frame: Any,
        output_frame: Any,
        prediction: ProbeData,
        video: Path,
        info: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not self.gates:
            return []
        context = GateContext(
            sample=sample,
            input_frame=input_frame,
            output_frame=output_frame,
            prediction=prediction,
            video=video,
            readout=info,
        )
        return [gate(context) for gate in self.gates]

    def _score(self, prediction: ProbeData, truth: ProbeData) -> dict[str, float]:
        scores: dict[str, float] = {}
        for metric in self.metrics:
            _check_kinds(metric, prediction, truth)
            scores.update(metric(prediction, truth))
        return scores

    def _best_over_frames(
        self,
        video: Path,
        sample: Sample,
        ctx: RunContext,
        truth: ProbeData,
        readout_scores: dict[str, float],
        info: dict[str, Any],
    ) -> dict[str, float]:
        """Score every eligible frame, keep the best, return it `best_`-prefixed.

        Frames before `start_index` are not candidates. The opening frames of a
        generated clip are still the conditioning frame — nothing has been
        painted yet — so a best-frame search free to pick them will sometimes return a
        score won by scene content the model never touched, which is the one
        thing a maximum over frames must not reward.

        Seeded with the readout frame's own scores when that frame is itself
        eligible, so the readout frame is always a candidate and `best_x >= x`
        holds by construction: the pair reads as a floor and a ceiling on the
        same clip. A readout landing before `start_index` breaks that — the two
        numbers then come from disjoint frame sets — so it is recorded rather
        than papered over.

        A frame the extractor chokes on is counted and skipped rather than
        raised — a best-frame pass must not cost the sample its real score.
        """
        rank_by = self.best_frame["rank_by"]
        if rank_by not in readout_scores:
            raise ValueError(
                f"best_frame.rank_by '{rank_by}' is not a score this probe produces; "
                f"available: {', '.join(sorted(readout_scores))}"
            )
        stride = int(self.best_frame.get("stride", 1))
        start_index = max(0, int(self.best_frame.get("start_index", 0)))

        readout_index = int(info.get("frame_index", -1))
        seeded = readout_index >= start_index
        best: dict[str, float] | None = dict(readout_scores) if seeded else None
        best_index = readout_index if seeded else -1
        n_scored, n_failed = (1 if seeded else 0), 0

        for index, frame in iter_frames(video, stride):
            if index < start_index or index == readout_index:
                continue
            try:
                prediction, _ = _align(self.extractor(frame, sample, ctx), truth)
                scores = self._score(prediction, truth)
            except Exception:  # noqa: BLE001 - a bad frame must not lose the readout score
                n_failed += 1
                continue
            n_scored += 1
            if best is None or scores[rank_by] > best[rank_by]:
                best, best_index = scores, index

        if best is None:  # clip shorter than start_index, or every frame failed
            best, best_index = dict(readout_scores), readout_index

        info["best_frame"] = {
            "frame_index": best_index,
            "readout_frame_index": readout_index,
            "rank_by": rank_by,
            "stride": stride,
            "start_index": start_index,
            "n_scored": n_scored,
            "n_failed": n_failed,
            "includes_readout_frame": seeded,
            "note": "chosen by the ground truth: an upper bound, not a deployable score",
        }
        scored = {f"best_{key}": value for key, value in best.items()}
        scored["best_frame_index"] = float(best_index)
        return scored


def _align(prediction: ProbeData, truth: ProbeData) -> tuple[ProbeData, ProbeData]:
    """Put the prediction on the ground truth's pixel grid.

    A generated video rarely comes back at the conditioning frame's resolution,
    so every dense comparison would otherwise be a shape error. Points are
    rescaled by the same factor; the scale is recorded for auditing.
    """
    target, source = truth.shape, prediction.shape
    if target is None or source is None or tuple(target) == tuple(source):
        return prediction, truth

    scale_y, scale_x = target[0] / source[0], target[1] / source[1]
    payload = dict(prediction.payload)
    if "depth" in payload:
        payload["depth"] = resize_to(payload["depth"], target)
    if "mask" in payload:
        payload["mask"] = resize_to(payload["mask"], target, nearest=True)
    if "masks" in payload:
        payload["masks"] = [resize_to(m, target, nearest=True) for m in payload["masks"]]
    if "points" in payload and len(payload["points"]):
        points = payload["points"].astype(float).copy()
        points[:, 0] *= scale_x
        points[:, 1] *= scale_y
        payload["points"] = points

    info = dict(prediction.info)
    info["resized_from"] = list(source)
    info["resized_to"] = list(target)
    return ProbeData(prediction.kind, payload, info), truth


def _check_kinds(metric: Any, prediction: ProbeData, truth: ProbeData) -> None:
    accepts = getattr(metric, "accepts", None)
    if accepts and prediction.kind not in accepts:
        raise TypeError(f"metric {type(metric).__name__} accepts {accepts}, got prediction kind '{prediction.kind}'")
    if prediction.kind != truth.kind:
        raise TypeError(f"kind mismatch: prediction '{prediction.kind}' vs ground truth '{truth.kind}'")
