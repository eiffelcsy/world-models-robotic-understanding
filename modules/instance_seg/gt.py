"""Pseudo ground truth for the instance segmentation probe.

The probe asks the world model to repaint the scene so that every object carries
its own flat color. What it is scored against is what SAM 3 sees in the
conditioning frame: a set of binary instance masks, derived from the input frame
alone and so never contaminated by the generated video.

Every answer is written to `results/instance_seg_gt/` as a mask file plus a
picture of it, and read back from there on the next run, so SAM 3 is loaded only
for the samples that do not have one yet.

SAM 3 is prompted with a concept rather than with points, so "what is an
instance" is a text decision and it is made here. `prompt` in probe.yaml sets the
default for the run; a sample overrides it through its manifest `meta` (see
`prompt_key`), which is how a scene with a known object list gets a tighter
prompt than the generic one.
"""

from __future__ import annotations

import numpy as np

from wmeval import GroundTruthSource, ProbeData, register
from wmeval.types import RunContext, Sample

from .sam3 import Instance, Sam3ImageSegmenter
from .viz import overlay


@register("gt", "sam3_instances")
class Sam3Instances(GroundTruthSource):
    """Input frame -> `ProbeData("instances", {"masks": [bool HxW, ...]})`.

    Params, all settable from probe.yaml:

      model_id         HF repo id or local path of the SAM 3 checkpoint
      device           cuda | cpu, or null to pick whatever torch reports
      dtype            float32 | float16 | bfloat16
      prompt           concept to segment; a list runs several and merges them
      prompt_key       manifest meta key overriding `prompt` for one sample
      score_threshold  detections below this confidence are dropped
      mask_threshold   logit cut turning a soft mask into a binary one
      min_area_frac    drop specks below this fraction of the frame
      max_area_frac    drop masks covering more than this — usually background
      dedup_iou        drop a mask overlapping a better-scoring one by more
      max_instances    keep at most this many, highest score first
      min_instances    fail the sample if fewer than this survive

    Answers are stored under `<gt_dir>/instance_seg_gt/<sample_id>_gt.npz`, with
    a `.png` of the masks beside each one, and a sample that has a file there is
    not re-segmented. Changing a param does not invalidate what is stored — the
    run logs a warning instead, and `preview.py` is the way to regenerate.
    """

    store_name = "instance_seg_gt"
    # Everything that changes the masks. `min_instances` only decides whether a
    # thin result raises, and so is not a reason to consider a file stale.
    store_params = (
        "model_id",
        "dtype",
        "prompt",
        "prompt_key",
        "score_threshold",
        "mask_threshold",
        "min_area_frac",
        "max_area_frac",
        "dedup_iou",
        "max_instances",
    )

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        device: str | None = None,
        dtype: str = "float32",
        prompt: str | list[str] = "object",
        prompt_key: str = "gt_prompt",
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        min_area_frac: float = 0.001,
        max_area_frac: float = 0.9,
        dedup_iou: float = 0.8,
        max_instances: int = 20,
        min_instances: int = 1,
    ) -> None:
        super().__init__(
            model_id=model_id,
            device=device,
            dtype=dtype,
            prompt=prompt,
            prompt_key=prompt_key,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
            min_area_frac=min_area_frac,
            max_area_frac=max_area_frac,
            dedup_iou=dedup_iou,
            max_instances=max_instances,
            min_instances=min_instances,
        )
        self._segmenter = Sam3ImageSegmenter(model_id, device, dtype)

    def compute(self, sample: Sample, input_frame: np.ndarray, ctx: RunContext) -> ProbeData:
        prompts = self._prompts_for(sample)
        found: list[Instance] = []
        for prompt in prompts:
            found.extend(
                self._segmenter.segment(
                    input_frame,
                    prompt,
                    score_threshold=self.score_threshold,
                    mask_threshold=self.mask_threshold,
                )
            )

        shape = tuple(input_frame.shape[:2])
        instances = self._select(found, shape)
        if len(instances) < self.min_instances:
            raise RuntimeError(
                f"SAM 3 returned {len(instances)} usable instance(s) for prompt {prompts} "
                f"on {sample.sample_id}; at least {self.min_instances} required"
            )

        return ProbeData(
            "instances",
            {"masks": [instance.mask for instance in instances]},
            {
                "source": "sam3",
                "model_id": self.model_id,
                "prompts": prompts,
                "frame_shape": list(shape),
                "n_instances": len(instances),
                "n_detections": len(found),
                "scores": [round(instance.score, 4) for instance in instances],
                "labels": [instance.label for instance in instances],
                "boxes": [[round(v, 1) for v in instance.box] for instance in instances],
            },
        )

    def render(self, data: ProbeData, input_frame: np.ndarray) -> np.ndarray:
        """The stored masks tinted over the frame they came from."""
        return overlay(input_frame, [np.asarray(m, dtype=bool) for m in data.get("masks") or []])

    def _prompts_for(self, sample: Sample) -> list[str]:
        prompt = sample.meta.get(self.prompt_key, self.prompt) if self.prompt_key else self.prompt
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts:
            raise ValueError(f"no concept prompt for {sample.sample_id}")
        return prompts

    def _select(self, instances: list[Instance], shape: tuple[int, int]) -> list[Instance]:
        """Score-ordered, area-filtered, overlap-deduplicated, capped.

        Several prompts over one frame will name the same object twice, and SAM 3
        happily returns a mask of the whole table alongside the things on it;
        both would inflate the instance count the model is scored against.
        """
        pixels = float(shape[0] * shape[1])
        kept: list[Instance] = []
        for instance in sorted(instances, key=lambda i: i.score, reverse=True):
            area = instance.area / pixels
            if area < self.min_area_frac or area > self.max_area_frac:
                continue
            if any(_iou(instance.mask, other.mask) >= self.dedup_iou for other in kept):
                continue
            kept.append(instance)
            if len(kept) >= self.max_instances:
                break
        return kept


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0
