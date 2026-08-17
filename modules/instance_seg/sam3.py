"""SAM 3 image inference, isolated.

Everything that depends on the exact shape of the SAM 3 API lives in `_load` and
`Sam3ImageSegmenter.segment`, so a change in `transformers` is a two-function
fix rather than a hunt through the probe.

torch and transformers are imported lazily and the checkpoint is loaded on first
use, so the rest of the pipeline — and every other probe — runs without them
installed. Loaded models are cached per (checkpoint, device, dtype) and shared
across samples: this is the expensive object in the run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

_LOADED: dict[tuple[str, str, str], tuple[Any, Any, str]] = {}


@dataclass
class Instance:
    """One detection: a binary mask on the input frame's pixel grid."""

    mask: np.ndarray  # bool (H, W)
    score: float
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    label: str  # the concept prompt that produced it

    @property
    def area(self) -> int:
        return int(self.mask.sum())


class Sam3ImageSegmenter:
    """Concept-prompted segmentation of a single image.

    SAM 3 takes a text concept ("the red block", "objects on the table") and
    returns every instance of it, so what counts as an instance is decided by
    the prompt rather than by the model alone.
    """

    def __init__(self, model_id: str, device: str | None = None, dtype: str = "float32") -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

    def segment(
        self,
        image: np.ndarray,
        prompt: str,
        *,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> list[Instance]:
        """Run one concept prompt over an RGB uint8 frame."""
        import torch

        processor, model, device = _load(self.model_id, self.device, self.dtype)
        torch_dtype = getattr(torch, self.dtype)

        inputs = processor(images=image, text=prompt, return_tensors="pt")
        # BatchFeature.to casts floating-point tensors only; token ids stay integral.
        inputs = inputs.to(device=device, dtype=torch_dtype)

        with torch.inference_mode():
            outputs = model(**inputs)

        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=score_threshold,
            mask_threshold=mask_threshold,
            target_sizes=[image.shape[:2]],
        )[0]

        masks = _to_numpy(result["masks"])
        if masks.dtype != bool:
            masks = masks > 0.5
        scores = _to_numpy(result["scores"]).astype(float).reshape(-1)
        boxes = _to_numpy(result["boxes"]).astype(float) if "boxes" in result else np.zeros((len(scores), 4))

        return [
            Instance(
                mask=np.ascontiguousarray(masks[i]),
                score=float(scores[i]),
                box=tuple(boxes[i][:4].tolist()),
                label=prompt,
            )
            for i in range(len(scores))
        ]


def _load(model_id: str, device: str | None, dtype: str) -> tuple[Any, Any, str]:
    try:
        import torch
        from transformers import Sam3Model, Sam3Processor
    except ImportError as exc:  # a missing optional dep should say what to install
        raise ImportError(
            "the instance_seg ground truth needs torch and a transformers with SAM 3: "
            "pip install -r modules/instance_seg/requirements.txt"
        ) from exc

    key = (model_id, device or "auto", dtype)
    if key not in _LOADED:
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        processor = Sam3Processor.from_pretrained(model_id)
        model = Sam3Model.from_pretrained(model_id)
        model.to(device=resolved, dtype=getattr(torch, dtype))
        model.eval()
        _LOADED[key] = (processor, model, resolved)
    return _LOADED[key]


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
