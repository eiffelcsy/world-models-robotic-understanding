"""Depth Anything 3 inference, isolated.

Everything that depends on the exact shape of the `depth_anything_3` API lives
in `_load` and `DepthAnything3Estimator.predict`, so a change in the package is
a two-function fix rather than a hunt through the probe.

The package is the authors' own, not a `transformers` class:

    git clone https://github.com/ByteDance-Seed/depth-anything-3
    cd depth-anything-3 && pip install -e .

Weights come from the Hub (`depth-anything/DA3-LARGE-1.1`, `DA3-BASE`, ...).

The import and checkpoint load are lazy and cached per (model_id, device), so
the rest of the pipeline — and every other probe — runs without the package
installed.
"""

from __future__ import annotations

import tempfile
from typing import Any

import numpy as np

_LOADED: dict[tuple[str, str], tuple[Any, str]] = {}


class DepthAnything3Estimator:
    """Monocular relative depth from a single RGB frame.

    DA3 is a multi-view model, but a single image is a valid view set and is
    what this probe conditions on. The returned field is *relative* depth: it
    is up to an unknown affine transform of metric depth, which is why the
    metrics align before scoring.
    """

    def __init__(self, model_id: str = "depth-anything/DA3-LARGE-1.1", device: str | None = None) -> None:
        self.model_id = model_id
        self.device = device

    def predict(self, image: np.ndarray) -> np.ndarray:
        """RGB uint8 (H, W, 3) -> relative depth (H, W) float32, larger = further."""
        model, device = _load(self.model_id, self.device)

        prediction = self._inference(model, image)
        depth = _to_numpy(prediction.depth)
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[0]
        return depth

    def _inference(self, model: Any, image: np.ndarray) -> Any:
        """Call `model.inference`, tolerating an export directory being required."""
        try:
            return model.inference([image])
        except TypeError:
            # Some builds require an export target even when only the arrays are
            # wanted; a scratch directory keeps the call honest without leaving
            # artefacts in the run.
            with tempfile.TemporaryDirectory() as tmp:
                return model.inference([image], export_dir=tmp)


def _load(model_id: str, device: str | None) -> tuple[Any, str]:
    try:
        import torch
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:  # a missing optional dep should say what to install
        raise ImportError(
            "the depth_perception ground truth needs the Depth Anything 3 package: "
            "pip install -r modules/depth_perception/requirements.txt"
        ) from exc

    key = (model_id, device or "auto")
    if key not in _LOADED:
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = DepthAnything3.from_pretrained(model_id)
        model = model.to(device=resolved)
        model.eval()
        _LOADED[key] = (model, resolved)
    return _LOADED[key]


def _to_numpy(value: Any) -> np.ndarray:
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
