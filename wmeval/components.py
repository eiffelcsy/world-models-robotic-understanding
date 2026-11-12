from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from wmcore.components import Component
from wmcore.utils import plain as _plain
from wmcore.utils import safe_name as _safe_name
from wmcore.utils import write_image

from .types import ProbeData, RunContext, Sample

log = logging.getLogger("wmeval")


class GroundTruthSource(Component):
    """Input frame -> pseudo ground truth.

    A source that sets `store_name` keeps its answers in a directory of readable
    files, one per sample:

        <ctx.gt_dir>/<store_name>/<sample_id>_gt.npz   the masks a metric scores
        <ctx.gt_dir>/<store_name>/<sample_id>_gt.png   what they look like

    The pipeline loads that file when it is there and only runs the model when it
    is not, because this is normally the expensive stage and it would otherwise
    be re-run every time a metric or an extractor changes. Regenerating is
    deleting the directory, or running the module's preview, which sets
    `ctx.refresh_gt` and recomputes unconditionally.

    Nothing about the params is in the filename, so a stored answer survives a
    threshold change rather than being orphaned by it. `store_params` names the
    params that decide what ends up in the file; a stored answer computed under
    different ones is loaded, but warned about.

    Leave `store_name` as None for a source that just reads a file.
    """

    store_name: str | None = None
    store_params: tuple[str, ...] = ()

    def compute(self, sample: Sample, input_frame: np.ndarray, ctx: RunContext) -> ProbeData:
        raise NotImplementedError

    def render(self, data: ProbeData, input_frame: np.ndarray) -> np.ndarray | None:
        """Optional RGB view of `data`, written beside the stored file. None = no view."""
        return None

    def store_path(self, sample: Sample, ctx: RunContext) -> Path | None:
        """Where this sample's ground truth lives, or None when it is not stored."""
        if self.store_name is None or ctx.gt_dir is None:
            return None
        return Path(ctx.gt_dir) / self.store_name / f"{_safe_name(sample.sample_id)}_gt.npz"

    def __call__(self, sample: Sample, input_frame: np.ndarray, ctx: RunContext) -> ProbeData:
        path = self.store_path(sample, ctx)
        if path is not None and path.exists() and not ctx.refresh_gt:
            try:
                data = load_probe_data(path)
            except Exception:  # noqa: BLE001 - a damaged file must not be fatal
                log.warning("unreadable ground truth %s; recomputing", path)
            else:
                self._warn_on_drift(data, path)
                return data

        data = self.compute(sample, input_frame, ctx)
        data.info.setdefault("params", _plain(self.params))
        if path is not None:
            save_probe_data(path, data)
            view = self.render(data, input_frame)
            if view is not None:
                write_image(path.with_suffix(".png"), view)
        return data

    def _warn_on_drift(self, data: ProbeData, path: Path) -> None:
        """Say so when a stored answer predates the params now configured."""
        stored = data.info.get("params")
        if not stored or not self.store_params:
            return
        current = _plain(self.params)
        changed = [k for k in self.store_params if k in stored and stored[k] != current.get(k)]
        if changed:
            log.warning(
                "%s was computed with a different %s; delete it or re-run the module preview "
                "to regenerate", path, ", ".join(changed)
            )


class Extractor(Component):
    """Output frame pixels -> probe data.

    The world model can only answer in pixels, so every answer arrives as an
    appearance change. An extractor is the decoder for one such convention.
    """

    def __call__(self, frame: np.ndarray, sample: Sample, ctx: RunContext) -> ProbeData:
        raise NotImplementedError


class Metric(Component):
    """(prediction, ground truth) -> named scores.

    `accepts` lists the `ProbeData` kinds this metric can score; the probe checks
    it before calling, so a mis-wired YAML raises a type error instead of a
    number that means nothing.
    """

    accepts: tuple[str, ...] = ()

    def __call__(self, prediction: ProbeData, truth: ProbeData) -> dict[str, float]:
        raise NotImplementedError


def save_probe_data(path: Path, data: ProbeData) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for key, value in data.payload.items():
        if isinstance(value, list):
            arrays[key] = np.stack(value) if len(value) else np.zeros((0, 1, 1))
        else:
            arrays[key] = np.asarray(value)
    np.savez_compressed(
        path,
        __kind__=np.array(data.kind),
        __info__=np.array(json.dumps(data.info, default=str)),
        __lists__=np.array(json.dumps([k for k, v in data.payload.items() if isinstance(v, list)])),
        **arrays,
    )


def load_probe_data(path: Path) -> ProbeData:
    with np.load(path, allow_pickle=False) as blob:
        kind = str(blob["__kind__"])
        info = json.loads(str(blob["__info__"]))
        as_list = set(json.loads(str(blob["__lists__"])))
        payload = {k: (list(blob[k]) if k in as_list else blob[k]) for k in blob.files if not k.startswith("__")}
    return ProbeData(kind, payload, info)


