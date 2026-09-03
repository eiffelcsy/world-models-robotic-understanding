"""Image and array IO shared by the pipelines."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, TypeVar

import cv2
import numpy as np

T = TypeVar("T")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image(path: str | Path) -> np.ndarray:
    """Read an image as RGB uint8."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def dump_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=_json_default)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")


def plain(params: dict[str, Any]) -> dict[str, Any]:
    """Params as they survive a trip through a file, so the two compare equal.

    Both stores compare the params a stored artefact was made with against the
    params configured now, and only a round-tripped copy compares equal.
    """
    return json.loads(json.dumps(params, sort_keys=True, default=str))


def safe_name(sample_id: str) -> str:
    """A sample id as a filename.

    Shared so that a sample's ground truth and its generated clip are named the
    same way and can be found from each other by eye. Ids come from frame stems,
    so this rarely bites.
    """
    return "".join(c if c.isalnum() or c in "-_.+=" else "_" for c in sample_id)


def stratified_sample(items: list[T], limit: int, key: Callable[[T], Any], seed: int = 0) -> list[T]:
    """`limit` items, drawn proportionally from the distribution of `key(item)`.

    A plain `items[:limit]` takes whatever a source happens to list first — for
    `frames_dir`, that is sorted-glob order, i.e. alphabetical by sample id. On a
    dataset with an uneven category mix (RoboTrustBench: 533 A_Normal down to
    101 D_Adversarial) that silently biases a smoke run: a `limit: 50` could
    easily draw zero adversarial samples and never notice. This keeps each
    group's share of `limit` close to its share of `items`, via largest-remainder
    apportionment — the same method used to divide parliamentary seats by vote
    share, because it is the same problem.

    Groups smaller than their exact quota are not padded to it and groups larger
    than `limit` are not capped below it beyond their fair share; a category
    absent from `items` is simply absent from the result, not an error. Which
    items are picked within a group is a `seed`-deterministic shuffle, not
    always the alphabetically-first ones, so a fixed `limit` does not always
    exercise the same slice of a group across runs while still being
    reproducible run to run. Output order follows `items`, not group order.

    `key` returning `None` groups those items together rather than dropping
    them — an unlabelled item still counts.
    """
    if limit is None or limit >= len(items):
        return list(items)
    if limit <= 0:
        return []

    groups: dict[Any, list[T]] = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    total = len(items)
    quotas = {k: len(g) * limit / total for k, g in groups.items()}
    counts = {k: int(quotas[k]) for k in groups}  # floor: never exceeds len(group), see below

    # Largest-remainder apportionment: hand out what floor() left unassigned to
    # whichever groups were closest to rounding up, largest fraction first, tied
    # groups broken by key so the result is deterministic. Safe to add 1 without
    # a bound check — floor(quota) < len(group) is guaranteed whenever
    # limit < total (quota = len(group) * limit/total, and limit/total < 1).
    remainder = limit - sum(counts.values())
    order = sorted(groups, key=lambda k: (-(quotas[k] - counts[k]), str(k)))
    for k in order[:remainder]:
        counts[k] += 1

    selected = {id(item) for k, g in groups.items() for item in g[: counts[k]]}
    return [item for item in items if id(item) in selected]


def resize_to(array: np.ndarray, shape: tuple[int, int], *, nearest: bool = False) -> np.ndarray:
    """Resize (H, W[, C]) to `shape` = (H, W). Nearest for labels and masks."""
    if array.shape[:2] == tuple(shape):
        return array
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    source = array
    cast_back = None
    if source.dtype == bool:
        source, cast_back = source.astype(np.uint8), bool
    resized = cv2.resize(source, (shape[1], shape[0]), interpolation=interp)
    return resized.astype(cast_back) if cast_back else resized
