"""Item 3, slow part: run the specialist models on the LIBERO conditioning frames.

    venv/bin/python scripts/rigour/specialists.py [device]

SAM 3 (twice: with each object's noun, as RoboTrustBench's pseudo ground truth is
made, and with the full referring phrase the world model is given) and Depth
Anything 3, with the settings in each probe.yaml. Outputs are cached in
results/rigour/specialists/ and read by references.py. No ground-truth store is
touched (the run context has no gt_dir).
"""
from __future__ import annotations

import sys

import numpy as np

import common as c

OUT = c.OUT / "specialists"
OUT.mkdir(exist_ok=True)
DEVICE = sys.argv[1] if len(sys.argv) > 1 else None


def phrases(sample_id: str) -> list[str]:
    """The referring phrase for each noun in `gt_prompt`, cut from `relevant_objects`.

    "the black bowl on the wooden cabinet and the plate" with nouns [black bowl,
    plate] -> ["black bowl on the wooden cabinet", "plate"]. Cut from the right, so
    a noun that also appears inside an earlier phrase is not split there.
    """
    meta = c.META[sample_id]
    nouns, rest, out = list(meta["gt_prompt"]), meta["relevant_objects"], []
    for noun in reversed(nouns[1:]):
        at = rest.rfind(noun)
        if at < 0:
            return nouns
        out.append(rest[at:].strip())
        rest = rest[:at].rstrip()
        for tail in (" and the", ", the", " the", " and", ","):
            if rest.endswith(tail):
                rest = rest[: -len(tail)].rstrip()
                break
    first = rest[4:] if rest.startswith("the ") else rest
    return [first.strip()] + out[::-1]


def run_sam3() -> None:
    from modules.instance_seg.gt import Sam3Instances

    params = dict(c.spec("instance_seg")["ground_truth"]["params"])
    params.update(min_instances=0, device=DEVICE)
    source = Sam3Instances(**params)
    for sample_id in c.ids("all"):
        frame = c.read_image(c.frame_path(sample_id))
        for name, prompts in (("noun", list(c.META[sample_id]["gt_prompt"])), ("phrase", phrases(sample_id))):
            path = OUT / f"sam3_{name}__{sample_id}.npz"
            if path.exists():
                continue
            s = c.sample(sample_id)
            s.meta = {**s.meta, "gt_prompt": prompts}
            data = source.compute(s, frame, c.CTX)
            masks = data["masks"]
            np.savez_compressed(path, masks=np.stack(masks) if masks else np.zeros((0,) + frame.shape[:2], bool),
                                prompts=np.array(prompts), labels=np.array(data.info["labels"]))
            print(f"sam3 {name:6s} {sample_id:32s} {prompts} -> {len(masks)} masks", flush=True)


def run_da3() -> None:
    from modules.depth_perception.gt import DepthAnything3Depth

    params = dict(c.spec("depth_perception")["ground_truth"]["params"])
    params.update(device=DEVICE)
    source = DepthAnything3Depth(**params)
    for sample_id in c.ids("all"):
        path = OUT / f"da3__{sample_id}.npz"
        if path.exists():
            continue
        data = source.compute(c.sample(sample_id), c.read_image(c.frame_path(sample_id)), c.CTX)
        np.savez_compressed(path, depth=np.asarray(data["depth"], np.float32))
        print(f"da3 {sample_id}", flush=True)


if __name__ == "__main__":
    for sample_id in c.ids("all"):
        if phrases(sample_id) != list(c.META[sample_id]["gt_prompt"]):
            print(f"phrase differs from noun: {sample_id}: {phrases(sample_id)}")
    run_da3()
    run_sam3()
