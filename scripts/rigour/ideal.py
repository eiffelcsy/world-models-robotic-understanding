"""The answer each prompt asks for, drawn from the ground truth onto the conditioning frame.

Used for the decoder ceilings (references.py) and the through-video ceilings
(ceiling.py). Every function returns an RGB uint8 image the size of the frame.
"""
from __future__ import annotations

import colorsys

import cv2
import numpy as np

import common as c

RED, GREEN = (230, 25, 75), (60, 180, 75)  # the pose decoder's palette, #e6194b and #3cb44b


def vivid(k: int) -> tuple[int, int, int]:
    return tuple(round(255 * v) for v in colorsys.hsv_to_rgb((0.83 + k * 0.31) % 1.0, 1.0, 1.0))


def seg(sample_id: str, same_colour: bool = False) -> np.ndarray:
    """Each true mask in its own flat saturated colour; the rest untouched."""
    out = c.read_image(c.frame_path(sample_id)).copy()
    for k, mask in enumerate(c.truth("instance_seg", sample_id)["masks"]):
        out[np.asarray(mask, bool)] = vivid(0 if same_colour else k)
    return out


def depth(sample_id: str, encoding: str = "disparity") -> np.ndarray:
    """Grayscale depth, near bright; linear in depth or in inverse depth. Bars stay black."""
    d = np.asarray(c.truth("depth_perception", sample_id)["depth"], np.float64)
    ok = np.isfinite(d)
    v = 1.0 / d[ok] if encoding == "disparity" else -d[ok]
    gray = np.zeros(d.shape)
    gray[ok] = (v - v.min()) / (v.max() - v.min())
    return np.repeat((255 * gray).round().astype(np.uint8)[..., None], 3, axis=2)


def dots(sample_id: str, points: np.ndarray, colours, radius: int) -> np.ndarray:
    out = c.read_image(c.frame_path(sample_id)).copy()
    for (x, y), colour in zip(np.asarray(points, float).reshape(-1, 2), colours):
        cv2.circle(out, (int(round(x)), int(round(y))), radius, colour, -1, lineType=cv2.LINE_AA)
    return out


def keypoints(sample_id: str, radius: int = 4) -> np.ndarray:
    points = c.truth("robot_keypoint_detection", sample_id)["points"]
    return dots(sample_id, points, [vivid(k) for k in range(len(points))], radius)


def pose(sample_id: str, radius: int = 4) -> np.ndarray:
    return dots(sample_id, c.truth("robot_pose_estimation", sample_id)["points"], [RED, GREEN], radius)


def grasp_mask(sample_id: str, kind: str = "grasp", radius_m: float = 0.01) -> np.ndarray:
    """A grasp answer as a mask: a dot on the demonstrated grasp, or a guess that knows only the object."""
    gt = c.truth("grasp_region_seg", sample_id)
    obj = np.asarray(gt["mask"], bool)
    radius = max(2, int(round(radius_m * float(gt["pixels_per_metre"]))))
    disc = np.zeros(obj.shape, np.uint8)
    if kind == "grasp":
        x, y = np.asarray(gt["grasp"])[0]
        cv2.circle(disc, (int(round(x)), int(round(y))), radius, 1, -1)
        return disc > 0
    if kind == "object":
        return obj
    rows, cols = np.nonzero(obj)
    if kind == "centroid":
        cv2.circle(disc, (int(round(cols.mean())), int(round(rows.mean()))), radius, 1, -1)
        return disc > 0
    if kind == "gripper":  # the object pixel nearest the gripper's grip point in the image
        grip = np.asarray(c.truth("robot_pose_estimation", sample_id)["points"], float)[0]
        k = int(np.argmin(np.hypot(cols - grip[0], rows - grip[1])))
        cv2.circle(disc, (int(cols[k]), int(rows[k])), radius, 1, -1)
        return disc > 0
    if kind == "top":  # the top quarter of the object's rows: a rim/lid guess
        return obj & (np.arange(obj.shape[0])[:, None] <= np.percentile(rows, 25))
    raise ValueError(kind)


def grasp(sample_id: str, kind: str = "grasp", radius_m: float = 0.01) -> np.ndarray:
    out = c.read_image(c.frame_path(sample_id)).copy()
    out[grasp_mask(sample_id, kind, radius_m)] = vivid(0)
    return out
