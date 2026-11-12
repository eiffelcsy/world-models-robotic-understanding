"""Robot pose estimation probe (LIBERO only).

Asks the world model to paint the robot's end-effector and a second marker along
its pointing direction, and scores the decoded position and heading against the
end-effector pose the simulator recorded.

    gt.py         recorded end-effector pose -> a center + heading point
    extractor.py  two colored markers -> a center and a heading point
    metrics.py    prediction vs ground truth: position and orientation error
    scan.py       CLI: best frame vs last frame of one generation

Importing the package registers the components `probe.yaml` names.
"""

from . import extractor, gt, metrics  # noqa: F401
