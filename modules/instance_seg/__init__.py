"""Instance segmentation probe.

Asks the world model to repaint the conditioning frame so that every object
carries its own flat color, and scores the instances that come back against
SAM 3 run on the input frame.

    gt.py         SAM 3 -> pseudo ground truth instance masks
    sam3.py       the checkpoint and the one call into transformers
    extractor.py  flat colors in the generated frame -> instance masks, plus
                  the scan that scores every frame of a generation
    metrics.py    prediction vs ground truth: one-to-one IoU matching
    preview.py    CLI: look at the ground truth on a frame
    scan.py       CLI: best frame vs last frame of one generation

Importing the package registers the components `probe.yaml` names; heavy imports
stay inside `sam3.py` and only happen when this probe actually runs. The two CLI
modules are deliberately not imported here — a module cannot be both imported at
registration time and run with `-m` without registering twice.
"""

from . import extractor, gt, metrics  # noqa: F401
