"""Probe modules.

One directory per probe. A directory is picked up as soon as it contains a
`probe.yaml`, and ignored as soon as it does not:

    modules/<probe_name>/
        __init__.py     imports this module's files so registration happens
        probe.yaml      readout + ground_truth + extractor + metrics
        gt.py           the model and pipeline producing pseudo ground truth
                        from the input frame
        extractor.py    decodes the world model's answer out of the output frame
        metrics.py      compares the two

Split across as many files as the module needs, or keep it in one. The only
requirement is that importing the package registers every component named in
`probe.yaml`, so `__init__.py` should import them:

    from . import gt, extractor, metrics   # noqa: F401

Register with the `wmeval.register` decorator, under the family the component
belongs to:

    from wmeval import GroundTruthSource, ProbeData, register

    @register("gt", "my_depth_model")
    class MyDepthModel(GroundTruthSource):
        def __init__(self, checkpoint: str, device: str = "cpu"):
            super().__init__(checkpoint=checkpoint, device=device)

        def compute(self, sample, input_frame, ctx) -> ProbeData:
            depth = ...                      # run the model on input_frame
            return ProbeData("depth", {"depth": depth})

Families are `readout`, `gt`, `extractor` and `metric`. A module's heavy imports
(torch, a checkpoint) stay inside it and are only loaded when that probe runs.
"""
