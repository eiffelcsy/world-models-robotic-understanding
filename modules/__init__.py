"""Probe modules.

One directory per probe, holding both halves of its convention — what to ask a
world model for, and how to decode what comes back:

    modules/<probe_name>/
        __init__.py     imports this module's files so registration happens
        prompt.yaml     what wmgen asks the world model for
        probe.yaml      readout + ground_truth + extractor + metrics
        gt.py           the model and pipeline producing pseudo ground truth
                        from the input frame
        extractor.py    decodes the world model's answer out of the output frame
        metrics.py      compares the two

The two YAML files are what each pipeline discovers the directory by: wmgen
looks for `prompt.yaml`, wmeval for `probe.yaml`. A directory with only one of
them does only that half, which is how a probe is built up — write the prompt,
look at what comes back, then write the extractor for it.

They belong together because they are one decision. `prompt.yaml` asking for
flat saturated fills and `extractor.py` thresholding saturation are the same
convention written twice; change one alone and the probe reports a model failure
that is really a wiring failure.

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
