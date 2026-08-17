"""wmeval - probe-based evaluation of video world models for robotic understanding.

The pipeline is model-agnostic: it consumes generated videos plus the frames they
were conditioned on, and knows nothing about what produced them. Probes live in
`modules/`.
"""

from .components import Extractor, GroundTruthSource, Metric
from .config import EvalConfig, ProbeConfig, load_eval_config, load_probe_config
from .probe import Probe
from .registry import register
from .runner import run, run_from_file
from .types import ProbeData, Result, RunContext, Sample

__version__ = "0.1.0"

__all__ = [
    "EvalConfig",
    "Extractor",
    "GroundTruthSource",
    "Metric",
    "Probe",
    "ProbeConfig",
    "ProbeData",
    "Result",
    "RunContext",
    "Sample",
    "load_eval_config",
    "load_probe_config",
    "register",
    "run",
    "run_from_file",
]
