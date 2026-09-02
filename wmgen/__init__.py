"""wmgen - generating the videos wmeval scores.

The step before the evaluation pipeline: conditioning frames plus a per-probe
prompt in, generated clips plus a manifest out. It is model-agnostic in the way
its half of the problem allows — wmeval never calls a generator, so a world
model is invisible to it; wmgen must call one, so a world model is a
`backends/<name>/backend.yaml` file rather than code.

A probe's prompt lives beside its `probe.yaml`, in `modules/<probe>/prompt.yaml`,
because what a probe asks for and what its extractor decodes are one convention
and have to change together.
"""

from wmcore.registry import register

from .components import ClipCheck, ClipUnavailable, FrameSource, PromptBuilder, VideoBackend
from .config import BackendConfig, GenConfig, PromptConfig, load_gen_config, load_prompt_config
from .runner import run, run_from_file
from .task import Task
from .types import Clip, Frame, GenContext, Prompt, Request

__version__ = "0.1.0"

__all__ = [
    "BackendConfig",
    "Clip",
    "ClipCheck",
    "ClipUnavailable",
    "Frame",
    "FrameSource",
    "GenConfig",
    "GenContext",
    "Prompt",
    "PromptBuilder",
    "PromptConfig",
    "Request",
    "Task",
    "VideoBackend",
    "load_gen_config",
    "load_prompt_config",
    "register",
    "run",
    "run_from_file",
]
