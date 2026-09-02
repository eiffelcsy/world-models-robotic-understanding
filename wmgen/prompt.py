"""Prompt builders: a probe's `prompt.yaml` plus one frame -> the text to send.

A world model can only answer in pixels, so a probe asks for an appearance
change and its extractor decodes that convention back out. This module is the
asking half; `modules/<probe>/extractor.py` is the decoding half. They are
written together and live in the same directory, because a convention changed in
one and not the other produces a score of zero that reads exactly like a model
failure.
"""

from __future__ import annotations

import logging
from string import Formatter
from typing import Any

from wmcore.registry import register

from .components import PromptBuilder
from .types import Frame, Prompt

log = logging.getLogger("wmgen")

_FORMATTER = Formatter()


def placeholders(text: str) -> set[str]:
    """The `{name}` fields a template expects. Literal braces are doubled."""
    return {name.split(".")[0].split("[")[0] for _, name, _, _ in _FORMATTER.parse(text) if name}


@register("prompt", "template")
class Template(PromptBuilder):
    """`{placeholder}` substitution from the frame's metadata.

    Fields come from `frame.meta`, plus the built-ins `sample_id` and every
    video param (`duration_s`, `aspect_ratio`, `fps`, `resolution`, `seed`), so
    a prompt can name its own duration without repeating the number.

    A placeholder with nothing to fill it raises. A prompt is the entire
    instruction to the model, and a hole in it — "over the next  seconds" — is a
    silently degraded generation that costs the same as a good one, so this is
    the one thing in the pipeline that must fail loudly and early. `--dry-run`
    is where it is meant to fail.

    Literal braces in prose must be doubled, `{{` and `}}`, as in `str.format`.
    """

    def __init__(self, text: str, negative: str = "", requires: list[str] | None = None) -> None:
        super().__init__(text=text, negative=negative, requires=list(requires or []))

    def __call__(self, frame: Frame, video: dict[str, Any]) -> Prompt:
        fields: dict[str, Any] = {"sample_id": frame.sample_id, **video, **frame.meta}

        missing = [key for key in self.requires if key not in frame.meta]
        if missing:
            raise KeyError(
                f"{frame.sample_id}: prompt requires meta {missing}, which this frame does not have. "
                f"Add it in the source's meta_file, or drop it from `requires` in prompt.yaml"
            )

        return Prompt(
            text=_render(self.text, fields, frame),
            negative=_render(self.negative, fields, frame) if self.negative else "",
            video=dict(video),
        )


def _render(text: str, fields: dict[str, Any], frame: Frame) -> str:
    unfilled = sorted(placeholders(text) - set(fields))
    if unfilled:
        raise KeyError(
            f"{frame.sample_id}: prompt template has no value for {unfilled}. "
            f"available: {', '.join(sorted(fields)) or '<none>'}"
        )
    return text.format_map(fields).strip()
