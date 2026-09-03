"""Instruction -> the objects a task involves. Once per frame, and cached.

A probe that scopes itself to a task ("mask the objects this task involves")
needs to know what those objects are, and the only place that is written down is
the instruction: free text, one sentence, no fixed vocabulary. This stage turns
it into a small strict record, which the prompt template then reads as
`{relevant_objects}`.

Text only: the extractor sees the instruction and never the image, so the schema
holds only what an instruction can actually answer. `phrase` is a bare noun
phrase — templates supply the article — because it is also handed to an
open-vocabulary segmenter verbatim, as `meta.gt_prompt`. That second consumer is
the point of doing this here rather than ad hoc: wmgen asks the world model to
paint the marker and the mug, and wmeval's SAM 3 ground truth is built from the
same two words. A probe that scopes its prompt but not its ground truth scores
the model against objects it was told to leave alone.

    <semantics_dir>/<sample_id>.json      the record, one file per sample

A sample that has a file is never re-requested, the same contract the clip store
and the ground-truth store keep. Regenerating is deleting the file.

Ported from the veo/ pipeline's `semantics` stage, minus the parts this repo
already has: probe-level slot gating (superseded by `requires:` in prompt.yaml),
its templating blocks (superseded by `wmgen.prompt.Template`), and the `openai`
SDK — one chat completion is a POST, so this talks to any OpenAI-compatible
endpoint over stdlib http and adds no dependency.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from wmcore.registry import register
from wmcore.utils import dump_json, load_json, safe_name

from .components import SlotSource, SlotsUnavailable
from .types import Frame, GenContext

log = logging.getLogger("wmgen")

ACTIONS = (
    "pick_place", "articulate", "press", "turn", "pour", "deform", "wipe", "multi_step", "unclear",
)

SYSTEM_PROMPT = """\
You convert a single robot manipulation instruction into a strict JSON object \
describing the objects it involves. You never see the scene, so describe only \
what the instruction itself states.

Return exactly this shape and nothing else:

{
  "action": one of ["pick_place","articulate","press","turn","pour","deform","wipe","multi_step","unclear"],
  "target": {"phrase": str, "plural": bool},
  "destination": {"phrase": str, "plural": bool} or null,
  "confidence": float between 0 and 1
}

Rules for "phrase":
- A bare noun phrase naming a physical object: lowercase, no article, no verb, \
no location clause. "blue marker", not "the blue marker on the table".
- One to three words. Keep a colour or material only when the instruction uses \
it to pick the object out.
- It is passed straight to an open-vocabulary image segmenter, so it must name \
something with a visible outline. Never a region, a side, or an action.
- Name what the instruction says even when it sounds impossible or absent from \
a plausible scene. You are recording the instruction, not judging it.

"target" is the object the robot manipulates and is always present. \
"destination" is where the target goes or what it is placed into or onto, and \
is null when the instruction names none. When the instruction is uninformative \
(for example "task" or an empty description), use action "unclear" and the best \
single noun phrase you can, with a low confidence."""

FEW_SHOT = [
    ("put the marker in the cup",
     {"action": "pick_place",
      "target": {"phrase": "marker", "plural": False},
      "destination": {"phrase": "cup", "plural": False},
      "confidence": 0.95}),
    ("open the top drawer",
     {"action": "articulate",
      "target": {"phrase": "drawer", "plural": False},
      "destination": None,
      "confidence": 0.9}),
    ("fold the towel and place it on the shelf",
     {"action": "multi_step",
      "target": {"phrase": "towel", "plural": False},
      "destination": {"phrase": "shelf", "plural": False},
      "confidence": 0.85}),
]


# --------------------------------------------------------------------------- #
# the components
# --------------------------------------------------------------------------- #

@register("semantics", "llm_slots")
class LlmSlots(SlotSource):
    """One chat completion per instruction, against an OpenAI-compatible endpoint.

    Params:

      model             the endpoint's model id
      base_url          the endpoint, or leave null and set `base_url_env`
      base_url_env      environment variable holding it (default LLM_BASE_URL)
      api_key_env       environment variable holding the key (default LLM_API_KEY)
      temperature       0 by default: this is extraction, not writing
      max_retries       attempts per call, with exponential backoff
      timeout_s         per-request timeout
      instruction_key   which `frame.meta` key holds the instruction text
      extra_gt_phrases  appended to `gt_prompt` only — see `SlotSource`
      overrides         JSON file of hand corrections, {sample_id: {patch}}

    The key is never a literal in config; it is read from the environment, so no
    deployment detail lands in git.
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        base_url: str | None = None,
        base_url_env: str = "LLM_BASE_URL",
        api_key_env: str = "LLM_API_KEY",
        temperature: float = 0.0,
        max_retries: int = 3,
        timeout_s: float = 60.0,
        instruction_key: str = "task",
        extra_gt_phrases: list[str] | None = None,
        overrides: str | None = None,
    ) -> None:
        super().__init__(
            model=model,
            base_url=base_url,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            temperature=temperature,
            max_retries=max_retries,
            timeout_s=timeout_s,
            instruction_key=instruction_key,
            extra_gt_phrases=list(extra_gt_phrases or []),
            overrides=overrides,
        )
        self._json_mode = True

    def preflight(self) -> None:
        """Resolve credentials once, before any frame is processed.

        A missing key is the run being misconfigured, not a per-sample failure to
        be retried three times over a thousand samples.
        """
        self._endpoint()

    def compute(self, frame: Frame, ctx: GenContext) -> dict[str, Any]:
        instruction = str(frame.meta.get(self.instruction_key) or "").strip()
        if not instruction:
            raise SlotsUnavailable(
                f"no instruction: frame.meta has no '{self.instruction_key}'. "
                f"Set it in the source's meta_file, or point `instruction_key` at the key that has it"
            )

        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for text, answer in FEW_SHOT:
            messages.append({"role": "user", "content": text})
            messages.append({"role": "assistant", "content": json.dumps(answer)})
        messages.append({"role": "user", "content": instruction})

        response = self._chat(messages)
        try:
            record = parse(response)
        except ValueError as exc:
            # A plain retry at temperature 0 replays the same failure. Hand the
            # model its own bad answer instead: different input, different output.
            log.info("  unparseable slots (%s); asking the model to correct it", exc)
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content":
                    f'That response was rejected: {exc}. Reply with only the JSON object, '
                    f'where "target" is an object with a non-empty "phrase" naming the object '
                    f'the robot manipulates.'},
            ]
            record = parse(self._chat(messages))

        record["instruction"] = instruction
        record["model"] = self.model
        return record

    # -- the endpoint ------------------------------------------------------- #

    def _endpoint(self) -> tuple[str, str]:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        base = self.base_url or os.environ.get(self.base_url_env or "")
        if not base:
            raise RuntimeError(f"no endpoint: set {self.base_url_env} or semantics.params.base_url")
        return base.rstrip("/"), key

    def _chat(self, messages: list[dict[str, str]]) -> str:
        base, key = self._endpoint()
        attempts = max(1, int(self.max_retries))
        last: Exception | None = None

        for attempt in range(attempts):
            body: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": float(self.temperature),
            }
            if self._json_mode:
                body["response_format"] = {"type": "json_object"}

            request = urllib.request.Request(
                f"{base}/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=float(self.timeout_s)) as response:
                    payload = json.loads(response.read())
                return payload["choices"][0]["message"]["content"] or ""
            except Exception as exc:  # noqa: BLE001 - endpoints differ in what they raise
                last = exc
                detail = _detail(exc)
                if self._json_mode and "response_format" in detail:
                    log.info("  endpoint rejects json mode; falling back to plain completions")
                    self._json_mode = False
                    continue
                if attempt < attempts - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(f"slot extraction failed after {attempts} attempts: {_detail(last)}")


@register("semantics", "cached")
class CachedSlots(SlotSource):
    """The store and the overrides file, and nothing else — no endpoint, no cost.

    For a rerun that must not call out, and for slots written by hand: put a
    record in `<semantics_dir>/<sample_id>.json`, or a patch in `overrides`, and
    a sample with neither is reported rather than invented.
    """

    def __init__(self, extra_gt_phrases: list[str] | None = None, overrides: str | None = None) -> None:
        super().__init__(extra_gt_phrases=list(extra_gt_phrases or []), overrides=overrides)

    def compute(self, frame: Frame, ctx: GenContext) -> dict[str, Any]:
        raise SlotsUnavailable("no stored or overridden slots (semantics type is 'cached')")


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #

def parse(response: str) -> dict[str, Any]:
    """A model's reply -> the record, or ValueError naming what was wrong."""
    text = (response or "").strip()
    if not text:
        raise ValueError("empty response")

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"no JSON object in response: {text[:200]!r}") from None
        data = json.loads(match.group(0))

    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object, got {type(data).__name__}")

    action = data.get("action", "unclear")
    record: dict[str, Any] = {
        "action": action if action in ACTIONS else "unclear",
        "target": _slot(data.get("target")),
        "destination": _slot(data.get("destination")),
        "confidence": _confidence(data.get("confidence")),
    }
    if record["target"] is None:
        raise ValueError("no target slot in response")
    return record


def _slot(value: Any) -> dict[str, Any] | None:
    """Tolerant of the shapes models reach for instead of the asked-for object:
    a bare string, a one-item list, or `name`/`object` in place of `phrase`."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, str):
        value = {"phrase": value}
    if not isinstance(value, dict):
        return None

    phrase = str(value.get("phrase") or value.get("name") or value.get("object") or "").strip().lower()
    phrase = re.sub(r"^(the|a|an)\s+", "", phrase)
    if not phrase:
        return None
    return {"phrase": phrase, "plural": bool(value.get("plural", False))}


def _confidence(value: Any) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None


def _detail(exc: Exception | None) -> str:
    """An HTTPError's body says why; its str() only says the status code."""
    body = ""
    if hasattr(exc, "read"):
        try:
            body = exc.read().decode("utf-8", "replace")[:300]  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            body = ""
    return f"{exc}{': ' + body if body else ''}"


# --------------------------------------------------------------------------- #
# the stage
# --------------------------------------------------------------------------- #

def object_phrases(record: dict[str, Any]) -> list[str]:
    """The objects a prompt names, in template order: target then destination."""
    phrases: list[str] = []
    for name in ("target", "destination"):
        slot = record.get(name)
        if isinstance(slot, dict) and slot.get("phrase") and slot["phrase"] not in phrases:
            phrases.append(slot["phrase"])
    return phrases


def phrase_list(phrases: list[str]) -> str:
    """`["marker", "mug"]` -> `"the marker and the mug"`.

    The template supplies nothing around this, so the articles are added here —
    the stored phrase stays bare because the segmenter wants it that way.
    """
    named = [f"the {phrase}" for phrase in phrases]
    if len(named) <= 1:
        return named[0] if named else ""
    return f"{', '.join(named[:-1])} and {named[-1]}"


def describe(record: dict[str, Any]) -> str:
    parts = [str(record.get("action"))]
    for name in ("target", "destination"):
        slot = record.get(name)
        if isinstance(slot, dict):
            parts.append(f"{name}={slot['phrase']}")
    return " ".join(parts)


def store_path(semantics_dir: Path | None, sample_id: str) -> Path | None:
    return None if semantics_dir is None else Path(semantics_dir) / f"{safe_name(sample_id)}.json"


def load_overrides(path: str | Path | None) -> dict[str, dict[str, Any]]:
    """Hand corrections, keyed by sample id. A missing file is not an error.

    Applied on read and never written back into the store, so deleting an entry
    here restores what the extractor said rather than leaving it baked in.
    """
    if not path or not Path(path).exists():
        return {}
    data = load_json(path)
    return data if isinstance(data, dict) else {}


def apply(
    frames: list[Frame],
    spec: dict[str, Any] | str | None,
    ctx: GenContext,
) -> list[dict[str, str]]:
    """Fill every frame's meta with the objects its instruction names.

    Mutates `frame.meta` in place — it is the same dict the prompt template reads
    and `build_manifest` writes out, so one assignment reaches both pipelines.
    Returns the frames it could not do, which the caller reports; the frame stays
    in the run and fails at prompt-render time on `requires:`, before any backend
    is called and so before any money is spent.
    """
    if not spec:
        return []

    from wmcore import registry  # local: keeps import order flat

    source: SlotSource = registry.build("semantics", spec)
    overrides = load_overrides(getattr(source, "overrides", None))

    def stored(frame: Frame) -> bool:
        path = store_path(ctx.semantics_dir, frame.sample_id)
        return path is not None and path.exists()

    if any(
        not stored(frame) and frame.sample_id not in overrides
        for frame in frames
        if not source.satisfied(frame)
    ):
        source.preflight()

    failures: list[dict[str, str]] = []
    computed = 0
    for frame in frames:
        if source.satisfied(frame):
            continue
        supplied = [key for key in source.provides if frame.meta.get(key)]
        if supplied:
            log.warning(
                "  %s already has %s but not %s; resolving the whole record and overwriting, "
                "since a prompt and a ground truth scoped differently score nothing useful",
                frame.sample_id,
                ", ".join(supplied),
                ", ".join(k for k in source.provides if k not in supplied),
            )
        try:
            record = source.resolve(frame, ctx, overrides)
        except Exception as exc:  # noqa: BLE001 - one instruction must not end the run
            log.warning("  slots failed for %-40s %s", frame.sample_id, exc)
            failures.append({"sample_id": frame.sample_id, "error": f"{type(exc).__name__}: {exc}"})
            continue
        frame.meta.update(source.to_meta(record))
        computed += 1
        log.info("  slots %-40s %s", frame.sample_id, describe(record))

    if computed:
        log.info("%d slot record(s) resolved%s", computed, f", {len(failures)} failed" if failures else "")
    return failures
