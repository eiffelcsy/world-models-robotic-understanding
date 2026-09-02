"""Name -> component registry, shared by both pipelines.

Every pluggable piece is a class registered under a family and a name, and named
from YAML by that name:

    extractor:
      type: my_overlay_decoder
      params: {color: red}

Families are namespaced strings, so the two pipelines cannot collide: wmeval
registers `readout`, `gt`, `extractor` and `metric`; wmgen registers `source`,
`prompt`, `backend` and `check`.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

_REGISTRY: dict[str, dict[str, type]] = {}

T = TypeVar("T", bound=type)


def register(family: str, name: str) -> Callable[[T], T]:
    def decorator(cls: T) -> T:
        table = _REGISTRY.setdefault(family, {})
        if name in table and table[name] is not cls:
            raise ValueError(f"{family}/{name} is already registered by {table[name]!r}")
        table[name] = cls
        cls.registry_name = name  # type: ignore[attr-defined]
        return cls

    return decorator


def get(family: str, name: str) -> type:
    table = _REGISTRY.get(family, {})
    if name not in table:
        known = ", ".join(sorted(table)) or "<none>"
        raise KeyError(f"unknown {family} '{name}'. registered: {known}")
    return table[name]


def build(family: str, spec: dict[str, Any] | str) -> Any:
    """Instantiate from a `{type, params}` spec (or a bare type string)."""
    if isinstance(spec, str):
        spec = {"type": spec}
    if "type" not in spec:
        raise ValueError(f"{family} spec is missing 'type': {spec!r}")
    return get(family, spec["type"])(**(spec.get("params") or {}))


def available(family: str) -> list[str]:
    return sorted(_REGISTRY.get(family, {}))
