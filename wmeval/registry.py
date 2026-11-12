"""The registry lives in `wmcore`, shared with wmgen. Re-exported here so that
`wmeval.registry` keeps working for modules that import it by that name."""

from wmcore.registry import available, build, get, register  # noqa: F401

__all__ = ["available", "build", "get", "register"]
