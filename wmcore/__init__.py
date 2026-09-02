"""wmcore - the mechanisms both halves of the repo are built from.

`wmgen` generates videos from conditioning frames; `wmeval` scores them. Neither
imports the other, and both extend the same way: a registry of named components,
plugin directories discovered by a marker file, YAML whose paths resolve against
a root. Those mechanisms live here.

Nothing in this package knows what a probe or a world model is.
"""

from .components import Component
from .config import read_yaml, resolver, root_of
from .modules import discover, import_module
from .registry import available, build, get, register

__all__ = [
    "Component",
    "available",
    "build",
    "discover",
    "get",
    "import_module",
    "read_yaml",
    "register",
    "resolver",
    "root_of",
]
