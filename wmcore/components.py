"""The base every registered component shares."""

from __future__ import annotations


class Component:
    """Constructed from a YAML `params` block; every param becomes an attribute.

    A param may not be named after one of the class's own methods. Splatting
    YAML onto attributes is what makes a component's config readable, but it
    also means a param called `submit` would quietly replace a `submit()` method
    with a dict, and the failure would surface much later as something calling a
    dict. A component wanting such a name keeps the value elsewhere and leaves
    the method alone.
    """

    def __init__(self, **params) -> None:
        shadowed = [key for key in params if callable(getattr(type(self), key, None))]
        if shadowed:
            raise TypeError(
                f"{type(self).__name__}: param(s) {shadowed} would shadow a method of the same name"
            )
        self.params = params
        for key, value in params.items():
            setattr(self, key, value)
