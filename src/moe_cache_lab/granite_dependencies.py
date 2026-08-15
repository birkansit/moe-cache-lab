"""Narrow optional-dependency boundary for the built-in Granite collector."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


DEFAULT_MODEL = "ibm-granite/granite-3.1-1b-a400m-instruct"
GRANITE_EXTRA_INSTALL = "moe-cache-lab[granite]"
GRANITE_DEPENDENCY_MESSAGE = (
    "Granite collection requires optional Torch and Transformers dependencies; "
    f"install them with `{GRANITE_EXTRA_INSTALL}`."
)


class GraniteDependencyError(RuntimeError):
    """Raised when a Granite-only operation lacks its optional dependencies."""


def load_granite_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Import the pinned Granite stack only when collection is requested."""
    try:
        torch = import_module("torch")
        transformers = import_module("transformers")
        collector = import_module("moe_cache_lab.collector")
    except ModuleNotFoundError as error:
        if error.name in {"torch", "transformers"}:
            raise GraniteDependencyError(GRANITE_DEPENDENCY_MESSAGE) from None
        raise
    return torch, transformers, collector
