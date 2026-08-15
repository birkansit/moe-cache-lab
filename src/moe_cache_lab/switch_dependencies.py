"""Narrow optional-dependency boundary for the built-in Switch collector."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


DEFAULT_SWITCH_MODEL = "google/switch-base-8"
SWITCH_MODEL_REVISION = "92fe2d22b024d9937146fe097ba3d3a7ba146e1b"
SWITCH_EXTRA_INSTALL = "moe-cache-lab[switch]"
SWITCH_DEPENDENCY_MESSAGE = (
    "Switch collection requires optional Torch and Transformers dependencies; "
    f"install them with `{SWITCH_EXTRA_INSTALL}`."
)


class SwitchDependencyError(RuntimeError):
    """Raised when a Switch-only operation lacks its optional dependencies."""


def load_switch_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Import the pinned Switch stack only when collection is requested."""
    try:
        torch = import_module("torch")
        transformers = import_module("transformers")
        collector = import_module("moe_cache_lab.switch_collector")
    except ModuleNotFoundError as error:
        if error.name in {"torch", "transformers"}:
            raise SwitchDependencyError(SWITCH_DEPENDENCY_MESSAGE) from None
        raise
    return torch, transformers, collector


__all__ = [
    "DEFAULT_SWITCH_MODEL",
    "SWITCH_DEPENDENCY_MESSAGE",
    "SWITCH_EXTRA_INSTALL",
    "SWITCH_MODEL_REVISION",
    "SwitchDependencyError",
    "load_switch_modules",
]
