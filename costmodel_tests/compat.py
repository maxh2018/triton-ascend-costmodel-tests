"""Compatibility for CI hardware checks across installed Triton versions."""

import importlib
import inspect


def is_compile_on_910_95(arch):
    """Ask the installed backend, whose older API uses the current target."""
    predicate = importlib.import_module("triton.backends.ascend.utils").is_compile_on_910_95
    try:
        inspect.signature(predicate).bind(arch)
    except TypeError:
        return predicate()
    return predicate(arch)
