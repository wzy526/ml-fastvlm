"""Silence the CuTe-DSL `struct.scalar.ptr` deprecation spam from FA4.

nvidia_cutlass_dsl emits this from `struct.scalar.value` on every kernel
compile, and it wraps the call in `catch_warnings() + simplefilter("always")`,
so `warnings.filterwarnings` / `PYTHONWARNINGS` / `-W ignore` cannot reach it.
The only lever left is `showwarning`, which `catch_warnings` saves and restores
but never resets — so an override installed here survives their context manager.

Set DAT_SILENCE_CUTLASS_WARN=0 to keep the warnings.
"""

import os
import warnings

_MUTED_SUBSTRINGS = (
    "struct.scalar.ptr",
    "Using `struct.scalar` as pointer is deprecated",
)

_installed = False


def install() -> None:
    global _installed
    if _installed or os.environ.get("DAT_SILENCE_CUTLASS_WARN", "1") == "0":
        return

    previous = warnings.showwarning

    def showwarning(message, category, filename, lineno, file=None, line=None):
        if category is DeprecationWarning and any(s in str(message) for s in _MUTED_SUBSTRINGS):
            return
        previous(message, category, filename, lineno, file, line)

    warnings.showwarning = showwarning
    _installed = True
