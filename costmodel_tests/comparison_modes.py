"""Explicit dashboard comparisons reuse only declared Mix implementations."""

BASE_MODES = (("auto", 1), ("simd", 1), ("simt_only", 1))


def comparison_modes(case):
    modes = list(BASE_MODES)
    for entry in case.get("extra_modes", []):
        key = (entry.get("route"), int(entry.get("factor", 1)))
        if key[0] == "mixed" and key not in modes:
            modes.append(key)
    return modes


def matrix_mode_flags(scope):
    return (["--comparison-modes"] if scope == "compare" else
            ["--include-explicit"] if scope == "all" else [])
