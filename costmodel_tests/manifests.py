"""Read a benchmark/pytest manifest without importing any NPU dependencies."""
from __future__ import annotations
import json
from pathlib import Path
import re


def load_manifest(path: Path) -> dict:
    # Drivers are relative to this JSON file, not the selected category.
    # Absolute paths and ../ references to other directories are supported.
    path = path.resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("cases"), list):
        raise ValueError("manifest must contain a cases list")
    names = set()
    for case in manifest["cases"]:
        if not isinstance(case, dict):
            raise ValueError("each case must be an object")
        name = case.get("name", "")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", name) or name in names:
            raise ValueError(f"invalid or duplicate case name: {name!r}")
        names.add(name)
        runner = case.setdefault("runner", "benchmark")
        if runner not in ("benchmark", "pytest"):
            raise ValueError(f"{name}: runner must be benchmark or pytest")
        driver = case.get("driver")
        if not isinstance(driver, str) or not driver:
            raise ValueError(f"{name}: driver must name a Python file")
        driver = (path.parent / driver).resolve()
        if not driver.is_file() or driver.suffix != ".py":
            raise ValueError(f"{name}: Python driver does not exist: {driver}")
        case["driver"] = str(driver)
        arguments = case.setdefault("arguments", [])
        if not isinstance(arguments, list) or any(not isinstance(v, str) for v in arguments):
            raise ValueError(f"{name}: arguments must be a list of strings")
        case.setdefault("family", "custom")
        case.setdefault("split", "holdout")
    return manifest


def select_cases(manifest: dict, names: list[str]) -> list[dict]:
    unknown = set(names) - {c["name"] for c in manifest["cases"]}
    if unknown:
        raise ValueError("unknown case(s): " + ", ".join(sorted(unknown)))
    return [c for c in manifest["cases"] if not names or c["name"] in names]
