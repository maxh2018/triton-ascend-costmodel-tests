#!/usr/bin/env python3
"""Refresh the two CI manifests. Does not run or validate NPU kernels."""
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / "cases/ci"
CANONICAL = CI / "manifest.json"
EXCLUDED = CI / "manifest_excluded.json"


def build_manifests():
    default = json.loads(CANONICAL.read_text(encoding="utf-8"))
    default[
        "description"] = "Default CI manifest: original PR #2528 100 cases."
    complete = deepcopy(default)
    complete[
        "description"] = "CI extended manifest with temporary failure exclusions; explicitly select to run."
    complete["validation_status"] = "unverified"
    complete[
        "inventory_note"] = "Pytest entries are files; their parameterized nodes are collected at runtime."
    # Path ordering depends on the OS; keep the manifest order stable everywhere.
    for path in sorted(
            CI.rglob("test_*.py"),
            key=lambda path: path.relative_to(CI).as_posix().casefold()):
        relative = path.relative_to(CI)
        name = "pytest_" + path.stem
        complete["cases"].append({
            "name": name,
            "runner": "pytest",
            "family": "pytest",
            "split": "holdout",
            "driver": relative.as_posix(),
            "arguments": []
        })
    names = [c["name"] for c in complete["cases"]]
    if len(names) != len(set(names)):
        raise ValueError(
            "duplicate generated names; qualify the conflicting pytest filenames"
        )
    excluded = json.loads(EXCLUDED.read_text(encoding="utf-8"))["cases"]
    excluded_names = [case["name"] for case in excluded]
    pytest_names = {
        case["name"]
        for case in complete["cases"] if case.get("runner") == "pytest"
    }
    if len(excluded_names) != len(set(excluded_names)):
        raise ValueError("duplicate case names in manifest_excluded.json")
    excluded_names = set(excluded_names)
    unknown = excluded_names - pytest_names
    if unknown:
        raise ValueError("excluded cases are not CI pytest entries: " +
                         ", ".join(sorted(unknown)))
    complete["cases"] = [
        case for case in complete["cases"]
        if case["name"] not in excluded_names
    ]
    complete["excluded_manifest"] = EXCLUDED.name
    default = {"description": default.pop("description"), **default}
    complete = {
        "description": complete.pop("description"),
        "validation_status": complete.pop("validation_status"),
        "inventory_note": complete.pop("inventory_note"),
        **complete
    }
    return default, complete


def main():
    default, complete = build_manifests()
    for name, manifest in (("manifest.json", default), ("manifest_all.json",
                                                        complete)):
        (CI / name).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")
        print(f"{CI / name}: {len(manifest['cases'])} entries")
    print(
        "manifest_all.json remains unverified; the default is manifest.json (100 cases)."
    )
    print("Temporary exclusions are preserved from manifest_excluded.json.")


if __name__ == "__main__":
    main()
