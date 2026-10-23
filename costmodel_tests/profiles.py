"""Snapshot a selection profile together with its runtime measurement dependency."""

import json
from pathlib import Path
import shutil


def snapshot_profile(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    profile = json.loads(source.read_text(encoding="utf-8"))
    reference = profile.get("microbenchmark_profile")
    if reference is None:
        if source != destination:
            shutil.copy2(source, destination)
        return
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"invalid microbenchmark_profile in {source}")
    dependency = (source.parent / reference).resolve()
    if not dependency.is_file():
        raise FileNotFoundError(f"microbenchmark_profile referenced by {source}: {dependency}")
    relative = Path("profile_dependencies") / dependency.name
    target = destination.parent / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if dependency != target.resolve():
        shutil.copy2(dependency, target)
    profile["microbenchmark_profile"] = relative.as_posix()
    destination.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
