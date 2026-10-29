"""Resolve profiles from the loaded TA, validate them, and freeze run inputs."""

from copy import deepcopy
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = "COSTMODEL_PROFILE_CONTEXT"
CONTEXT_ID = "COSTMODEL_PROFILE_CONTEXT_ID"


def json_object(raw, label):

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON value {value} in {label}")

    value = json.loads(raw, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {label}")
    return value


def read_json(path):
    return json_object(Path(path).read_bytes(), path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns
    }


def runtime_version(runtime):
    return (runtime.get("ta_distribution_version") or runtime.get("ta_version")
            or runtime.get("triton_version"))


def installed_runtime():
    """Executed in a fresh interpreter with the selected device environment."""
    triton = importlib.import_module("triton")
    ascend = importlib.import_module("triton.backends.ascend")
    driver = importlib.import_module("triton.runtime").driver
    target = driver.active.get_current_target()
    if target.backend not in ("npu", "ascend"):
        raise ValueError(f"expected an Ascend NPU target, got {target!r}")
    package = Path(ascend.__file__).resolve().parent
    native = importlib.import_module("triton._C.libtriton")
    roots = []
    if getattr(native, "__file__", None):
        roots.append(
            Path(native.__file__).resolve().parent /
            "ascend/costmodel_profiles")
    roots += [package / "costmodel_profiles", package / "costmodel/profiles"]
    # Editable installations may load the actual backend source directory.
    if package.name == "backend" and package.parent.name == "ascend":
        roots.append(package.parent / "costmodel/profiles")
    files = [
        Path(triton.__file__),
        Path(ascend.__file__), package / "compiler.py"
    ]
    if getattr(native, "__file__", None):
        files.append(Path(native.__file__))
    distribution_version = None
    try:
        distribution = package_metadata.distribution("triton-ascend")
        owned_backend = Path(
            distribution.locate_file(
                "triton/backends/ascend/__init__.py")).resolve()
        if owned_backend == Path(ascend.__file__).resolve():
            distribution_version = distribution.version
    except package_metadata.PackageNotFoundError:
        pass
    return {
        "triton_version": getattr(triton, "__version__", None),
        "ta_version": getattr(ascend, "__version__", None),
        "ta_distribution_version": distribution_version,
        "triton_package": str(Path(triton.__file__).resolve()),
        "ta_package": str(package),
        "files": [fingerprint(path) for path in files if path.is_file()],
        "profile_roots":
        [str(path.resolve()) for path in roots if path.is_dir()],
        "target": {
            "backend": target.backend,
            "arch": str(target.arch)
        }
    }


def probe_runtime(env):
    child = env.copy()
    child["PYTHONPATH"] = str(ROOT) + os.pathsep + child.get("PYTHONPATH", "")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "costmodel_tests.profiles", "--probe"],
            env=child,
            capture_output=True,
            text=True,
            check=False,
            timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(
            f"Cannot inspect the installed TA/NPU: {error}") from error
    marker = "COSTMODEL_RUNTIME_JSON="
    for line in reversed(result.stdout.splitlines()):
        if result.returncode == 0 and line.startswith(marker):
            return json.loads(line[len(marker):])
    detail = (result.stderr or result.stdout).strip()[-2000:]
    raise ValueError(
        "Cannot inspect the installed TA and NPU target. Load the TA/CANN "
        f"environment first. {detail}")


def validate_json(data, schema_path, label, provenance_warnings=None):
    try:
        import jsonschema
    except ImportError as error:
        raise ValueError(
            "Profile validation requires jsonschema; install costmodel_tests/requirements.txt"
        ) from error
    schema = schema_path if isinstance(schema_path,
                                       dict) else read_json(schema_path)
    validator = jsonschema.validators.validator_for(schema)
    try:
        validator.check_schema(schema)
    except jsonschema.SchemaError as error:
        raise ValueError(
            f"Invalid schema shipped by the installed TA: {error.message}"
        ) from error
    for error in validator(schema).iter_errors(data):
        location = ".".join(str(part)
                            for part in error.absolute_path) or "<root>"
        # Some TA wheels ship this provenance tag before updating its enum.
        # It describes where a measurement came from, not its numeric value.
        path = list(error.absolute_path)
        if (provenance_warnings is not None
                and label == "microbenchmark profile"
                and len(path) == 3 and path[0] == "measurements"
                and path[2] == "source_kind"
                and error.validator == "enum"
                and error.instance == "derived_calibration"
                and error.validator_value == [
                    "isolated_microbenchmark", "device_configuration",
                    "architecture_fact"
                ]):
            provenance_warnings.append(
                f"TA profile/schema provenance mismatch at {location}: "
                "source_kind=derived_calibration is absent from the bundled enum; "
                "continuing with the original parameters. Numeric, target and "
                "remaining schema checks still apply.")
            continue
        raise ValueError(
            f"{label} is incompatible with the installed TA schema at {location}: {error.message}"
        )


def validate_profile(source, runtimes, schemas, provenance_warnings=None):
    raw = source.read_bytes()
    data = json_object(raw, source)
    for runtime in runtimes:
        arch = runtime["target"]["arch"]
        patterns = data.get("compatible_targets", [])
        if not isinstance(patterns, list) or not any(
                isinstance(pattern, str)
                and fnmatch.fnmatchcase(arch, pattern)
                for pattern in patterns):
            raise ValueError(
                f"Profile {source} does not support NPU target {arch}")
    validate_json(data, schemas[0], "selection profile")
    reference = data.get("microbenchmark_profile")
    if not isinstance(reference, str) or not reference:
        raise ValueError(
            f"selection profile has no microbenchmark_profile: {source}")
    dependency = (source.parent / reference).resolve()
    hardware_raw = dependency.read_bytes()
    hardware = json_object(hardware_raw, dependency)
    validate_json(hardware, schemas[1], "microbenchmark profile", provenance_warnings)
    if hardware.get("target") != data.get("target"):
        raise ValueError(
            "Selection and microbenchmark profiles have different targets")
    measurements = hardware.get("measurements", {})

    def check_references(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if (key == "measurement"
                        or key.endswith("_measurement")) and isinstance(
                            item, str):
                    if item not in measurements:
                        raise ValueError(
                            f"Missing hardware measurement: {item}")
                check_references(item)
        elif isinstance(value, list):
            for item in value:
                check_references(item)

    check_references(data)
    materials = {source: raw, dependency: hardware_raw}
    if "indirect_memory_profile" in data:
        reference = data["indirect_memory_profile"]
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"invalid indirect_memory_profile in {source}")
        indirect = (source.parent / reference).resolve()
        indirect_raw = indirect.read_bytes()
        # No separate schema is shipped for this optional TA resource.
        # Reject malformed/non-finite JSON here; TA validates model dimensions.
        json_object(indirect_raw, indirect)
        materials[indirect] = indirect_raw
    return data, dependency, materials


def resolve_profile(requested, env, devices=None):
    """Only installed TA assets supply compatibility schemas; never bundled fallbacks."""
    device_ids = devices or env.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    device_ids = list(
        dict.fromkeys(device.strip() for device in device_ids
                      if device.strip())) or ["default"]
    runtimes = []
    for device in device_ids:
        child = env.copy()
        if device != "default":
            child["ASCEND_RT_VISIBLE_DEVICES"] = device
        runtime = probe_runtime(child)
        runtime["device"] = device
        runtimes.append(runtime)
    roots = [Path(path) for path in runtimes[0]["profile_roots"]]
    pairs = [(root / "simd_simt/simd_simt_profile_schema.json",
              root / "microbench/microbenchmark_profile_schema.json")
             for root in roots]
    schemas = next(
        (pair for pair in pairs if all(path.is_file() for path in pair)), None)
    if schemas is None:
        version = runtime_version(runtimes[0])
        raise ValueError(
            f"The installed TA {version} at {runtimes[0]['ta_package']} "
            "does not ship CostModel profile schemas. Install a TA build "
            "with costmodel_profiles (including simd_simt and microbench schemas). "
            "The test repository's profiles are not an automatic fallback.")
    schema_raw = {path: path.read_bytes() for path in schemas}
    schema_objects = [json_object(schema_raw[path], path) for path in schemas]
    selected = requested or env.get("TRITON_ASCEND_AUTO_SIMT_PROFILE")
    source_kind = "explicit" if requested is not None else "environment" if selected else "installed_ta"
    validation_warnings = []
    if selected:
        source = Path(selected).resolve()
        data, dependency, materials = validate_profile(source, runtimes,
                                                       schema_objects, validation_warnings)
    else:
        candidates, rejected = [], []
        for source in sorted({
                path.resolve()
                for root in roots
                for path in (root / "simd_simt").glob("*.json")
        }):
            if source.name.endswith("schema.json"):
                continue
            try:
                candidate_warnings = []
                data, dependency, materials = validate_profile(
                    source, runtimes, schema_objects, candidate_warnings)
                candidates.append((source, data, dependency, materials, candidate_warnings))
            except (OSError, ValueError) as error:
                rejected.append(f"{source.name}: {error}")
        if len(candidates) != 1:
            detail = "; ".join(str(item[0])
                               for item in candidates) or "; ".join(rejected)
            raise ValueError(
                "Expected one compatible profile in the installed TA; "
                f"found {len(candidates)}. Use --profile to choose explicitly. {detail}"
            )
        source, data, dependency, materials, validation_warnings = candidates[0]
    return {
        "source": source,
        "data": data,
        "dependency": dependency,
        "schemas": schemas,
        "runtimes": runtimes,
        "source_kind": source_kind,
        "validation_warnings": validation_warnings,
        "materials": {
            **materials,
            **schema_raw
        }
    }


def _inherit_context(source, destination, env, devices):
    """Reuse this run's verified inputs in workers without initializing every NPU again."""
    if not env.get(CONTEXT) or not env.get(CONTEXT_ID):
        return False
    context_path = Path(env[CONTEXT]).resolve()
    metadata = read_json(context_path)
    if metadata.get("context_id") != env[CONTEXT_ID] or metadata.get(
            "host") != socket.gethostname():
        raise ValueError("Profile context does not belong to this run/host")
    if source != context_path.parent / "costmodel_profile.json":
        return False
    selected = devices or env.get("ASCEND_RT_VISIBLE_DEVICES", "").split(",")
    selected = {item.strip()
                for item in selected if item.strip()} or {"default"}
    if not selected <= {item["device"] for item in metadata["runtimes"]}:
        raise ValueError(
            "Worker device was not included in profile compatibility checks")
    # Detect a different Python environment or a package changed after preflight.
    triton = importlib.import_module("triton")
    if str(Path(triton.__file__).resolve()
           ) != metadata["runtimes"][0]["triton_package"]:
        raise ValueError("Worker loaded a different TA installation")
    ascend = importlib.import_module("triton.backends.ascend")
    if str(Path(ascend.__file__).resolve().parent
           ) != metadata["runtimes"][0]["ta_package"]:
        raise ValueError("Worker loaded a different Ascend backend")
    for runtime in metadata["runtimes"]:
        for item in runtime["files"]:
            stat = Path(item["path"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (item["size"],
                                                    item["mtime_ns"]):
                raise ValueError(
                    "TA installation changed after profile preflight")
    for name, expected in metadata["artifacts"].items():
        if not (context_path.parent / name).resolve().is_relative_to(
                context_path.parent):
            raise ValueError("Invalid path in profile snapshot manifest")
        if digest(context_path.parent / name) != expected:
            raise ValueError(
                f"Profile snapshot changed after preflight: {name}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for name in metadata["artifacts"]:
        src, dst = context_path.parent / name, destination.parent / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copy2(src, dst)
    (destination.parent / "profile_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    env[CONTEXT] = str(destination.parent / "profile_metadata.json")
    return True


def prepare_profile(requested,
                    destination,
                    env,
                    devices=None,
                    mode="costmodel"):
    """Preflight before any kernel runs; return the frozen profile or None for host-only work."""
    destination = Path(destination).resolve()
    if mode not in ("costmodel", "functional", "collect_only"):
        raise ValueError(f"Unknown profile validation mode: {mode}")
    source = requested or env.get("TRITON_ASCEND_AUTO_SIMT_PROFILE")
    if mode != "costmodel":
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "validation_scope": mode,
            "profile_used_by_costmodel": False
        }
        if source:
            snapshot_profile(Path(source), destination)
            metadata["recorded_profile_sha256"] = digest(destination)
        (destination.parent / "profile_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return destination if source else None
    if source and _inherit_context(
            Path(source).resolve(), destination, env, devices):
        env["TRITON_ASCEND_AUTO_SIMT_PROFILE"] = str(destination)
        return destination
    try:
        plan = resolve_profile(requested, env, devices)
    except (OSError, ValueError) as error:
        raise ValueError(f"Profile preflight failed: {error}") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = plan["source"]
    raw = destination.parent / "profile_sources/selection.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(plan["materials"][source])
    data = deepcopy(plan["data"])
    relative_dependency = Path(
        "profile_dependencies") / plan["dependency"].name
    dependency_target = destination.parent / relative_dependency
    dependency_target.parent.mkdir(parents=True, exist_ok=True)
    dependency_target.write_bytes(plan["materials"][plan["dependency"]])
    data["microbenchmark_profile"] = relative_dependency.as_posix()
    indirect_name = None
    if "indirect_memory_profile" in data:
        indirect = (source.parent / data["indirect_memory_profile"]).resolve()
        relative = Path("profile_dependencies/indirect_memory") / indirect.name
        target = destination.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(plan["materials"][indirect])
        indirect_name = data["indirect_memory_profile"] = relative.as_posix()
    data["$schema"] = "profile_schemas/simd_simt_profile_schema.json"
    destination.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for schema, relative in zip(
            plan["schemas"],
        ("profile_schemas/simd_simt_profile_schema.json",
         "profile_dependencies/microbenchmark_profile_schema.json")):
        target = destination.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(plan["materials"][schema])
    names = [
        destination.name, "profile_sources/selection.json",
        data["microbenchmark_profile"],
        "profile_schemas/simd_simt_profile_schema.json",
        "profile_dependencies/microbenchmark_profile_schema.json"
    ]
    if indirect_name:
        names.append(indirect_name)
    metadata = {
        "validation_scope": "costmodel",
        "profile_used_by_costmodel": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "context_id": uuid.uuid4().hex,
        "source_kind": plan["source_kind"],
        "validation_warnings": plan["validation_warnings"],
        "source_path": str(source),
        "source_sha256": digest(raw),
        "profile_version": plan["data"]["profile_version"],
        "schema_version": plan["data"]["schema_version"],
        "target": plan["data"]["target"],
        "runtimes": plan["runtimes"],
        "artifacts": {
            name: digest(destination.parent / name)
            for name in names
        }
    }
    context_path = destination.parent / "profile_metadata.json"
    context_path.write_text(json.dumps(metadata, indent=2) + "\n",
                            encoding="utf-8")
    env["TRITON_ASCEND_AUTO_SIMT_PROFILE"] = str(destination)
    env[CONTEXT], env[CONTEXT_ID] = str(context_path), metadata["context_id"]
    print(
        f"Profile: {source} ({plan['source_kind']}); target={metadata['target']}",
        flush=True)
    print(f"Profile SHA256: {metadata['source_sha256']}", flush=True)
    for warning in metadata["validation_warnings"]:
        print("WARNING: " + warning, flush=True)
    return destination


def snapshot_profile(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    profile = json.loads(source.read_text(encoding="utf-8"))
    keys = [key for key in ("microbenchmark_profile", "indirect_memory_profile") if key in profile]
    if not keys:
        if source != destination:
            shutil.copy2(source, destination)
        return
    for key in keys:
        reference = profile[key]
        if not isinstance(reference, str) or not reference:
            raise ValueError(f"invalid {key} in {source}")
        dependency = (source.parent / reference).resolve()
        if not dependency.is_file():
            raise FileNotFoundError(f"{key} referenced by {source}: {dependency}")
        folder = Path("profile_dependencies")
        if key == "indirect_memory_profile":
            folder /= "indirect_memory"
        relative = folder / dependency.name
        target = destination.parent / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if dependency != target.resolve():
            shutil.copy2(dependency, target)
        profile[key] = relative.as_posix()
    destination.write_text(json.dumps(profile, indent=2) + "\n",
                           encoding="utf-8")


def profile_report_lines(output):
    path = Path(output) / "profile_metadata.json"
    if not path.is_file():
        return []
    metadata = read_json(path)
    if not metadata.get("profile_used_by_costmodel"):
        return [
            "Profile: not used by CostModel (" + metadata["validation_scope"] +
            ").", ""
        ]
    runtime = metadata["runtimes"][0]
    return [
        "## Profile and TA", "",
        f"- TA: `{runtime_version(runtime)}`; `{runtime['ta_package']}`",
        f"- Hardware: `{', '.join(item['target']['arch'] for item in metadata['runtimes'])}`",
        f"- Profile: `{metadata['profile_version']}` ({metadata['source_kind']})",
        f"- Original profile SHA256: `{metadata['source_sha256']}`",
        "- [Versions, hashes and snapshot manifest](profile_metadata.json)",
        *("- WARNING: " + warning for warning in metadata.get("validation_warnings", [])), ""
    ]


if __name__ == "__main__":
    if sys.argv[1:] != ["--probe"]:
        raise SystemExit(
            "Internal runtime probe: python -m costmodel_tests.profiles --probe"
        )
    print("COSTMODEL_RUNTIME_JSON=" + json.dumps(installed_runtime()))
