"""Reproducible, isolated CUDA 13.2 runtime for Rust ORT self-play."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from email.parser import BytesParser
from email.policy import default as email_policy
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "alphamini.cuda-runtime.v1"
CUDA_RELEASE = "13.2"
REQUIREMENTS_RELATIVE = Path("configs/alphamini/cuda13-runtime-requirements.txt")
STAGE_RELATIVE = Path("target/alphamini-cuda-runtime") / CUDA_RELEASE
LOADER_DIRECTORIES = (Path("nvidia/cu13/lib"), Path("nvidia/cudnn/lib"))
ORT_CRATE_VERSION = "2.0.0-rc.13"
ORT_RUNTIME_VERSION = "1.28.0"
ORT_DISTRIBUTION_SHA256 = "b89451babb9d4ec77c3f381e8ef92b640e125021b59f9cc7747e73c9fe8d549b"
ORT_STATIC_ARCHIVE_SHA256 = "a918aed6fb410efd053772172b7ab1931e42c4b75da719dce3d3bd1ca1e9aa0d"
ORT_PROVIDER_SHA256 = {
    "libonnxruntime_providers_cuda.so": (
        "fea89a0df08436582b275eaa56e3ec4b716bbd7a253d73cf48d97c3651e48b83"
    ),
    "libonnxruntime_providers_shared.so": (
        "5f07ccc053d33441efd8f2ef963c4616330b2c5c778e8f58bac88b3f2aa9bec5"
    ),
}
PACKAGE_VERSIONS = {
    "nvidia-cublas": "13.2.2.2",
    "nvidia-cuda-nvrtc": "13.2.86",
    "nvidia-cuda-runtime": "13.2.86",
    "nvidia-cudnn-cu13": "9.23.2.1",
    "nvidia-cufft": "12.2.0.57",
    "nvidia-curand": "10.4.2.66",
    "nvidia-nvjitlink": "13.2.86",
}
REQUIRED_STAGE_SONAMES = (
    "libcublas.so.13",
    "libcublasLt.so.13",
    "libcudart.so.13",
    "libcudnn.so",
    "libcudnn.so.9",
    "libcufft.so.12",
    "libcurand.so.10",
    "libnvrtc.so.13",
)
DIRECT_PROVIDER_SONAMES = (
    "libcublas.so.13",
    "libcublasLt.so.13",
    "libcudart.so.13",
    "libcurand.so.10",
)


class CudaRuntimeError(RuntimeError):
    """The pinned CUDA runtime is missing, corrupt, or incompatible."""


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _normalize_package(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _relative(path: Path, root: Path, label: str) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise CudaRuntimeError(f"{label} escapes the CUDA runtime stage: {path}") from error


def resolve_stage(repository: Path | str, stage: Path | str | None = None) -> Path:
    root = Path(repository).resolve()
    value = stage or os.environ.get("ALPHAMINI_CUDA_RUNTIME_DIR") or STAGE_RELATIVE
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    target = (root / "target").resolve()
    try:
        relative = candidate.relative_to(target)
    except ValueError as error:
        raise CudaRuntimeError(
            f"CUDA runtime stage must be beneath {target}, not {candidate}"
        ) from error
    if not relative.parts:
        raise CudaRuntimeError("CUDA runtime stage cannot overlay the Cargo target directory")
    training_venv = (root / "alphamini-train/.venv").resolve()
    if candidate == training_venv or training_venv in candidate.parents:
        raise CudaRuntimeError("CUDA runtime stage cannot overlay alphamini-train/.venv")
    return candidate


def parse_requirements(path: Path) -> list[dict[str, str]]:
    pattern = re.compile(
        r"^([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:([0-9a-f]{64})$"
    )
    packages: list[dict[str, str]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = pattern.fullmatch(line)
        if match is None:
            raise CudaRuntimeError(
                f"{path}:{number}: require one exact pin and one SHA-256 per line"
            )
        name, version, wheel_sha256 = match.groups()
        packages.append(
            {
                "name": _normalize_package(name),
                "version": version,
                "wheel_sha256": wheel_sha256,
            }
        )
    actual = {item["name"]: item["version"] for item in packages}
    if actual != PACKAGE_VERSIONS or len(actual) != len(packages):
        raise CudaRuntimeError(
            f"CUDA requirements must contain exactly the reviewed package pins: {PACKAGE_VERSIONS}"
        )
    return sorted(packages, key=lambda item: item["name"])


def _metadata_packages(stage: Path, requirements: list[dict[str, str]]) -> list[dict[str, str]]:
    installed: dict[str, str] = {}
    for metadata_path in sorted(stage.glob("*.dist-info/METADATA")):
        message = BytesParser(policy=email_policy).parsebytes(metadata_path.read_bytes())
        name = _normalize_package(str(message.get("Name", "")))
        version = str(message.get("Version", ""))
        if not name or not version or name in installed:
            raise CudaRuntimeError(f"invalid or duplicate wheel metadata: {metadata_path}")
        installed[name] = version
    expected = {item["name"]: item["version"] for item in requirements}
    if installed != expected:
        raise CudaRuntimeError(
            f"isolated stage package metadata differs: expected {expected}, found {installed}"
        )
    return requirements


def _runtime_files(stage: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    libraries: list[dict[str, Any]] = []
    symlinks: list[dict[str, str]] = []
    names: set[str] = set()
    for relative_directory in LOADER_DIRECTORIES:
        directory = stage / relative_directory
        if not directory.is_dir():
            raise CudaRuntimeError(f"missing CUDA loader directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                resolved = path.resolve()
                if not resolved.is_file():
                    raise CudaRuntimeError(f"broken CUDA runtime symlink: {path}")
                _relative(resolved, stage, "CUDA runtime symlink target")
                symlinks.append(
                    {
                        "path": _relative(path, stage, "CUDA runtime symlink"),
                        "target": os.readlink(path),
                    }
                )
                names.add(path.name)
            elif path.is_file() and ".so" in path.name:
                libraries.append(
                    {
                        "path": _relative(path, stage, "CUDA runtime library"),
                        "sha256": _sha256(path),
                        "size": path.stat().st_size,
                    }
                )
                names.add(path.name)
    missing = sorted(set(REQUIRED_STAGE_SONAMES) - names)
    if missing:
        raise CudaRuntimeError(f"CUDA runtime is missing required SONAME files: {missing}")
    return libraries, symlinks


def _provider_records(stage: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for name, expected_hash in sorted(ORT_PROVIDER_SHA256.items()):
        path = stage / "ort" / name
        if not path.is_file() or path.is_symlink():
            raise CudaRuntimeError(f"staged ORT provider must be a regular file: {path}")
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise CudaRuntimeError(
                f"ORT provider SHA-256 mismatch for {name}: {actual_hash} != {expected_hash}"
            )
        records.append(
            {
                "name": name,
                "path": f"ort/{name}",
                "sha256": actual_hash,
                "size": path.stat().st_size,
            }
        )
    return records


def _wrapper_content() -> str:
    return (
        "#!/bin/sh\n"
        "set -eu\n"
        'runtime_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)\n'
        'repository_root=$(CDPATH= cd -- "$runtime_dir/../../.." && pwd)\n'
        'export ALPHAMINI_CUDA_RUNTIME_MANIFEST="$runtime_dir/manifest.json"\n'
        'export CARGO_TARGET_DIR="$repository_root/target"\n'
        'if [ -n "${LD_LIBRARY_PATH:-}" ]; then\n'
        '  export LD_LIBRARY_PATH="$runtime_dir/nvidia/cu13/lib:'
        '$runtime_dir/nvidia/cudnn/lib:$LD_LIBRARY_PATH"\n'
        "else\n"
        '  export LD_LIBRARY_PATH="$runtime_dir/nvidia/cu13/lib:'
        '$runtime_dir/nvidia/cudnn/lib"\n'
        "fi\n"
        'exec "$@"\n'
    )


def _write_wrapper(stage: Path) -> dict[str, Any]:
    wrapper = stage / "with-alphamini-cuda"
    content = _wrapper_content()
    wrapper.write_text(content, encoding="utf-8")
    wrapper.chmod(0o755)
    return {
        "path": wrapper.name,
        "sha256": _sha256(wrapper),
        "command": [wrapper.name, "<command>", "<args>"],
    }


def _validate_platform() -> None:
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
        raise CudaRuntimeError("the pinned CUDA/ORT artifacts support Linux x86_64 only")


def _run(command: Sequence[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    result = subprocess.run(
        list(command), cwd=cwd, env=env, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-4000:]
        raise CudaRuntimeError(
            f"command exited {result.returncode}: {' '.join(command)}\n{detail}"
        )


def _fresh_provider_sources(repository: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    release = repository / "target/release"
    release.mkdir(parents=True, exist_ok=True)
    for name in ORT_PROVIDER_SHA256:
        destination = release / name
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(repository / "target")
    _run(
        ["cargo", "clean", "-p", f"ort-sys@{ORT_CRATE_VERSION}"],
        cwd=repository,
        env=env,
    )
    _run(
        [
            "cargo",
            "build",
            "--locked",
            "--release",
            "-p",
            "alphamini",
            "--bin",
            "alphamini-selfplay",
            "--features",
            "cuda",
        ],
        cwd=repository,
        env=env,
    )
    sources: dict[str, Path] = {}
    distribution_directory: Path | None = None
    for name, expected_hash in ORT_PROVIDER_SHA256.items():
        link = release / name
        if not link.is_symlink():
            raise CudaRuntimeError(
                f"fresh CUDA build did not emit the expected copy-dylibs link: {link}"
            )
        source = link.resolve()
        if not source.is_file() or _sha256(source) != expected_hash:
            raise CudaRuntimeError(f"fresh CUDA build emitted an unexpected {name}")
        if source.parent.name != ORT_DISTRIBUTION_SHA256:
            raise CudaRuntimeError(
                f"unexpected ORT distribution identity: {source.parent.name}"
            )
        if distribution_directory is not None and source.parent != distribution_directory:
            raise CudaRuntimeError("ORT providers came from different distributions")
        distribution_directory = source.parent
        sources[name] = source
    assert distribution_directory is not None
    archive = distribution_directory / "libonnxruntime.a"
    if not archive.is_file() or _sha256(archive) != ORT_STATIC_ARCHIVE_SHA256:
        raise CudaRuntimeError(
            "ORT static archive identity differs from the reviewed rc.13 artifact"
        )
    return sources, {
        "crate_version": ORT_CRATE_VERSION,
        "runtime_version": ORT_RUNTIME_VERSION,
        "distribution_sha256": ORT_DISTRIBUTION_SHA256,
        "static_archive_sha256": ORT_STATIC_ARCHIVE_SHA256,
    }


def _select_python(repository: Path, requested: Path | str | None) -> Path:
    if requested is not None:
        executable = Path(requested).resolve()
    else:
        venv_python = repository / "alphamini-train/.venv/bin/python"
        executable = venv_python.resolve() if venv_python.is_file() else Path(sys.executable)
    if not executable.is_file():
        raise CudaRuntimeError(f"Python executable does not exist: {executable}")
    return executable


def _install_wheels(
    repository: Path, temporary_stage: Path, python: Path, requirements: Path
) -> None:
    env = dict(os.environ)
    env.update(
        {
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--no-deps",
            "--only-binary=:all:",
            "--no-compile",
            "--target",
            str(temporary_stage),
            "-r",
            str(requirements),
        ],
        cwd=repository,
        env=env,
    )


def _ensure_cudnn_loader_name(stage: Path) -> None:
    directory = stage / "nvidia/cudnn/lib"
    versioned = directory / "libcudnn.so.9"
    generic = directory / "libcudnn.so"
    if not versioned.is_file():
        raise CudaRuntimeError(f"pinned cuDNN wheel lacks {versioned.name}")
    if generic.exists() or generic.is_symlink():
        if not generic.is_symlink() or os.readlink(generic) != versioned.name:
            raise CudaRuntimeError(f"unexpected pre-existing generic cuDNN loader name: {generic}")
    else:
        generic.symlink_to(versioned.name)


def _manifest_identity(manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    libraries = manifest.get("runtime_libraries")
    symlinks = manifest.get("runtime_symlinks")
    if not isinstance(libraries, list) or not isinstance(symlinks, list):
        raise CudaRuntimeError("CUDA runtime manifest lacks library identities")
    library_set_sha256 = hashlib.sha256(
        json.dumps(
            {"runtime_libraries": libraries, "runtime_symlinks": symlinks},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "status": "recorded",
        "schema": manifest.get("schema"),
        "cuda_release": manifest.get("cuda_release"),
        "manifest_sha256": _sha256(manifest_path),
        "requirements_sha256": manifest.get("requirements", {}).get("sha256"),
        "packages": manifest.get("requirements", {}).get("packages"),
        "runtime_library_set_sha256": library_set_sha256,
        "ort": manifest.get("ort"),
    }


def recorded_runtime_identity(repository: Path | str | None = None) -> dict[str, Any]:
    root = Path(repository or repository_root()).resolve()
    try:
        stage = resolve_stage(root)
    except CudaRuntimeError as error:
        return {"status": "invalid", "error": str(error)}
    manifest_path = stage / "manifest.json"
    manifest_relative = _relative(manifest_path, root, "CUDA runtime manifest")
    if not manifest_path.is_file():
        return {"status": "missing", "manifest_path": manifest_relative}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
            raise CudaRuntimeError("unsupported CUDA runtime manifest")
        identity = _manifest_identity(manifest_path, manifest)
        identity["manifest_path"] = manifest_relative
        return identity
    except (CudaRuntimeError, OSError, json.JSONDecodeError) as error:
        return {"status": "invalid", "manifest_path": manifest_relative, "error": str(error)}


def _loader_environment(stage: Path) -> dict[str, str]:
    env = dict(os.environ)
    prefix = os.pathsep.join(str(stage / path) for path in LOADER_DIRECTORIES)
    inherited = env.get("LD_LIBRARY_PATH")
    env["LD_LIBRARY_PATH"] = prefix if not inherited else f"{prefix}{os.pathsep}{inherited}"
    env["ALPHAMINI_CUDA_RUNTIME_MANIFEST"] = str(stage / "manifest.json")
    return env


def _is_staged_dependency(soname: str) -> bool:
    return soname.startswith(
        (
            "libcublas",
            "libcudart",
            "libcudnn",
            "libcufft",
            "libcurand",
            "libcusparse",
            "libnvJitLink",
            "libnvrtc",
        )
    )


def _verify_ldd(
    stage: Path,
    provider_records: list[dict[str, Any]],
    runtime_records: list[dict[str, Any]],
) -> dict[str, Any]:
    ldd = shutil.which("ldd")
    if ldd is None:
        raise CudaRuntimeError("ldd is required to verify the CUDA provider dependency closure")
    reports: dict[str, Any] = {}
    targets = [*provider_records, *runtime_records]
    provider_names = {record["name"] for record in provider_records}
    for record in targets:
        path = stage / record["path"]
        result = subprocess.run(
            [ldd, str(path)],
            env=_loader_environment(stage),
            capture_output=True,
            text=True,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}".strip()
        if result.returncode != 0 or "not found" in output:
            raise CudaRuntimeError(
                f"ldd did not resolve {record['name']} (exit {result.returncode}):\n{output}"
            )
        resolved: dict[str, str] = {}
        for line in output.splitlines():
            match = re.match(r"\s*(\S+)\s+=>\s+(\S+)", line)
            if match is not None:
                soname, value = match.groups()
                resolved[soname] = value
        label = record.get("name", record["path"])
        if label == "libonnxruntime_providers_cuda.so":
            for soname in DIRECT_PROVIDER_SONAMES:
                value = resolved.get(soname)
                if value is None:
                    raise CudaRuntimeError(f"ldd omitted required CUDA dependency {soname}")
                resolved_path = Path(value).resolve()
                _relative(resolved_path, stage, f"ldd resolution for {soname}")
        normalized: dict[str, str] = {}
        for soname, value in sorted(resolved.items()):
            resolved_path = Path(value).resolve()
            if _is_staged_dependency(soname):
                normalized[soname] = _relative(
                    resolved_path, stage, f"ldd resolution for {soname}"
                )
            else:
                normalized[soname] = value
        report_name = label if label in provider_names else record["path"]
        reports[report_name] = normalized
    return reports


def _validate_manifest(
    repository: Path, stage: Path, *, require_target_providers: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requirements_path = repository / REQUIREMENTS_RELATIVE
    requirements = parse_requirements(requirements_path)
    manifest_path = stage / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CudaRuntimeError(f"cannot read CUDA runtime manifest: {error}") from error
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise CudaRuntimeError("unsupported CUDA runtime manifest")
    exact_top_level = {
        "schema",
        "platform",
        "cuda_release",
        "loader_directories",
        "requirements",
        "runtime_libraries",
        "runtime_symlinks",
        "ort",
        "wrapper",
    }
    if set(manifest) != exact_top_level:
        raise CudaRuntimeError("CUDA runtime manifest fields differ from the v1 schema")
    if manifest["platform"] != {
        "cargo_target": "x86_64-unknown-linux-gnu",
        "machine": "x86_64",
        "system": "Linux",
    }:
        raise CudaRuntimeError("CUDA runtime manifest platform differs")
    if manifest["cuda_release"] != CUDA_RELEASE:
        raise CudaRuntimeError("CUDA runtime release differs")
    if manifest["loader_directories"] != [path.as_posix() for path in LOADER_DIRECTORIES]:
        raise CudaRuntimeError("CUDA loader directory contract differs")
    expected_requirements = {
        "path": REQUIREMENTS_RELATIVE.as_posix(),
        "sha256": _sha256(requirements_path),
        "packages": requirements,
    }
    if manifest["requirements"] != expected_requirements:
        raise CudaRuntimeError("CUDA requirements identity differs from the current source")
    _metadata_packages(stage, requirements)
    libraries, symlinks = _runtime_files(stage)
    if manifest["runtime_libraries"] != libraries or manifest["runtime_symlinks"] != symlinks:
        raise CudaRuntimeError("CUDA runtime library hashes or symlinks differ from the manifest")
    providers = _provider_records(stage)
    expected_ort = {
        "crate_version": ORT_CRATE_VERSION,
        "runtime_version": ORT_RUNTIME_VERSION,
        "distribution_sha256": ORT_DISTRIBUTION_SHA256,
        "static_archive_sha256": ORT_STATIC_ARCHIVE_SHA256,
        "providers": providers,
    }
    if manifest["ort"] != expected_ort:
        raise CudaRuntimeError("Rust ORT runtime/provider identity differs")
    wrapper = stage / "with-alphamini-cuda"
    wrapper_record = manifest["wrapper"]
    expected_wrapper = {
        "path": wrapper.name,
        "sha256": _sha256(wrapper) if wrapper.is_file() else None,
        "command": [wrapper.name, "<command>", "<args>"],
    }
    if (
        not wrapper.is_file()
        or wrapper.is_symlink()
        or not os.access(wrapper, os.X_OK)
        or wrapper.read_text(encoding="utf-8") != _wrapper_content()
        or wrapper_record != expected_wrapper
    ):
        raise CudaRuntimeError("CUDA environment wrapper is missing or corrupt")
    if require_target_providers:
        for provider in providers:
            target_path = repository / "target/release" / provider["name"]
            if (
                not target_path.is_file()
                or target_path.is_symlink()
                or _sha256(target_path) != provider["sha256"]
            ):
                raise CudaRuntimeError(
                    f"release provider is absent, a cache symlink, or corrupt: {target_path}"
                )
    return manifest, providers


def verify_runtime(
    repository: Path | str | None = None,
    stage: Path | str | None = None,
    *,
    require_target_providers: bool = True,
) -> dict[str, Any]:
    _validate_platform()
    root = Path(repository or repository_root()).resolve()
    resolved_stage = resolve_stage(root, stage)
    manifest, providers = _validate_manifest(
        root, resolved_stage, require_target_providers=require_target_providers
    )
    ldd = _verify_ldd(resolved_stage, providers, manifest["runtime_libraries"])
    identity = _manifest_identity(resolved_stage / "manifest.json", manifest)
    identity.update(
        {
            "status": "verified",
            "manifest_path": _relative(
                resolved_stage / "manifest.json", root, "CUDA runtime manifest"
            ),
            "runtime_files_verified": len(manifest["runtime_libraries"])
            + len(manifest["runtime_symlinks"]),
            "ldd": ldd,
        }
    )
    return identity


def _publish_target_providers(repository: Path, stage: Path) -> None:
    release = repository / "target/release"
    release.mkdir(parents=True, exist_ok=True)
    for name in ORT_PROVIDER_SHA256:
        source = stage / "ort" / name
        destination = release / name
        temporary = release / f".{name}.alphamini-runtime-{os.getpid()}"
        try:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()
            shutil.copyfile(source, temporary)
            temporary.chmod(source.stat().st_mode & 0o777)
            os.replace(temporary, destination)
        finally:
            if temporary.exists() or temporary.is_symlink():
                temporary.unlink()


def setup_runtime(
    repository: Path | str | None = None,
    stage: Path | str | None = None,
    *,
    python: Path | str | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    _validate_platform()
    root = Path(repository or repository_root()).resolve()
    resolved_stage = resolve_stage(root, stage)
    requirements_path = root / REQUIREMENTS_RELATIVE
    requirements = parse_requirements(requirements_path)
    if resolved_stage.exists() and not replace:
        verify_runtime(root, resolved_stage, require_target_providers=False)
        _publish_target_providers(root, resolved_stage)
        return verify_runtime(root, resolved_stage)
    resolved_stage.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{resolved_stage.name}.tmp-", dir=resolved_stage.parent)
    )
    backup: Path | None = None
    try:
        provider_sources, ort_identity = _fresh_provider_sources(root)
        installer = _select_python(root, python)
        _install_wheels(root, temporary, installer, requirements_path)
        _ensure_cudnn_loader_name(temporary)
        _metadata_packages(temporary, requirements)
        provider_directory = temporary / "ort"
        provider_directory.mkdir()
        for name, source in provider_sources.items():
            shutil.copyfile(source, provider_directory / name)
            (provider_directory / name).chmod(source.stat().st_mode & 0o777)
        providers = _provider_records(temporary)
        libraries, symlinks = _runtime_files(temporary)
        wrapper = _write_wrapper(temporary)
        manifest = {
            "schema": SCHEMA,
            "platform": {
                "cargo_target": "x86_64-unknown-linux-gnu",
                "machine": "x86_64",
                "system": "Linux",
            },
            "cuda_release": CUDA_RELEASE,
            "loader_directories": [path.as_posix() for path in LOADER_DIRECTORIES],
            "requirements": {
                "path": REQUIREMENTS_RELATIVE.as_posix(),
                "sha256": _sha256(requirements_path),
                "packages": requirements,
            },
            "runtime_libraries": libraries,
            "runtime_symlinks": symlinks,
            "ort": {**ort_identity, "providers": providers},
            "wrapper": wrapper,
        }
        (temporary / "manifest.json").write_bytes(_json_bytes(manifest))
        verify_runtime(root, temporary, require_target_providers=False)
        if resolved_stage.exists():
            backup = resolved_stage.with_name(
                f".{resolved_stage.name}.backup-{os.getpid()}"
            )
            if backup.exists():
                raise CudaRuntimeError(f"refusing to overwrite CUDA runtime backup: {backup}")
            os.replace(resolved_stage, backup)
        os.replace(temporary, resolved_stage)
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
        _publish_target_providers(root, resolved_stage)
        return verify_runtime(root, resolved_stage)
    except Exception:
        if backup is not None and backup.exists() and not resolved_stage.exists():
            os.replace(backup, resolved_stage)
            backup = None
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def runtime_environment(
    repository: Path | str | None = None, stage: Path | str | None = None
) -> tuple[dict[str, str], dict[str, Any]]:
    root = Path(repository or repository_root()).resolve()
    resolved_stage = resolve_stage(root, stage)
    identity = verify_runtime(root, resolved_stage)
    env = _loader_environment(resolved_stage)
    env["CARGO_TARGET_DIR"] = str(root / "target")
    return env, identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage and verify AlphaMini's isolated Rust ORT CUDA 13.2 runtime"
    )
    parser.add_argument("--repository", type=Path, default=repository_root())
    parser.add_argument("--stage", type=Path)
    subparsers = parser.add_subparsers(dest="action", required=True)
    setup = subparsers.add_parser("setup", help="build and install the pinned isolated stage")
    setup.add_argument("--python", type=Path)
    setup.add_argument("--replace", action="store_true")
    subparsers.add_parser("verify", help="rehash providers/runtime and run fail-closed ldd")
    execute = subparsers.add_parser("exec", help="verify, scope the loader environment, and exec")
    execute.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parser = _parser()
    options = parser.parse_args(arguments)
    try:
        if options.action == "setup":
            result = setup_runtime(
                options.repository,
                options.stage,
                python=options.python,
                replace=options.replace,
            )
        elif options.action == "verify":
            result = verify_runtime(options.repository, options.stage)
        else:
            command = list(options.command)
            if command and command[0] == "--":
                command.pop(0)
            if not command:
                parser.error("exec requires a command after --")
            env, _ = runtime_environment(options.repository, options.stage)
            os.execvpe(command[0], command, env)
            raise AssertionError("os.execvpe returned")
    except (CudaRuntimeError, OSError) as error:
        print(f"alphamini CUDA runtime: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
