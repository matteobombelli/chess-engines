from __future__ import annotations

import shutil
from pathlib import Path

import pytest

import alphamini_train.cuda_runtime as cuda_runtime

from conftest import REPOSITORY


def test_reviewed_requirements_are_exact_and_hash_locked() -> None:
    requirements = cuda_runtime.parse_requirements(
        REPOSITORY / cuda_runtime.REQUIREMENTS_RELATIVE
    )
    assert {item["name"]: item["version"] for item in requirements} == (
        cuda_runtime.PACKAGE_VERSIONS
    )
    assert all(len(item["wheel_sha256"]) == 64 for item in requirements)


def test_stage_must_be_isolated_beneath_cargo_target(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    expected = repository / cuda_runtime.STAGE_RELATIVE
    assert cuda_runtime.resolve_stage(repository) == expected
    with pytest.raises(cuda_runtime.CudaRuntimeError, match="must be beneath"):
        cuda_runtime.resolve_stage(repository, repository / "alphamini-train/.venv")
    with pytest.raises(cuda_runtime.CudaRuntimeError, match="cannot overlay"):
        cuda_runtime.resolve_stage(repository, repository / "target")


def _fixture_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, dict[str, Path]]:
    repository = tmp_path / "repository"
    requirements_path = repository / cuda_runtime.REQUIREMENTS_RELATIVE
    requirements_path.parent.mkdir(parents=True)
    shutil.copyfile(REPOSITORY / cuda_runtime.REQUIREMENTS_RELATIVE, requirements_path)
    requirements = cuda_runtime.parse_requirements(requirements_path)
    stage = repository / cuda_runtime.STAGE_RELATIVE
    cuda_directory = stage / "nvidia/cu13/lib"
    cudnn_directory = stage / "nvidia/cudnn/lib"
    cuda_directory.mkdir(parents=True)
    cudnn_directory.mkdir(parents=True)
    runtime_paths: dict[str, Path] = {}
    for name in cuda_runtime.REQUIRED_STAGE_SONAMES:
        directory = cudnn_directory if name.startswith("libcudnn") else cuda_directory
        path = directory / name
        if name == "libcudnn.so":
            path.symlink_to("libcudnn.so.9")
        else:
            path.write_bytes(f"fixture:{name}\n".encode())
        runtime_paths[name] = path
    for package in requirements:
        metadata = stage / f"{package['name'].replace('-', '_')}-{package['version']}.dist-info"
        metadata.mkdir()
        (metadata / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {package['name']}\nVersion: {package['version']}\n"
        )
    provider_directory = stage / "ort"
    provider_directory.mkdir()
    provider_hashes: dict[str, str] = {}
    for name in cuda_runtime.ORT_PROVIDER_SHA256:
        path = provider_directory / name
        path.write_bytes(f"fixture:{name}\n".encode())
        provider_hashes[name] = cuda_runtime._sha256(path)
    monkeypatch.setattr(cuda_runtime, "ORT_PROVIDER_SHA256", provider_hashes)
    providers = cuda_runtime._provider_records(stage)
    libraries, symlinks = cuda_runtime._runtime_files(stage)
    wrapper = cuda_runtime._write_wrapper(stage)
    manifest = {
        "schema": cuda_runtime.SCHEMA,
        "platform": {
            "cargo_target": "x86_64-unknown-linux-gnu",
            "machine": "x86_64",
            "system": "Linux",
        },
        "cuda_release": cuda_runtime.CUDA_RELEASE,
        "loader_directories": [
            path.as_posix() for path in cuda_runtime.LOADER_DIRECTORIES
        ],
        "requirements": {
            "path": cuda_runtime.REQUIREMENTS_RELATIVE.as_posix(),
            "sha256": cuda_runtime._sha256(requirements_path),
            "packages": requirements,
        },
        "runtime_libraries": libraries,
        "runtime_symlinks": symlinks,
        "ort": {
            "crate_version": cuda_runtime.ORT_CRATE_VERSION,
            "runtime_version": cuda_runtime.ORT_RUNTIME_VERSION,
            "distribution_sha256": cuda_runtime.ORT_DISTRIBUTION_SHA256,
            "static_archive_sha256": cuda_runtime.ORT_STATIC_ARCHIVE_SHA256,
            "providers": providers,
        },
        "wrapper": wrapper,
    }
    (stage / "manifest.json").write_bytes(cuda_runtime._json_bytes(manifest))
    release = repository / "target/release"
    release.mkdir()
    for provider in providers:
        shutil.copyfile(stage / provider["path"], release / provider["name"])
    monkeypatch.setattr(cuda_runtime, "_verify_ldd", lambda *_: {"fixture": "passed"})
    return repository, stage, runtime_paths


def test_verify_rehashes_runtime_and_records_rust_ort_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, stage, runtime_paths = _fixture_stage(tmp_path, monkeypatch)
    result = cuda_runtime.verify_runtime(repository, stage)
    assert result["status"] == "verified"
    assert result["requirements_sha256"] == cuda_runtime._sha256(
        repository / cuda_runtime.REQUIREMENTS_RELATIVE
    )
    assert result["ort"]["crate_version"] == "2.0.0-rc.13"
    assert {item["name"] for item in result["ort"]["providers"]} == set(
        cuda_runtime.ORT_PROVIDER_SHA256
    )
    recorded = cuda_runtime.recorded_runtime_identity(repository)
    assert recorded["status"] == "recorded"
    assert recorded["manifest_sha256"] == result["manifest_sha256"]
    assert recorded["ort"]["providers"] == result["ort"]["providers"]

    runtime_paths["libcudart.so.13"].write_bytes(b"tampered\n")
    with pytest.raises(cuda_runtime.CudaRuntimeError, match="hashes or symlinks differ"):
        cuda_runtime.verify_runtime(repository, stage)


def test_verify_rejects_release_provider_cache_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, stage, _ = _fixture_stage(tmp_path, monkeypatch)
    name = "libonnxruntime_providers_cuda.so"
    release_provider = repository / "target/release" / name
    release_provider.unlink()
    release_provider.symlink_to(stage / "ort" / name)
    with pytest.raises(cuda_runtime.CudaRuntimeError, match="cache symlink"):
        cuda_runtime.verify_runtime(repository, stage)


def test_generated_environment_wrapper_is_manifest_bound_and_stage_relative(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, stage, _ = _fixture_stage(tmp_path, monkeypatch)
    content = (stage / "with-alphamini-cuda").read_text()
    assert content == cuda_runtime._wrapper_content()
    assert 'repository_root=$(CDPATH= cd -- "$runtime_dir/../../.." && pwd)' in content
    assert 'export CARGO_TARGET_DIR="$repository_root/target"' in content
