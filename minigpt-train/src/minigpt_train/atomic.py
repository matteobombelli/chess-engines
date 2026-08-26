"""Small, auditable durability primitives used by every run-state mutation."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import socket
import tempfile
from pathlib import Path
from typing import Any, Iterator

from .errors import IntegrityError, RunLockedError

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def free_bytes(path: Path) -> int:
    """Space available to this user on the filesystem holding `path`."""

    status = os.statvfs(path)
    return int(status.f_bavail) * int(status.f_frsize)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path: Path, value: bytes, mode: int = 0o644) -> None:
    """Write, fsync, rename, then fsync the containing directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        fsync_directory(path.parent)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value))


def append_jsonl(path: Path, value: Any) -> None:
    """Append one durable line; the log is replayable but never rewritten."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def read_json(path: Path) -> Any:
    try:
        with path.open("rb") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise IntegrityError(f"cannot read JSON {path}: {error}") from error


class ObjectStore:
    """Content-addressed immutable JSON objects."""

    def __init__(self, root: Path):
        self.root = root

    def path_for(self, digest: str) -> Path:
        if not SHA256_RE.fullmatch(digest):
            raise IntegrityError(f"invalid object hash: {digest!r}")
        return self.root / digest[:2] / f"{digest}.json"

    def put(self, value: Any) -> str:
        payload = canonical_json_bytes(value)
        digest = sha256_bytes(payload)
        path = self.path_for(digest)
        if path.exists():
            if path.read_bytes() != payload:
                raise IntegrityError(f"content-address collision at {path}")
            return digest
        atomic_write_bytes(path, payload, mode=0o444)
        return digest

    def get(self, digest: str) -> Any:
        path = self.path_for(digest)
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise IntegrityError(f"missing object {digest}: {error}") from error
        if sha256_bytes(payload) != digest:
            raise IntegrityError(f"object checksum mismatch: {path}")
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise IntegrityError(f"invalid object JSON: {path}") from error


def write_pointer(path: Path, digest: str) -> None:
    if not SHA256_RE.fullmatch(digest):
        raise IntegrityError(f"refusing invalid pointer hash: {digest!r}")
    atomic_write_bytes(path, f"{digest}\n".encode("ascii"))


def read_pointer(path: Path) -> str:
    try:
        digest = path.read_text(encoding="ascii").strip()
    except OSError as error:
        raise IntegrityError(f"cannot read pointer {path}: {error}") from error
    if not SHA256_RE.fullmatch(digest):
        raise IntegrityError(f"malformed pointer {path}")
    return digest


class AdvisoryLock:
    """Non-blocking POSIX advisory lock with owner diagnostics."""

    def __init__(self, path: Path):
        self.path = path
        self._handle = None

    def __enter__(self) -> "AdvisoryLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._handle.seek(0)
            owner = self._handle.read().strip() or "unknown owner"
            self._handle.close()
            self._handle = None
            raise RunLockedError(f"run is locked by {owner}") from error
        self._handle.seek(0)
        self._handle.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname()}, self._handle)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return self

    def __exit__(self, *_: object) -> None:
        if self._handle is not None:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            self._handle.close()
            self._handle = None


@contextlib.contextmanager
def locked(path: Path) -> Iterator[None]:
    with AdvisoryLock(path):
        yield
