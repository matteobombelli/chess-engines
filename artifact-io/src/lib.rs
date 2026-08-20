//! Shared primitives for content-addressed, immutable artifacts.
//!
//! Publication stages complete bytes in the destination directory, syncs the
//! staged file, and uses a hard link as an atomic no-replace operation. The API
//! intentionally accepts complete byte slices: callers cannot observe or leak
//! a temporary path, and a successful destination is never overwritten.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

static TEMPORARY_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Lowercase SHA-256 of `bytes`.
pub fn sha256_bytes(bytes: &[u8]) -> String {
    hex_digest(&Sha256::digest(bytes))
}

/// Lowercase SHA-256 of the exact bytes stored at `path`.
pub fn sha256_file(path: impl AsRef<Path>) -> io::Result<String> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(hex_digest(&digest.finalize()))
}

/// Publish `bytes` at a previously absent path without ever replacing a file.
///
/// The temporary file is created in the destination directory, making the
/// final hard link a same-filesystem, atomic no-replace operation. A racing
/// winner is returned as [`io::ErrorKind::AlreadyExists`].
pub fn publish_bytes_new(path: impl AsRef<Path>, bytes: &[u8]) -> io::Result<()> {
    let path = path.as_ref();
    let parent = nonempty_parent(path);
    fs::create_dir_all(parent)?;
    if path.try_exists()? {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("immutable artifact {} already exists", path.display()),
        ));
    }

    let temporary = unique_temporary_path(path)?;
    let guard = TemporaryGuard(temporary.clone());
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);

    // Unlike rename on Unix, hard_link never replaces an existing destination.
    fs::hard_link(&temporary, path)?;
    fs::remove_file(&temporary)?;
    drop(guard);
    sync_parent_directory(parent)
}

/// Publish immutable bytes, accepting an existing destination only when its
/// length and checksum exactly match. This makes crash retries idempotent while
/// still rejecting conflicting writers.
pub fn publish_bytes_idempotent(path: impl AsRef<Path>, bytes: &[u8]) -> io::Result<()> {
    let path = path.as_ref();
    match publish_bytes_new(path, bytes) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            let metadata = fs::metadata(path)?;
            if metadata.len() == bytes.len() as u64 && sha256_file(path)? == sha256_bytes(bytes) {
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::AlreadyExists,
                    format!(
                        "existing immutable artifact {} differs from retry bytes",
                        path.display()
                    ),
                ))
            }
        }
        Err(error) => Err(error),
    }
}

fn nonempty_parent(path: &Path) -> &Path {
    path.parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."))
}

fn unique_temporary_path(path: &Path) -> io::Result<PathBuf> {
    let name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("artifact path {} needs a file name", path.display()),
        )
    })?;
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    Ok(nonempty_parent(path).join(format!(
        ".{}.{}.{}.{}.partial",
        name.to_string_lossy(),
        std::process::id(),
        suffix,
        TEMPORARY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    )))
}

fn sync_parent_directory(parent: &Path) -> io::Result<()> {
    File::open(parent)?.sync_all()
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(char::from(HEX[usize::from(byte >> 4)]));
        output.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    output
}

struct TemporaryGuard(PathBuf);

impl Drop for TemporaryGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::*;

    #[test]
    fn sha256_has_frozen_byte_and_file_vectors() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("abc.bin");
        fs::write(&path, b"abc").unwrap();
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert_eq!(sha256_bytes(b"abc"), expected);
        assert_eq!(sha256_file(path).unwrap(), expected);
    }

    #[test]
    fn racing_publishers_never_clobber_the_winner() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("winner.bin");
        let barrier = Arc::new(Barrier::new(4));
        let handles: Vec<_> = (0_u8..4)
            .map(|value| {
                let output = output.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    let bytes = vec![value; 32];
                    barrier.wait();
                    (bytes.clone(), publish_bytes_new(output, &bytes).is_ok())
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();
        assert_eq!(results.iter().filter(|(_, won)| *won).count(), 1);
        let winner = results.iter().find(|(_, won)| *won).unwrap();
        assert_eq!(fs::read(output).unwrap(), winner.0);
    }

    #[test]
    fn retries_accept_only_identical_bytes() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("artifact.bin");
        publish_bytes_idempotent(&output, b"first").unwrap();
        publish_bytes_idempotent(&output, b"first").unwrap();
        let error = publish_bytes_idempotent(&output, b"second").unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::AlreadyExists);
        assert_eq!(fs::read(output).unwrap(), b"first");
    }

    #[test]
    fn same_bytes_are_idempotent_under_a_race() {
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("artifact.bin");
        let barrier = Arc::new(Barrier::new(2));
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let output = output.clone();
                let barrier = Arc::clone(&barrier);
                thread::spawn(move || {
                    barrier.wait();
                    publish_bytes_idempotent(output, b"same")
                })
            })
            .collect();
        for handle in handles {
            handle.join().unwrap().unwrap();
        }
        assert_eq!(fs::read(output).unwrap(), b"same");
    }
}
