//! Atomic file replacement and a cross-process advisory lock.
//!
//! `registry.json` is the single authoritative file, so writes must be
//! all-or-nothing and concurrent writers must serialize. We write to a sibling
//! temp file and rename it over the target (atomic on a single filesystem), and
//! guard mutation with a lock file created via `create_new`.

use std::fs::{self, File, OpenOptions};
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::thread::sleep;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};

/// Write `bytes` to `path` atomically via a temp file and rename.
///
/// # Errors
///
/// Returns an error if the parent cannot be created or the write/rename fails.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
    fs::create_dir_all(&parent)
        .with_context(|| format!("creating directory {}", parent.display()))?;

    let tmp = parent.join(format!(
        ".{}.tmp.{}.{}",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("registry.json"),
        std::process::id(),
        nanos()
    ));

    // Scope the file so it is flushed and closed before the rename.
    {
        let mut file =
            File::create(&tmp).with_context(|| format!("creating temp file {}", tmp.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("writing temp file {}", tmp.display()))?;
        file.sync_all()
            .with_context(|| format!("flushing temp file {}", tmp.display()))?;
    }

    fs::rename(&tmp, path).map_err(|err| {
        let _ = fs::remove_file(&tmp);
        anyhow::Error::new(err).context(format!("replacing {}", path.display()))
    })
}

/// A held advisory lock, released on drop.
#[derive(Debug)]
pub struct LockGuard {
    path: PathBuf,
}

impl LockGuard {
    /// Acquire the lock at `lock_path`, retrying until `timeout` elapses.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock cannot be acquired within `timeout` (another
    /// holder is still active), or the parent directory cannot be created.
    pub fn acquire(lock_path: &Path, timeout: Duration) -> Result<Self> {
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating lock directory {}", parent.display()))?;
        }
        let start = Instant::now();
        loop {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(lock_path)
            {
                Ok(_) => {
                    return Ok(Self {
                        path: lock_path.to_path_buf(),
                    });
                }
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
                    if start.elapsed() >= timeout {
                        bail!(
                            "voice registry is locked ({}); another process may be writing",
                            lock_path.display()
                        );
                    }
                    sleep(Duration::from_millis(25));
                }
                Err(err) => {
                    return Err(anyhow::Error::new(err)
                        .context(format!("acquiring lock {}", lock_path.display())));
                }
            }
        }
    }
}

impl Drop for LockGuard {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn nanos() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos())
}

#[cfg(test)]
mod tests {
    use super::{LockGuard, write_atomic};
    use std::time::Duration;

    fn temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "voqx-test-{}-{}",
            std::process::id(),
            super::nanos()
        ));
        let _ = std::fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn atomic_write_roundtrips() {
        let dir = temp_dir();
        let path = dir.join("registry.json");
        write_atomic(&path, b"hello").expect("first write");
        write_atomic(&path, b"world").expect("replacing write");
        let body = std::fs::read_to_string(&path).expect("reading back");
        assert_eq!(body, "world");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn lock_is_exclusive_then_reusable() {
        let dir = temp_dir();
        let lock = dir.join(".lock");
        let held = LockGuard::acquire(&lock, Duration::from_millis(50)).expect("first acquire");
        assert!(LockGuard::acquire(&lock, Duration::from_millis(50)).is_err());
        drop(held);
        let again = LockGuard::acquire(&lock, Duration::from_millis(50));
        assert!(again.is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }
}
