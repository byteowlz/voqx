//! Bounded, deterministic inventory. No filesystem mutation or link traversal.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Component, Path};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Maximum entries in one root inventory.
pub const MAX_ENTRIES: usize = 10_000;
/// Maximum directory nesting below a root.
const MAX_DEPTH: usize = 32;
/// Maximum bytes hashed per root (8 gibibytes), including all assets.
const MAX_BYTES: u64 = 8 * 1024 * 1024 * 1024;

/// Why a tree cannot be safely inventoried. No sensitive path is echoed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ScanIssue {
    /// Filesystem read failed.
    Unavailable,
    /// Symlink, special file or non-directory root.
    UnsafeTree,
    /// Entry count, depth, byte or portable-path bound exceeded.
    LimitExceeded,
    /// File metadata/length changed while hashing.
    Changed,
    /// Writer lock observed.
    Busy,
}

/// Tree-entry type; symlinks and special files are never accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum EntryKind {
    /// Directory, including empty asset directories.
    Directory,
    /// Regular file.
    File,
}

/// Content-preservation receipt for one registry-relative entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    /// Portable registry-relative path; may expose legacy names. Keep private.
    pub path: String,
    /// Entry type.
    pub kind: EntryKind,
    /// Exact file byte count; zero for directories.
    pub bytes: u64,
    /// SHA-256 of exact bytes; absent for directories.
    pub sha256: Option<String>,
}

pub(super) type Inventory = BTreeMap<String, FileEntry>;

pub(super) fn scan(root: &Path) -> Result<Inventory, ScanIssue> {
    scan_inner(root, false, false)
}

pub(super) fn scan_held(root: &Path, recovery: bool) -> Result<Inventory, ScanIssue> {
    scan_inner(root, true, recovery)
}

#[derive(Debug)]
struct Scan {
    entries: Inventory,
    bytes: u64,
    held: bool,
    recovery: bool,
}

impl Scan {
    fn skips(&self, key: &str) -> bool {
        (self.held && key == ".lock") || (self.recovery && key == super::apply::RECOVERY_DIR)
    }

    fn record(&mut self, path: &Path, key: String) -> Result<bool, ScanIssue> {
        if self.entries.len() >= MAX_ENTRIES {
            return Err(ScanIssue::LimitExceeded);
        }
        let metadata = std::fs::symlink_metadata(path).map_err(|_| ScanIssue::Unavailable)?;
        if metadata.file_type().is_symlink() {
            return Err(ScanIssue::UnsafeTree);
        }
        let entry = if metadata.is_dir() {
            FileEntry {
                path: key.clone(),
                kind: EntryKind::Directory,
                bytes: 0,
                sha256: None,
            }
        } else if metadata.is_file() {
            let remaining = MAX_BYTES
                .checked_sub(self.bytes)
                .ok_or(ScanIssue::LimitExceeded)?;
            if metadata.len() > remaining {
                return Err(ScanIssue::LimitExceeded);
            }
            let (size, hash) = hash_file(path, metadata.len(), remaining)?;
            self.bytes += size;
            FileEntry {
                path: key.clone(),
                kind: EntryKind::File,
                bytes: size,
                sha256: Some(hash),
            }
        } else {
            return Err(ScanIssue::UnsafeTree);
        };
        let directory = entry.kind == EntryKind::Directory;
        self.entries.insert(key, entry);
        Ok(directory)
    }
}

fn scan_inner(root: &Path, held: bool, recovery: bool) -> Result<Inventory, ScanIssue> {
    let mut scan = Scan {
        entries: Inventory::new(),
        bytes: 0,
        held,
        recovery,
    };
    match std::fs::symlink_metadata(root) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(scan.entries),
        Err(_) => return Err(ScanIssue::Unavailable),
        Ok(metadata) if !metadata.is_dir() || metadata.file_type().is_symlink() => {
            return Err(ScanIssue::UnsafeTree);
        }
        Ok(_) => {}
    }
    if !held {
        super::check_unlocked(root).map_err(|_| ScanIssue::Busy)?;
    }
    walk(root, root, 0, &mut scan)?;
    if !held {
        super::check_unlocked(root).map_err(|_| ScanIssue::Busy)?;
    }
    Ok(scan.entries)
}

fn walk(root: &Path, dir: &Path, depth: usize, scan: &mut Scan) -> Result<(), ScanIssue> {
    if depth > MAX_DEPTH {
        return Err(ScanIssue::LimitExceeded);
    }
    for child in std::fs::read_dir(dir).map_err(|_| ScanIssue::Unavailable)? {
        let path = child.map_err(|_| ScanIssue::Unavailable)?.path();
        let relative = path.strip_prefix(root).map_err(|_| ScanIssue::UnsafeTree)?;
        let key = portable_path(relative).ok_or(ScanIssue::UnsafeTree)?;
        if scan.skips(&key) {
            continue;
        }
        if key == ".lock" {
            return Err(ScanIssue::Busy);
        }
        if scan.record(&path, key)? {
            walk(root, &path, depth + 1, scan)?;
        }
    }
    Ok(())
}

pub(super) fn hash_file(
    path: &Path,
    expected: u64,
    remaining: u64,
) -> Result<(u64, String), ScanIssue> {
    let file = std::fs::File::open(path).map_err(|_| ScanIssue::Unavailable)?;
    let before = file.metadata().map_err(|_| ScanIssue::Unavailable)?;
    let mut reader = file.take(remaining + 1);
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 16 * 1024];
    let mut size = 0_u64;
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|_| ScanIssue::Unavailable)?;
        if count == 0 {
            break;
        }
        size += count as u64;
        if size > remaining {
            return Err(ScanIssue::LimitExceeded);
        }
        hasher.update(&buffer[..count]);
    }
    let after = std::fs::symlink_metadata(path).map_err(|_| ScanIssue::Unavailable)?;
    if size != expected
        || before.len() != after.len()
        || before.modified().ok() != after.modified().ok()
        || !after.is_file()
        || after.file_type().is_symlink()
    {
        return Err(ScanIssue::Changed);
    }
    Ok((size, format!("{:x}", hasher.finalize())))
}

pub(super) fn portable_path(path: &Path) -> Option<String> {
    if path.as_os_str().is_empty() {
        return None;
    }
    let mut parts = Vec::new();
    for part in path.components() {
        let Component::Normal(part) = part else {
            return None;
        };
        let part = part.to_str()?;
        // Reject Windows separators/prefixes even when running on Unix.
        if part.contains(['\\', ':']) || part.chars().any(char::is_control) {
            return None;
        }
        parts.push(part);
    }
    Some(parts.join("/"))
}
