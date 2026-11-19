//! Private staging, bounded recovery state and atomic no-overwrite publication.

use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::ops::ControlFlow;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{MigrationCheckpoint, MigrationFailure as Failure, RECOVERY_DIR, Result, checkpoint};
use crate::registry_discovery::{EntryKind, FileEntry, MigrationPlan, TransferState};

const MAX_STATE_BYTES: u64 = 4096;
const STATE_FILE: &str = "state.json";
const PENDING_FILE: &str = "pending";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    Copying,
    Completed,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Journal {
    protocol: String,
    plan_revision: String,
    pub(super) phase: Phase,
    staging: Option<usize>,
}

impl Journal {
    pub(super) fn load(plan: &MigrationPlan) -> Result<Option<Self>> {
        let root = plan.destination.join(RECOVERY_DIR);
        match std::fs::symlink_metadata(&root) {
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Ok(meta) if meta.is_dir() && !meta.file_type().is_symlink() => {}
            _ => return Err(Failure::InvalidRecovery),
        }
        let mut names = Vec::new();
        for entry in std::fs::read_dir(&root).map_err(|_| Failure::InvalidRecovery)? {
            let entry = entry.map_err(|_| Failure::InvalidRecovery)?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| Failure::InvalidRecovery)?;
            let meta =
                std::fs::symlink_metadata(entry.path()).map_err(|_| Failure::InvalidRecovery)?;
            if !meta.is_file()
                || meta.file_type().is_symlink()
                || !(name == STATE_FILE
                    || name == PENDING_FILE
                    || name.starts_with(".state.json.tmp."))
                || names.len() >= 32
            {
                return Err(Failure::InvalidRecovery);
            }
            names.push(name);
        }
        if names.is_empty() {
            return Ok(None);
        }
        let mut bytes = Vec::new();
        File::open(root.join(STATE_FILE))
            .and_then(|file| file.take(MAX_STATE_BYTES + 1).read_to_end(&mut bytes))
            .map_err(|_| Failure::InvalidRecovery)?;
        if bytes.len() as u64 > MAX_STATE_BYTES {
            return Err(Failure::InvalidRecovery);
        }
        let journal: Self = serde_json::from_slice(&bytes).map_err(|_| Failure::InvalidRecovery)?;
        if journal.protocol != "voqx-registry-migration-state/1"
            || Some(&journal.plan_revision) != plan.revision.as_ref()
            || (journal.staging.is_none() && names.iter().any(|name| name == PENDING_FILE))
            || (journal.phase == Phase::Completed && journal.staging.is_some())
        {
            return Err(Failure::InvalidRecovery);
        }
        if let Some(index) = journal.staging
            && !plan.entries.get(index).is_some_and(|entry| {
                entry.state == TransferState::Copy
                    && entry
                        .source
                        .as_ref()
                        .is_some_and(|file| file.kind == EntryKind::File)
            })
        {
            return Err(Failure::InvalidRecovery);
        }
        Ok(Some(journal))
    }

    pub(super) fn begin(plan: &MigrationPlan) -> Result<Self> {
        private_dir(&plan.destination.join(RECOVERY_DIR))?;
        let journal = Self {
            protocol: "voqx-registry-migration-state/1".into(),
            plan_revision: plan.revision.clone().ok_or(Failure::InvalidPlan)?,
            phase: Phase::Copying,
            staging: None,
        };
        journal.save(plan)?;
        Ok(journal)
    }

    pub(super) fn save(&self, plan: &MigrationPlan) -> Result<()> {
        let bytes = serde_json::to_vec(self).map_err(|_| Failure::InvalidRecovery)?;
        let path = plan.destination.join(RECOVERY_DIR).join(STATE_FILE);
        crate::atomic::write_atomic(&path, &bytes).map_err(|_| Failure::Io)?;
        sync_parent(&path)
    }

    pub(super) fn prepare_resume(&mut self, plan: &MigrationPlan) -> Result<()> {
        if self.staging.is_some() {
            let path = plan.destination.join(RECOVERY_DIR).join(PENDING_FILE);
            match std::fs::symlink_metadata(&path) {
                Ok(meta) if meta.is_file() && !meta.file_type().is_symlink() => {
                    // Only the journal-declared, private unfinished staging file.
                    std::fs::remove_file(&path).map_err(|_| Failure::Io)?;
                    sync_parent(&path)?;
                }
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                _ => return Err(Failure::InvalidRecovery),
            }
        }
        self.staging = None;
        self.phase = Phase::Copying;
        self.save(plan)
    }

    pub(super) fn publish(
        &mut self,
        plan: &MigrationPlan,
        entry: &FileEntry,
        progress: &mut impl FnMut(MigrationCheckpoint) -> ControlFlow<()>,
        registry: bool,
    ) -> Result<()> {
        let index = plan
            .entries
            .iter()
            .position(|item| item.source.as_ref() == Some(entry))
            .ok_or(Failure::InvalidPlan)?;
        self.staging = Some(index);
        self.save(plan)?;
        let pending = plan.destination.join(RECOVERY_DIR).join(PENDING_FILE);
        stage(&plan.source.root.join(&entry.path), &pending, entry)?;
        checkpoint(progress, MigrationCheckpoint::Staged(index))?;
        let metadata =
            std::fs::symlink_metadata(&pending).map_err(|_| Failure::VerificationFailed)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(Failure::VerificationFailed);
        }
        let (_, actual_hash) =
            crate::registry_discovery::inventory::hash_file(&pending, entry.bytes, entry.bytes)
                .map_err(|_| Failure::VerificationFailed)?;
        if entry.sha256.as_ref() != Some(&actual_hash) {
            return Err(Failure::VerificationFailed);
        }
        let target = plan.destination.join(&entry.path);
        // Same-volume hard linking is atomic and refuses an existing final path.
        std::fs::hard_link(&pending, &target).map_err(|error| {
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                Failure::DestinationChanged
            } else {
                Failure::Io
            }
        })?;
        sync_parent(&target)?;
        std::fs::remove_file(&pending).map_err(|_| Failure::Io)?;
        sync_parent(&pending)?;
        self.staging = None;
        self.save(plan)?;
        if !registry {
            checkpoint(progress, MigrationCheckpoint::Published(index))?;
        }
        Ok(())
    }
}

pub(super) fn private_dir(path: &Path) -> Result<()> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder.create(path).map_err(|_| Failure::Io)?;
    let meta = std::fs::symlink_metadata(path).map_err(|_| Failure::Io)?;
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return Err(Failure::Io);
    }
    Ok(())
}

fn stage(source: &Path, pending: &Path, expected: &FileEntry) -> Result<()> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut output = options.open(pending).map_err(|_| Failure::Io)?;
    let mut input = File::open(source)
        .map_err(|_| Failure::Io)?
        .take(expected.bytes.saturating_add(1));
    let mut hash = Sha256::new();
    let mut bytes = 0_u64;
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = input.read(&mut buffer).map_err(|_| Failure::Io)?;
        if read == 0 {
            break;
        }
        bytes += read as u64;
        if bytes > expected.bytes {
            return Err(Failure::VerificationFailed);
        }
        output.write_all(&buffer[..read]).map_err(|_| Failure::Io)?;
        hash.update(&buffer[..read]);
    }
    if bytes != expected.bytes || Some(format!("{:x}", hash.finalize())) != expected.sha256 {
        return Err(Failure::VerificationFailed);
    }
    output.sync_all().map_err(|_| Failure::Io)?;
    sync_parent(pending)
}

fn sync_parent(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(path.parent().ok_or(Failure::Io)?)
            .and_then(|file| file.sync_all())
            .map_err(|_| Failure::Io)?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}
