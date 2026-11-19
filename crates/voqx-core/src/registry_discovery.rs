//! Read-only registry discovery. No config initialization, inference or migration.
//!
//! Roots are selected exactly as before: an explicit override, otherwise the
//! legacy default. The proposed identity root is inspected, never auto-selected.

mod apply;
mod inventory;
mod migration;
mod readiness;
#[cfg(test)]
mod tests;

use std::io::Read;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{REGISTRY_VERSION, VoiceRegistry};

pub use apply::{
    MigrationAction, MigrationCheckpoint, MigrationFailure, MigrationOutcome, MigrationReceipt,
    execute_migration,
};
pub use inventory::{EntryKind, FileEntry, ScanIssue};
pub use migration::{MigrationPlan, PlanBlocker, PlanEntry, PlanState, TransferState};
pub use readiness::{ProfileState, RecognitionReadiness};

/// Maximum registry JSON bytes accepted by discovery, before parsing.
pub const MAX_REGISTRY_BYTES: u64 = 16 * 1024 * 1024;

/// Observed registry-file state, not recognition or authorization readiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RegistryState {
    /// Registry JSON parsed at the supported format version.
    Ready,
    /// Root or registry file is absent; no empty registry is fabricated.
    Missing,
    /// A registry writer lock was observed. No lock is acquired by discovery.
    Busy,
    /// Malformed JSON or invalid registry structure.
    Invalid,
    /// Registry format differs from the supported version.
    UnsupportedVersion,
    /// Reading failed; not equivalent to missing or empty.
    Unavailable,
    /// Root/file is a symlink or not a regular directory/file.
    UnsafePath,
    /// Registry exceeds the bounded discovery size.
    LimitExceeded,
}

/// A bounded metadata-only view; no names, IDs, vectors or media are returned.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RegistryProbe {
    /// Caller-selected root, which may be private.
    pub root: PathBuf,
    /// Observed state.
    pub state: RegistryState,
    /// Parsed format version, if available.
    pub version: Option<u32>,
    /// SHA-256 of the exact JSON bytes, only for a successfully parsed registry.
    pub revision: Option<String>,
    /// Number of identities, only for a successfully parsed registry.
    pub identity_count: Option<usize>,
}

impl RegistryProbe {
    /// Inspect without creating directories, locks, config or defaults.
    #[must_use]
    pub fn inspect(root: PathBuf) -> Self {
        let mut probe = Self {
            root,
            state: RegistryState::Missing,
            version: None,
            revision: None,
            identity_count: None,
        };
        match read_registry(&probe.root) {
            Ok((registry, bytes)) => {
                probe.state = RegistryState::Ready;
                probe.version = Some(registry.version);
                probe.revision = Some(digest(&bytes));
                probe.identity_count = Some(registry.identities.len());
            }
            Err((state, version)) => {
                probe.state = state;
                probe.version = version;
            }
        }
        probe
    }
}

/// How the effective root was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RootSelection {
    /// Explicit registry override; other roots are not read.
    Explicit,
    /// Existing default, even when the proposed root contains a registry.
    LegacyDefault,
}

/// Versioned root discovery response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RegistryDiscovery {
    /// `voqx-registry-discovery/1`.
    pub protocol: String,
    /// Selection policy; discovery never switches defaults.
    pub selection: RootSelection,
    /// Root current consumers actually use.
    pub effective: RegistryProbe,
    /// Proposed root, not read when an explicit override is present.
    pub proposed: Option<RegistryProbe>,
    /// Always false: filesystem presence is not recognition qualification.
    pub ready_to_filter: bool,
}

impl RegistryDiscovery {
    /// Inspect default candidates or only the explicit override.
    #[must_use]
    pub fn inspect(legacy: PathBuf, proposed: PathBuf, explicit: Option<PathBuf>) -> Self {
        let (selection, effective, proposed) = explicit.map_or_else(
            || {
                (
                    RootSelection::LegacyDefault,
                    RegistryProbe::inspect(legacy),
                    Some(RegistryProbe::inspect(proposed)),
                )
            },
            |root| (RootSelection::Explicit, RegistryProbe::inspect(root), None),
        );
        Self {
            protocol: "voqx-registry-discovery/1".into(),
            selection,
            effective,
            proposed,
            ready_to_filter: false,
        }
    }
}

type ProbeResult<T> = std::result::Result<T, (RegistryState, Option<u32>)>;

fn read_registry(root: &Path) -> ProbeResult<(VoiceRegistry, Vec<u8>)> {
    read_registry_inner(root, false)
}

fn read_registry_inner(root: &Path, held: bool) -> ProbeResult<(VoiceRegistry, Vec<u8>)> {
    check_type(root, true)?;
    if !held {
        check_unlocked(root)?;
    }
    let file = root.join("registry.json");
    check_type(&file, false)?;
    let mut bytes = Vec::new();
    std::fs::File::open(file)
        .and_then(|file| file.take(MAX_REGISTRY_BYTES + 1).read_to_end(&mut bytes))
        .map_err(|_| (RegistryState::Unavailable, None))?;
    if bytes.len() as u64 > MAX_REGISTRY_BYTES {
        return Err((RegistryState::LimitExceeded, None));
    }
    let value: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|_| (RegistryState::Invalid, None))?;
    let version = value
        .get("version")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())
        .ok_or((RegistryState::Invalid, None))?;
    if version != REGISTRY_VERSION {
        return Err((RegistryState::UnsupportedVersion, Some(version)));
    }
    let registry =
        serde_json::from_value(value).map_err(|_| (RegistryState::Invalid, Some(version)))?;
    if !held {
        check_unlocked(root)?;
    }
    Ok((registry, bytes))
}

fn check_type(path: &Path, directory: bool) -> ProbeResult<()> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        let state = if error.kind() == std::io::ErrorKind::NotFound {
            RegistryState::Missing
        } else {
            RegistryState::Unavailable
        };
        (state, None)
    })?;
    if metadata.file_type().is_symlink()
        || (directory && !metadata.is_dir())
        || (!directory && !metadata.is_file())
    {
        return Err((RegistryState::UnsafePath, None));
    }
    Ok(())
}

fn check_unlocked(root: &Path) -> ProbeResult<()> {
    match std::fs::symlink_metadata(root.join(".lock")) {
        Ok(_) => Err((RegistryState::Busy, None)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(_) => Err((RegistryState::Unavailable, None)),
    }
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
