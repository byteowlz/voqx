//! Copy-first planning only. A plan is not an executable migration receipt.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::inventory::{Inventory, portable_path, scan};
use super::{EntryKind, FileEntry, RegistryProbe, RegistryState, ScanIssue, read_registry};
use crate::VoiceRegistry;

/// Overall byte-preservation plan state; no state authorizes applying it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PlanState {
    /// Source entries are missing at destination; existing entries match exactly.
    CopyRequired,
    /// Both inventories match exactly; no copy is proposed.
    AlreadyPresent,
    /// At least one conflict, invalid reference or scan failure prevents planning.
    Blocked,
}

/// Destination comparison for an entry, without mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum TransferState {
    /// Not present at destination.
    Copy,
    /// Same type and exact bytes/hash at destination.
    AlreadyPresent,
    /// Same path with different type or bytes.
    Conflict,
    /// Destination-only entry; no deletion/merge is proposed.
    DestinationOnly,
}

/// Entry-by-entry preservation comparison. Keep plans private.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PlanEntry {
    /// Source receipt; absent for destination-only entries.
    pub source: Option<FileEntry>,
    /// Destination receipt; absent when missing.
    pub destination: Option<FileEntry>,
    /// Proposed comparison, never an applied effect.
    pub state: TransferState,
}

/// A bounded reason a plan cannot proceed. No label or reference value is echoed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case", tag = "reason")]
pub enum PlanBlocker {
    /// Source registry is absent, busy, malformed or unsupported.
    SourceNotReady,
    /// Same root or ancestor/descendant roots, including existing path aliases.
    OverlappingRoots,
    /// Cannot resolve root locations without changing the filesystem.
    RootUnavailable,
    /// Source tree cannot be inventoried safely.
    SourceScan {
        /// Bounded failure reason.
        issue: ScanIssue,
    },
    /// Destination tree cannot be inventoried safely.
    DestinationScan {
        /// Bounded failure reason.
        issue: ScanIssue,
    },
    /// Registry references a missing, non-file or non-relative asset.
    InvalidReferences,
    /// Source JSON revision changed between inspection and inventory.
    SourceChanged,
    /// Existing destination entries conflict or are not in the source tree.
    DestinationConflict,
    /// Source contains the reserved migration recovery directory.
    ReservedPath,
}

/// Read-only, versioned migration plan over exact bytes, not ID/schema migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct MigrationPlan {
    /// `voqx-registry-migration-plan/1`.
    pub protocol: String,
    /// SHA-256 of the normalized plan with this field null; absent if unencodable.
    pub revision: Option<String>,
    /// Source registry inspection.
    pub source: RegistryProbe,
    /// Proposed destination; never created by planning.
    pub destination: PathBuf,
    /// Aggregate state.
    pub state: PlanState,
    /// Reasons for refusal.
    pub blockers: Vec<PlanBlocker>,
    /// Deterministically ordered private path/hash comparisons.
    pub entries: Vec<PlanEntry>,
    /// This plan is eligible for explicit application after locked revalidation.
    pub apply_supported: bool,
}

impl MigrationPlan {
    /// Hash and compare roots without writing, acquiring locks or loading models.
    ///
    /// Hashes are point-in-time observations, not an atomic filesystem snapshot.
    /// A future apply operation must revalidate everything under writer exclusion.
    #[must_use]
    pub fn inspect(source: PathBuf, destination: PathBuf) -> Self {
        let mut plan = Self {
            protocol: "voqx-registry-migration-plan/1".into(),
            revision: None,
            source: RegistryProbe::inspect(source),
            destination,
            state: PlanState::Blocked,
            blockers: Vec::new(),
            entries: Vec::new(),
            apply_supported: false,
        };
        if let Err(blocker) = plan.populate() {
            plan.blockers.push(blocker);
        }
        plan.revision = plan.content_revision().ok();
        plan
    }

    /// Recompute the review revision independently of the stored revision field.
    ///
    /// # Errors
    /// Returns an error if a root cannot be serialized (for example non-UTF-8).
    pub fn content_revision(&self) -> Result<String, serde_json::Error> {
        let mut normalized = self.clone();
        normalized.revision = None;
        serde_json::to_vec(&normalized).map(|bytes| super::digest(&bytes))
    }

    fn populate(&mut self) -> Result<(), PlanBlocker> {
        if self.source.state != RegistryState::Ready {
            return Err(PlanBlocker::SourceNotReady);
        }
        let source_location = location(&self.source.root).ok_or(PlanBlocker::RootUnavailable)?;
        let destination_location =
            location(&self.destination).ok_or(PlanBlocker::RootUnavailable)?;
        if source_location.starts_with(&destination_location)
            || destination_location.starts_with(&source_location)
        {
            return Err(PlanBlocker::OverlappingRoots);
        }
        self.source.root = source_location;
        self.destination = destination_location;
        let source = scan(&self.source.root).map_err(|issue| PlanBlocker::SourceScan { issue })?;
        if source.contains_key(super::apply::RECOVERY_DIR) {
            return Err(PlanBlocker::ReservedPath);
        }
        let destination =
            scan(&self.destination).map_err(|issue| PlanBlocker::DestinationScan { issue })?;
        let (registry, bytes) =
            read_registry(&self.source.root).map_err(|_| PlanBlocker::SourceChanged)?;
        let revision = super::digest(&bytes);
        if self.source.revision.as_ref() != Some(&revision)
            || source
                .get("registry.json")
                .and_then(|entry| entry.sha256.as_ref())
                != Some(&revision)
        {
            return Err(PlanBlocker::SourceChanged);
        }
        validate_references(&registry, &source)?;
        self.entries = compare(&source, &destination);
        if self.entries.iter().any(|entry| {
            matches!(
                entry.state,
                TransferState::Conflict | TransferState::DestinationOnly
            )
        }) {
            return Err(PlanBlocker::DestinationConflict);
        }
        self.state = if self
            .entries
            .iter()
            .any(|entry| entry.state == TransferState::Copy)
        {
            PlanState::CopyRequired
        } else {
            PlanState::AlreadyPresent
        };
        self.apply_supported =
            !(self.state == PlanState::CopyRequired && destination.contains_key("registry.json"));
        Ok(())
    }
}

fn compare(source: &Inventory, destination: &Inventory) -> Vec<PlanEntry> {
    let keys: BTreeSet<_> = source.keys().chain(destination.keys()).collect();
    keys.into_iter()
        .map(|key| {
            let source = source.get(key).cloned();
            let destination = destination.get(key).cloned();
            let state = match (&source, &destination) {
                (Some(a), Some(b)) if a == b => TransferState::AlreadyPresent,
                (Some(_), Some(_)) => TransferState::Conflict,
                (Some(_), None) => TransferState::Copy,
                _ => TransferState::DestinationOnly,
            };
            PlanEntry {
                source,
                destination,
                state,
            }
        })
        .collect()
}

pub(super) fn validate_references(
    registry: &VoiceRegistry,
    entries: &Inventory,
) -> Result<(), PlanBlocker> {
    let mut references = Vec::new();
    for identity in &registry.identities {
        for source in &identity.sources {
            references.push(source.audio.as_str());
            references.extend(source.transcript.as_deref());
        }
        for profile in &identity.recognition_profiles {
            references.extend(profile.samples.iter().map(String::as_str));
        }
        for profile in &identity.synthesis_profiles {
            for reference in &profile.references {
                references.extend([reference.audio.as_str(), reference.transcript.as_str()]);
            }
        }
        references.extend(
            identity
                .wake_profiles
                .iter()
                .filter_map(|profile| profile.artifact.as_deref()),
        );
    }
    for reference in references {
        let key = portable_path(Path::new(reference)).ok_or(PlanBlocker::InvalidReferences)?;
        if !entries
            .get(&key)
            .is_some_and(|entry| entry.kind == EntryKind::File)
        {
            return Err(PlanBlocker::InvalidReferences);
        }
    }
    Ok(())
}

// Resolve existing ancestors so aliases cannot disguise overlapping roots.
pub(super) fn location(path: &Path) -> Option<PathBuf> {
    let absolute = std::path::absolute(path).ok()?;
    if absolute
        .components()
        .any(|part| part == std::path::Component::ParentDir)
    {
        return None;
    }
    let mut parent = absolute.as_path();
    let mut suffix = Vec::new();
    loop {
        match std::fs::symlink_metadata(parent) {
            Ok(_) => break,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                suffix.push(parent.file_name()?);
                parent = parent.parent()?;
            }
            Err(_) => return None,
        }
    }
    let mut result = parent.canonicalize().ok()?;
    for part in suffix.into_iter().rev() {
        result.push(part);
    }
    Some(result)
}
