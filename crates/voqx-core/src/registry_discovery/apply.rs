//! Explicit, reviewed copy operations. Never switch defaults or remove originals.

mod storage;
mod validation;

use std::ops::ControlFlow;
use std::time::Duration;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::inventory::Inventory;
use super::{EntryKind, MigrationPlan};
use crate::atomic::LockGuard;
use storage::{Journal, Phase};

pub(super) const RECOVERY_DIR: &str = ".voqx-migration";

/// Explicit migration intent; none changes the default registry root.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MigrationAction {
    /// Start a reviewed copy, or verify an already-complete destination.
    Apply,
    /// Continue journal-bound partial work after revalidation.
    Resume,
    /// Abort before registry publication, retaining all copied data.
    Abort,
    /// Read-only revalidation; no locks, directories or journal are created.
    Validate,
}

/// Cooperative cancellation/progress points. No path or identity is disclosed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MigrationCheckpoint {
    /// Journal is durable; registry JSON has not been published.
    Prepared,
    /// A reviewed file is durable in private staging.
    Staged(usize),
    /// A non-registry reviewed file was published without overwrite.
    Published(usize),
    /// All assets are verified; registry publication has not started.
    BeforeRegistry,
}

/// Versioned operation outcome. Aborted never means copied bytes were deleted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MigrationOutcome {
    /// Read-only validation passed; this is not permission to write.
    Validated,
    /// Exact destination bytes were verified after publication.
    Completed,
    /// The complete destination already matched, with no payload copy needed.
    AlreadyComplete,
    /// Incomplete intent was marked aborted; partial data is retained.
    Aborted,
    /// Cooperative cancellation before registry publication; explicit resume required.
    Interrupted,
    /// Input/state validation failed before migration payload writes.
    Refused,
    /// A write/verification operation failed; effects may be partial.
    Failed,
}

/// Bounded failure classes; never echo private paths, labels or file contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum MigrationFailure {
    /// CLI confirmation was not supplied; the library is not an authority boundary.
    ConfirmationRequired,
    /// A CLI option cannot be honored by the reviewed-plan migration operation.
    UnsupportedOptions,
    /// Supplied plan/expected revision is invalid, unsupported or altered.
    InvalidPlan,
    /// Source no longer matches the reviewed exact inventory/references.
    SourceChanged,
    /// Destination differs from the baseline or permitted exact partial copy.
    DestinationChanged,
    /// Source/destination writer exclusion could not be obtained.
    Busy,
    /// Existing partial work requires explicit resume, or no journal exists to resume.
    RecoveryRequired,
    /// Recovery state is malformed, foreign, unsafe or not bound to this plan.
    InvalidRecovery,
    /// Registry JSON is already published; abort is not an unpublish operation.
    AlreadyPublished,
    /// Local filesystem operation failed. No replacing/copy fallback is attempted.
    Io,
    /// A staged or final file did not match its reviewed byte/hash receipt.
    VerificationFailed,
    /// Caller requested cancellation before registry publication.
    Interrupted,
}

type Result<T> = std::result::Result<T, MigrationFailure>;

/// A structured receipt, including conservative partial-effect/recovery information.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct MigrationReceipt {
    /// `voqx-registry-migration/1`.
    pub protocol: String,
    /// Reviewed plan revision, if supplied by the plan.
    pub plan_revision: Option<String>,
    /// Observed effect outcome.
    pub outcome: MigrationOutcome,
    /// Failure reason, absent on success/abort/validation.
    pub error: Option<MigrationFailure>,
    /// True once write-phase metadata or payload effects may have happened.
    pub destination_may_have_changed: bool,
    /// True for partial/uncertain work requiring inspection, not a guarantee of resumability.
    pub recovery_required: bool,
    /// Exact verified registry JSON revision on complete outcomes only.
    pub registry_revision: Option<String>,
    /// Files verified for complete outcomes only.
    pub verified_files: usize,
    /// Bytes verified for complete outcomes only.
    pub verified_bytes: u64,
    /// Always false; migration never writes source payloads.
    pub source_payloads_written: bool,
    /// Always false; migration never changes configuration/default selection.
    pub defaults_changed: bool,
}

impl MigrationReceipt {
    /// Construct a refusal before attempting any migration operation.
    #[must_use]
    pub fn refused(plan: Option<&MigrationPlan>, error: MigrationFailure) -> Self {
        Self {
            protocol: "voqx-registry-migration/1".into(),
            plan_revision: plan.and_then(|plan| plan.revision.clone()),
            outcome: MigrationOutcome::Refused,
            error: Some(error),
            destination_may_have_changed: false,
            recovery_required: false,
            registry_revision: None,
            verified_files: 0,
            verified_bytes: 0,
            source_payloads_written: false,
            defaults_changed: false,
        }
    }
}

/// Apply, resume, abort or validate an explicitly reviewed exact-byte copy plan.
///
/// `expected` must match both stored and independently recomputed plan revisions.
/// Progress callbacks can cancel before registry publication. The writer protocol
/// excludes cooperative writers, not hostile filesystem replacement. A hard crash
/// leaves create-new locks for operator review; no stale lock is removed here.
/// Receipts do not grant permission or promise automatic rollback of asset writes.
pub fn execute_migration(
    plan: &MigrationPlan,
    expected: &str,
    action: MigrationAction,
    mut progress: impl FnMut(MigrationCheckpoint) -> ControlFlow<()>,
) -> MigrationReceipt {
    let mut receipt = MigrationReceipt::refused(Some(plan), MigrationFailure::InvalidPlan);
    let result = run(plan, expected, action, &mut progress, &mut receipt);
    if let Err(error) = result {
        receipt.outcome = if error == MigrationFailure::Interrupted {
            MigrationOutcome::Interrupted
        } else if receipt.destination_may_have_changed {
            MigrationOutcome::Failed
        } else {
            MigrationOutcome::Refused
        };
        receipt.error = Some(error);
    }
    receipt
}

fn run(
    plan: &MigrationPlan,
    expected: &str,
    action: MigrationAction,
    progress: &mut impl FnMut(MigrationCheckpoint) -> ControlFlow<()>,
    receipt: &mut MigrationReceipt,
) -> Result<()> {
    let reviewed = validation::Reviewed::new(plan, expected)?;
    let journal = Journal::load(plan)?;
    let initial = reviewed.inspect(false, journal.is_some())?;
    if initial.contains_key("registry.json") && initial != reviewed.source {
        return Err(MigrationFailure::AlreadyPublished);
    }
    if journal
        .as_ref()
        .is_some_and(|state| state.phase == Phase::Completed)
        && initial != reviewed.source
    {
        return Err(MigrationFailure::InvalidRecovery);
    }
    if action == MigrationAction::Validate {
        receipt.outcome = MigrationOutcome::Validated;
        receipt.error = None;
        receipt.recovery_required = journal.is_some() && initial != reviewed.source;
        return Ok(());
    }
    if action == MigrationAction::Abort && initial == reviewed.source {
        return Err(MigrationFailure::AlreadyPublished);
    }
    if journal.is_some() && action == MigrationAction::Apply && initial != reviewed.source {
        return Err(MigrationFailure::RecoveryRequired);
    }
    if journal.is_none()
        && initial != reviewed.source
        && matches!(action, MigrationAction::Resume | MigrationAction::Abort)
    {
        return Err(MigrationFailure::RecoveryRequired);
    }
    receipt.destination_may_have_changed = true;
    let _locks = acquire_locks(plan)?;
    let current_journal = Journal::load(plan)?;
    if current_journal != journal {
        return Err(MigrationFailure::InvalidRecovery);
    }
    let destination = reviewed.inspect(true, journal.is_some())?;
    if destination == reviewed.source {
        if let Some(mut journal) = journal
            && journal.phase != Phase::Completed
        {
            receipt.recovery_required = true;
            journal.prepare_resume(plan)?;
            journal.phase = Phase::Completed;
            journal.save(plan)?;
        }
        complete(
            receipt,
            plan,
            &destination,
            MigrationOutcome::AlreadyComplete,
        );
        return Ok(());
    }
    let mut journal = match journal {
        Some(journal) => journal,
        None => Journal::begin(plan)?,
    };
    receipt.recovery_required = true;
    if action == MigrationAction::Abort {
        if destination.contains_key("registry.json") {
            return Err(MigrationFailure::AlreadyPublished);
        }
        journal.phase = Phase::Aborted;
        journal.save(plan)?;
        receipt.outcome = MigrationOutcome::Aborted;
        receipt.error = None;
        return Ok(());
    }
    journal.prepare_resume(plan)?;
    checkpoint(progress, MigrationCheckpoint::Prepared)?;
    copy_payloads(plan, &reviewed, &destination, &mut journal, progress)?;
    checkpoint(progress, MigrationCheckpoint::BeforeRegistry)?;
    reviewed.verify_before_registry()?;
    let registry = reviewed
        .source
        .get("registry.json")
        .ok_or(MigrationFailure::InvalidPlan)?;
    journal.publish(plan, registry, progress, true)?;
    let final_inventory = reviewed.inspect(true, true)?;
    if final_inventory != reviewed.source {
        return Err(MigrationFailure::VerificationFailed);
    }
    journal.phase = Phase::Completed;
    journal.save(plan)?;
    complete(receipt, plan, &final_inventory, MigrationOutcome::Completed);
    Ok(())
}

fn acquire_locks(plan: &MigrationPlan) -> Result<Vec<LockGuard>> {
    storage::private_dir(&plan.destination)?;
    let mut roots = [&plan.source.root, &plan.destination];
    roots.sort();
    roots
        .into_iter()
        .map(|root| {
            LockGuard::acquire(&root.join(".lock"), Duration::ZERO)
                .map_err(|_| MigrationFailure::Busy)
        })
        .collect()
}

fn copy_payloads(
    plan: &MigrationPlan,
    reviewed: &validation::Reviewed<'_>,
    destination: &Inventory,
    journal: &mut Journal,
    progress: &mut impl FnMut(MigrationCheckpoint) -> ControlFlow<()>,
) -> Result<()> {
    for entry in reviewed.source.values() {
        if entry.path == "registry.json" || destination.contains_key(&entry.path) {
            continue;
        }
        if entry.kind == EntryKind::Directory {
            storage::private_dir(&plan.destination.join(&entry.path))?;
        } else {
            journal.publish(plan, entry, progress, false)?;
        }
    }
    Ok(())
}

fn checkpoint(
    progress: &mut impl FnMut(MigrationCheckpoint) -> ControlFlow<()>,
    point: MigrationCheckpoint,
) -> Result<()> {
    if progress(point).is_break() {
        return Err(MigrationFailure::Interrupted);
    }
    Ok(())
}

fn complete(
    receipt: &mut MigrationReceipt,
    plan: &MigrationPlan,
    entries: &Inventory,
    outcome: MigrationOutcome,
) {
    receipt.outcome = outcome;
    receipt.error = None;
    receipt.recovery_required = false;
    receipt.registry_revision.clone_from(&plan.source.revision);
    receipt.verified_files = entries
        .values()
        .filter(|entry| entry.kind == EntryKind::File)
        .count();
    receipt.verified_bytes = entries.values().map(|entry| entry.bytes).sum();
}
