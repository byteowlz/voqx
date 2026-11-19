//! Confirmed migration adapter. All execution logic lives in the ML-free library.

use std::io::Read;
use std::ops::ControlFlow;
use std::path::PathBuf;

use anyhow::{Result, bail};
use clap::{Args, ValueEnum};
use voqx_core::registry_discovery::{
    MigrationAction, MigrationFailure, MigrationOutcome, MigrationPlan, MigrationReceipt,
    execute_migration,
};

const MAX_PLAN_BYTES: u64 = 32 * 1024 * 1024;

/// Arguments for explicit reviewed-plan migration (no implicit root selection).
#[derive(Debug, Args)]
pub struct MigrateArgs {
    /// Reviewed JSON plan from registry migration-plan (kept private)
    #[arg(long, value_name = "PATH")]
    plan: PathBuf,
    /// Exact revision field from the reviewed plan, not a hash of its JSON formatting
    #[arg(long, value_name = "SHA256")]
    expected_plan: String,
    /// Apply new work, resume matching partial work, or abort without deleting payloads
    #[arg(long, value_enum, default_value_t = Action::Apply)]
    action: Action,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Action {
    Apply,
    Resume,
    Abort,
}

pub(super) fn run(args: &MigrateArgs, common: &crate::CommonOpts) -> Result<()> {
    let receipt = execute(args, common);
    crate::output::print(&receipt, common.yaml)?;
    if !matches!(
        receipt.outcome,
        MigrationOutcome::Validated
            | MigrationOutcome::Completed
            | MigrationOutcome::AlreadyComplete
            | MigrationOutcome::Aborted
    ) {
        bail!(
            "registry migration refused or incomplete; inspect the structured receipt before retrying"
        );
    }
    Ok(())
}

fn execute(args: &MigrateArgs, common: &crate::CommonOpts) -> MigrationReceipt {
    if !common.dry_run && !common.assume_yes {
        return MigrationReceipt::refused(None, MigrationFailure::ConfirmationRequired);
    }
    // Roots come only from the reviewed plan. Do not pretend to honor an override
    // or a timeout/parallel limit which this bounded synchronous adapter lacks.
    if common.registry.is_some() || common.timeout.is_some() || common.parallel.is_some() {
        return MigrationReceipt::refused(None, MigrationFailure::UnsupportedOptions);
    }
    let Some(plan) = read_plan(args) else {
        return MigrationReceipt::refused(None, MigrationFailure::InvalidPlan);
    };
    let action = if common.dry_run {
        MigrationAction::Validate
    } else {
        match args.action {
            Action::Apply => MigrationAction::Apply,
            Action::Resume => MigrationAction::Resume,
            Action::Abort => MigrationAction::Abort,
        }
    };
    execute_migration(&plan, &args.expected_plan, action, |_| {
        ControlFlow::Continue(())
    })
}

fn read_plan(args: &MigrateArgs) -> Option<MigrationPlan> {
    let mut bytes = Vec::new();
    std::fs::File::open(&args.plan)
        .ok()?
        .take(MAX_PLAN_BYTES + 1)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() as u64 > MAX_PLAN_BYTES {
        return None;
    }
    serde_json::from_slice(&bytes).ok()
}
