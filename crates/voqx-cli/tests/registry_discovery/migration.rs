//! End-to-end CLI migration proof using reviewed synthetic plan files only.

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::Output;

use super::Scratch;
use anyhow::{Context, Result, ensure};
use voqx_core::registry_discovery::{
    MigrationAction, MigrationCheckpoint, MigrationPlan, execute_migration,
};

#[derive(Debug)]
struct Fixture {
    dir: Scratch,
    plan: MigrationPlan,
    path: PathBuf,
    revision: String,
}
impl Fixture {
    fn new() -> Result<Self> {
        let dir = Scratch::new()?;
        let source = dir.0.join("source");
        voqx_core::VoiceRegistry::default()
            .save(&voqx_core::RegistryPaths::from_root(source.clone()))?;
        std::fs::write(source.join("payload"), b"frozen synthetic bytes")?;
        let plan = MigrationPlan::inspect(source, dir.0.join("destination"));
        let revision = plan.revision.clone().context("missing review revision")?;
        let path = dir.0.join("reviewed-plan.json");
        std::fs::write(&path, serde_json::to_vec_pretty(&plan)?)?;
        Ok(Self {
            dir,
            plan,
            path,
            revision,
        })
    }
    fn invoke(&self, options: &[&str]) -> Result<(Output, serde_json::Value)> {
        let mut args = vec![
            "registry",
            "migrate",
            "--plan",
            self.path.to_str().context("invalid fixture path")?,
            "--expected-plan",
            &self.revision,
            "--json",
        ];
        args.extend_from_slice(options);
        let output = self.dir.invoke(&args)?;
        let value = serde_json::from_slice(&output.stdout)?;
        Ok((output, value))
    }
}

#[test]
fn dry_run_and_confirmation_guards_never_initialize_state_or_copy() -> Result<()> {
    let fixture = Fixture::new()?;
    let (output, receipt) = fixture.invoke(&[])?;
    ensure!(!output.status.success());
    ensure!(receipt["error"] == "confirmation_required");
    ensure!(!fixture.plan.destination.exists());
    let (output, receipt) = fixture.invoke(&["--dry-run"])?;
    ensure!(output.status.success(), "{receipt}");
    ensure!(receipt["outcome"] == "validated");
    ensure!(receipt["destination_may_have_changed"] == false);
    ensure!(!fixture.plan.destination.exists());
    for name in ["config", "state", "cache", "hf", "data"] {
        ensure!(!fixture.dir.0.join(name).exists());
    }
    Ok(())
}

#[test]
fn unsupported_overrides_are_refused_instead_of_silently_ignored() -> Result<()> {
    let fixture = Fixture::new()?;
    for options in [
        vec!["--yes", "--registry", "/unreviewed"],
        vec!["--yes", "--timeout", "1"],
        vec!["--yes", "--parallel", "2"],
    ] {
        let (output, receipt) = fixture.invoke(&options)?;
        ensure!(!output.status.success() && receipt["error"] == "unsupported_options");
        ensure!(!fixture.plan.destination.exists());
    }
    Ok(())
}

#[test]
fn reviewed_plan_cli_copies_exact_bytes_and_returns_idempotent_receipts() -> Result<()> {
    let fixture = Fixture::new()?;
    let (output, receipt) = fixture.invoke(&["--yes"])?;
    ensure!(output.status.success(), "{receipt}");
    ensure!(receipt["protocol"] == "voqx-registry-migration/1");
    ensure!(receipt["plan_revision"] == fixture.revision);
    ensure!(receipt["outcome"] == "completed");
    ensure!(receipt["source_payloads_written"] == false && receipt["defaults_changed"] == false);
    for name in ["registry.json", "payload"] {
        ensure!(
            std::fs::read(fixture.plan.source.root.join(name))?
                == std::fs::read(fixture.plan.destination.join(name))?
        );
    }
    let (output, receipt) = fixture.invoke(&["--yes"])?;
    ensure!(output.status.success() && receipt["outcome"] == "already_complete");
    ensure!(!fixture.dir.0.join("config").exists());
    Ok(())
}

#[test]
fn cli_abort_retains_partial_data_and_resume_is_explicit() -> Result<()> {
    let fixture = Fixture::new()?;
    let interrupted = execute_migration(
        &fixture.plan,
        &fixture.revision,
        MigrationAction::Apply,
        |point| {
            if point == MigrationCheckpoint::BeforeRegistry {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        },
    );
    ensure!(interrupted.recovery_required);
    let (output, receipt) = fixture.invoke(&["--yes"])?;
    ensure!(!output.status.success() && receipt["error"] == "recovery_required");
    let (output, receipt) = fixture.invoke(&["--yes", "--action", "abort"])?;
    ensure!(output.status.success() && receipt["outcome"] == "aborted");
    ensure!(fixture.plan.destination.join("payload").exists());
    ensure!(!fixture.plan.destination.join("registry.json").exists());
    let (output, receipt) = fixture.invoke(&["--yes", "--action", "resume"])?;
    ensure!(output.status.success() && receipt["outcome"] == "completed");
    Ok(())
}

#[test]
fn altered_oversized_and_unknown_field_plans_fail_closed() -> Result<()> {
    let fixture = Fixture::new()?;
    let mut altered = fixture.plan.clone();
    altered.destination = fixture.dir.0.join("unreviewed");
    std::fs::write(&fixture.path, serde_json::to_vec(&altered)?)?;
    let (output, receipt) = fixture.invoke(&["--yes"])?;
    ensure!(!output.status.success() && receipt["error"] == "invalid_plan");
    ensure!(!altered.destination.exists());
    let mut fields = serde_json::to_value(&fixture.plan)?;
    fields["source"]["caller_grant"] = true.into();
    std::fs::write(&fixture.path, serde_json::to_vec(&fields)?)?;
    let (output, receipt) = fixture.invoke(&["--yes"])?;
    ensure!(!output.status.success() && receipt["error"] == "invalid_plan");
    std::fs::File::create(&fixture.path)?.set_len(32 * 1024 * 1024 + 1)?;
    let (output, receipt) = fixture.invoke(&["--yes"])?;
    ensure!(!output.status.success() && receipt["error"] == "invalid_plan");
    ensure!(!fixture.plan.destination.exists());
    Ok(())
}
