//! Real filesystem operations over synthetic byte fixtures; no model or user data.

use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};
use voqx_core::registry_discovery::{
    MigrationAction, MigrationCheckpoint, MigrationFailure, MigrationOutcome, MigrationPlan,
    execute_migration,
};
use voqx_core::{RegistryPaths, VoiceRegistry};

#[path = "registry_migration/recovery.rs"]
mod recovery;
#[path = "registry_migration/refusal.rs"]
mod refusal;

fn setup() -> Result<(Scratch, MigrationPlan)> {
    let dir = Scratch::new()?;
    let source = dir.0.join("source");
    fixture(&source)?;
    let plan = MigrationPlan::inspect(source, dir.0.join("destination"));
    Ok((dir, plan))
}

fn perform(
    plan: &MigrationPlan,
    action: MigrationAction,
) -> Result<voqx_core::registry_discovery::MigrationReceipt> {
    Ok(execute_migration(
        plan,
        plan.revision.as_deref().context("missing plan revision")?,
        action,
        |_| ControlFlow::Continue(()),
    ))
}

#[derive(Debug)]
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "voqx-migrate-{}-{nonce}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root)?;
        Ok(Self(root))
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(root: &Path) -> Result<()> {
    std::fs::create_dir_all(root.join("assets/empty"))?;
    std::fs::write(
        root.join("assets/audio.bin"),
        b"frozen synthetic audio\0\x01",
    )?;
    std::fs::write(root.join("assets/text.txt"), b"Synthetic transcript\n")?;
    let mut registry = VoiceRegistry::default();
    let id = registry.create_identity("Synthetic fixture");
    let identity = registry.find_mut(&id).context("missing fixture")?;
    identity.sources.push(voqx_core::SourceRecording {
        id: "source-1".into(),
        audio: "assets/audio.bin".into(),
        transcript: Some("assets/text.txt".into()),
        language: Some("en".into()),
        note: "fixture consent note".into(),
        created_at_unix: 1,
    });
    identity
        .recognition_profiles
        .push(voqx_core::RecognitionProfile::new(
            "fixture-engine",
            "fixture-model",
            vec![vec![1.0, -0.0, 0.5]],
            vec!["assets/audio.bin".into()],
        ));
    registry.save(&RegistryPaths::from_root(root.to_path_buf()))?;
    Ok(())
}

#[test]
fn reviewed_copy_preserves_bytes_originals_and_is_idempotent() -> Result<()> {
    let dir = Scratch::new()?;
    let source = dir.0.join("source");
    let destination = dir.0.join("destination");
    fixture(&source)?;
    let before = std::fs::read(source.join("registry.json"))?;
    let plan = MigrationPlan::inspect(source.clone(), destination.clone());
    let revision = plan.revision.as_deref().context("plan has no revision")?;
    let report = execute_migration(&plan, revision, MigrationAction::Apply, |_| {
        ControlFlow::Continue(())
    });
    ensure!(
        report.outcome == MigrationOutcome::Completed,
        "receipt: {report:?}"
    );
    ensure!(!report.source_payloads_written && !report.defaults_changed);
    ensure!(report.verified_files == 3);
    for path in ["registry.json", "assets/audio.bin", "assets/text.txt"] {
        ensure!(std::fs::read(source.join(path))? == std::fs::read(destination.join(path))?);
    }
    ensure!(std::fs::read(source.join("registry.json"))? == before);
    ensure!(destination.join("assets/empty").is_dir());
    let repeated = execute_migration(&plan, revision, MigrationAction::Apply, |_| {
        ControlFlow::Continue(())
    });
    ensure!(
        repeated.outcome == MigrationOutcome::AlreadyComplete,
        "receipt: {repeated:?}"
    );
    ensure!(!source.join(".lock").exists() && !destination.join(".lock").exists());
    Ok(())
}

#[test]
fn every_precommit_checkpoint_can_resume_with_a_reloaded_reviewed_plan() -> Result<()> {
    for stop in 0..7 {
        let dir = Scratch::new()?;
        let source = dir.0.join("source");
        let destination = dir.0.join("destination");
        fixture(&source)?;
        let before = std::fs::read(source.join("registry.json"))?;
        let plan = MigrationPlan::inspect(source.clone(), destination.clone());
        let revision = plan.revision.as_deref().context("missing plan revision")?;
        let mut count = 0;
        let report = execute_migration(&plan, revision, MigrationAction::Apply, |_| {
            let cancel = count == stop;
            count += 1;
            if cancel {
                ControlFlow::Break(())
            } else {
                ControlFlow::Continue(())
            }
        });
        ensure!(
            report.outcome == MigrationOutcome::Interrupted,
            "checkpoint {stop}: {report:?}"
        );
        ensure!(report.error == Some(MigrationFailure::Interrupted));
        ensure!(report.recovery_required);
        ensure!(!destination.join("registry.json").exists());
        ensure!(std::fs::read(source.join("registry.json"))? == before);
        let replay = execute_migration(&plan, revision, MigrationAction::Apply, |_| {
            ControlFlow::Continue(())
        });
        ensure!(replay.error == Some(MigrationFailure::RecoveryRequired));
        let reloaded: MigrationPlan = serde_json::from_slice(&serde_json::to_vec(&plan)?)?;
        let report = execute_migration(&reloaded, revision, MigrationAction::Resume, |_| {
            ControlFlow::Continue(())
        });
        ensure!(
            report.outcome == MigrationOutcome::Completed,
            "checkpoint {stop}: {report:?}"
        );
        ensure!(std::fs::read(destination.join("registry.json"))? == before);
        ensure!(std::fs::read(source.join("registry.json"))? == before);
    }
    Ok(())
}

#[test]
fn changed_private_staging_is_not_published() -> Result<()> {
    let dir = Scratch::new()?;
    let source = dir.0.join("source");
    let destination = dir.0.join("destination");
    fixture(&source)?;
    let plan = MigrationPlan::inspect(source, destination.clone());
    let revision = plan.revision.as_deref().context("missing plan revision")?;
    let mut corrupted = false;
    let report = execute_migration(&plan, revision, MigrationAction::Apply, |point| {
        if !corrupted && matches!(point, MigrationCheckpoint::Staged(_)) {
            corrupted = std::fs::write(
                destination.join(".voqx-migration/pending"),
                b"wrong staged bytes",
            )
            .is_ok();
        }
        ControlFlow::Continue(())
    });
    ensure!(corrupted);
    ensure!(
        report.error == Some(MigrationFailure::VerificationFailed),
        "{report:?}"
    );
    ensure!(!destination.join("assets/audio.bin").exists());
    ensure!(!destination.join("registry.json").exists());
    Ok(())
}
