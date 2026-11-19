//! Recovery/refusal proofs through the public migration API.

use super::*;

fn interrupt(plan: &MigrationPlan, staged: bool) -> Result<()> {
    let revision = plan.revision.as_deref().context("missing revision")?;
    let report = execute_migration(plan, revision, MigrationAction::Apply, |point| {
        if (staged && matches!(point, MigrationCheckpoint::Staged(_)))
            || (!staged && matches!(point, MigrationCheckpoint::Published(_)))
        {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    });
    ensure!(
        report.outcome == MigrationOutcome::Interrupted,
        "{report:?}"
    );
    Ok(())
}

#[test]
fn abort_retains_partial_copies_and_requires_explicit_resume() -> Result<()> {
    let (_dir, plan) = setup()?;
    let source = std::fs::read(plan.source.root.join("registry.json"))?;
    interrupt(&plan, false)?;
    let copied = std::fs::read(plan.destination.join("assets/audio.bin"))?;
    let aborted = perform(&plan, MigrationAction::Abort)?;
    ensure!(aborted.outcome == MigrationOutcome::Aborted && aborted.recovery_required);
    ensure!(std::fs::read(plan.destination.join("assets/audio.bin"))? == copied);
    ensure!(!plan.destination.join("registry.json").exists());
    ensure!(std::fs::read(plan.source.root.join("registry.json"))? == source);
    ensure!(
        perform(&plan, MigrationAction::Apply)?.error == Some(MigrationFailure::RecoveryRequired)
    );
    ensure!(perform(&plan, MigrationAction::Resume)?.outcome == MigrationOutcome::Completed);
    Ok(())
}

#[test]
fn declared_partial_stage_can_be_discarded_but_unknown_recovery_data_is_retained() -> Result<()> {
    let (_dir, plan) = setup()?;
    interrupt(&plan, true)?;
    let pending = plan.destination.join(".voqx-migration/pending");
    std::fs::write(&pending, b"partial chunk")?;
    ensure!(perform(&plan, MigrationAction::Resume)?.outcome == MigrationOutcome::Completed);
    ensure!(!pending.exists());
    let unknown = plan.destination.join(".voqx-migration/unknown");
    std::fs::write(&unknown, b"retain for operator")?;
    ensure!(
        perform(&plan, MigrationAction::Resume)?.error == Some(MigrationFailure::InvalidRecovery)
    );
    ensure!(std::fs::read(unknown)? == b"retain for operator");
    Ok(())
}

#[test]
fn invalid_foreign_or_unbound_journals_cannot_resume() -> Result<()> {
    let (_dir, plan) = setup()?;
    interrupt(&plan, true)?;
    let path = plan.destination.join(".voqx-migration/state.json");
    let original = std::fs::read(&path)?;
    let valid: serde_json::Value = serde_json::from_slice(&original)?;
    let mut foreign = valid.clone();
    foreign["plan_revision"] = "wrong".into();
    let mut bad_index = valid.clone();
    bad_index["staging"] = 1_000_000.into();
    let mut unknown_field = valid;
    unknown_field["caller_grant"] = true.into();
    for altered in [
        b"not JSON".to_vec(),
        serde_json::to_vec(&foreign)?,
        serde_json::to_vec(&bad_index)?,
        serde_json::to_vec(&unknown_field)?,
    ] {
        std::fs::write(&path, &altered)?;
        let report = perform(&plan, MigrationAction::Resume)?;
        ensure!(
            report.error == Some(MigrationFailure::InvalidRecovery),
            "{report:?}"
        );
        ensure!(std::fs::read(&path)? == altered);
        ensure!(!plan.destination.join("registry.json").exists());
    }
    std::fs::write(path, original)?;
    ensure!(perform(&plan, MigrationAction::Resume)?.outcome == MigrationOutcome::Completed);
    Ok(())
}

#[test]
fn leftover_crash_locks_are_not_guessed_stale_or_deleted() -> Result<()> {
    let (_dir, plan) = setup()?;
    interrupt(&plan, false)?;
    let lock = plan.source.root.join(".lock");
    std::fs::write(&lock, b"synthetic stopped-holder marker")?;
    let report = perform(&plan, MigrationAction::Resume)?;
    ensure!(report.error == Some(MigrationFailure::Busy));
    ensure!(std::fs::read(&lock)? == b"synthetic stopped-holder marker");
    ensure!(!plan.destination.join("registry.json").exists());
    // The fixture operator knows the synthetic holder is stopped; not engine policy.
    std::fs::remove_file(lock)?;
    ensure!(perform(&plan, MigrationAction::Resume)?.outcome == MigrationOutcome::Completed);
    Ok(())
}

#[test]
fn reviewed_existing_assets_are_retained_but_published_partial_registries_refuse_apply()
-> Result<()> {
    let (_dir, mut plan) = setup()?;
    std::fs::create_dir_all(plan.destination.join("assets"))?;
    let existing = plan.destination.join("assets/audio.bin");
    std::fs::copy(plan.source.root.join("assets/audio.bin"), &existing)?;
    let modified = std::fs::metadata(&existing)?.modified()?;
    plan = MigrationPlan::inspect(plan.source.root, plan.destination);
    ensure!(perform(&plan, MigrationAction::Apply)?.outcome == MigrationOutcome::Completed);
    ensure!(std::fs::metadata(existing)?.modified()? == modified);
    std::fs::remove_file(plan.destination.join("assets/text.txt"))?;
    // A newly reviewed plan is still not an implicit repair of a published registry.
    let repair = MigrationPlan::inspect(plan.source.root, plan.destination);
    ensure!(!repair.apply_supported);
    ensure!(perform(&repair, MigrationAction::Apply)?.error == Some(MigrationFailure::InvalidPlan));
    Ok(())
}

#[test]
fn publication_before_final_journal_update_is_verified_without_republishing() -> Result<()> {
    let (_dir, plan) = setup()?;
    ensure!(perform(&plan, MigrationAction::Apply)?.outcome == MigrationOutcome::Completed);
    let path = plan.destination.join(".voqx-migration/state.json");
    let mut state: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    let index = plan
        .entries
        .iter()
        .position(|entry| {
            entry
                .source
                .as_ref()
                .is_some_and(|file| file.path == "registry.json")
        })
        .context("missing registry entry")?;
    state["phase"] = "copying".into();
    state["staging"] = index.into();
    std::fs::write(&path, serde_json::to_vec(&state)?)?;
    let registry = plan.destination.join("registry.json");
    std::fs::hard_link(&registry, plan.destination.join(".voqx-migration/pending"))?;
    let before = std::fs::read(&registry)?;
    let report = perform(&plan, MigrationAction::Resume)?;
    ensure!(
        report.outcome == MigrationOutcome::AlreadyComplete,
        "{report:?}"
    );
    ensure!(std::fs::read(&registry)? == before);
    ensure!(!plan.destination.join(".voqx-migration/pending").exists());
    let state: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    ensure!(state["phase"] == "completed" && state["staging"].is_null());
    Ok(())
}

#[test]
fn published_registry_is_not_silently_repaired_or_aborted() -> Result<()> {
    let (_dir, plan) = setup()?;
    ensure!(perform(&plan, MigrationAction::Apply)?.outcome == MigrationOutcome::Completed);
    let before = std::fs::read(plan.destination.join("registry.json"))?;
    let abort = perform(&plan, MigrationAction::Abort)?;
    ensure!(abort.error == Some(MigrationFailure::AlreadyPublished));
    std::fs::remove_file(plan.destination.join("assets/audio.bin"))?;
    let resume = perform(&plan, MigrationAction::Resume)?;
    ensure!(
        resume.error == Some(MigrationFailure::AlreadyPublished),
        "{resume:?}"
    );
    ensure!(!plan.destination.join("assets/audio.bin").exists());
    ensure!(std::fs::read(plan.destination.join("registry.json"))? == before);
    Ok(())
}
