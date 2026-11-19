//! Reviewed-plan, conflict and writer-exclusion refusal properties.

use super::*;

#[test]
fn invalid_revision_and_altered_or_unsafe_plans_cannot_write() -> Result<()> {
    let (_dir, plan) = setup()?;
    let wrong = execute_migration(&plan, "wrong", MigrationAction::Apply, |_| {
        ControlFlow::Continue(())
    });
    ensure!(wrong.error == Some(MigrationFailure::InvalidPlan));
    ensure!(!plan.destination.exists());
    let mut altered = plan.clone();
    altered.destination = altered.destination.with_file_name("other");
    let report = perform(&altered, MigrationAction::Apply)?;
    ensure!(report.error == Some(MigrationFailure::InvalidPlan));
    ensure!(!altered.destination.exists());
    let mut unsafe_plan = plan.clone();
    let entry = unsafe_plan
        .entries
        .iter_mut()
        .find_map(|entry| entry.source.as_mut())
        .context("empty plan")?;
    entry.path = "../outside".into();
    unsafe_plan.revision = Some(unsafe_plan.content_revision()?);
    ensure!(
        perform(&unsafe_plan, MigrationAction::Apply)?.error == Some(MigrationFailure::InvalidPlan)
    );
    ensure!(!plan.destination.exists());
    Ok(())
}

#[test]
fn source_and_destination_changes_after_review_are_refused_without_writes() -> Result<()> {
    let (_dir, plan) = setup()?;
    let audio = plan.source.root.join("assets/audio.bin");
    let original = std::fs::read(&audio)?;
    std::fs::write(&audio, b"changed by fixture operator")?;
    ensure!(perform(&plan, MigrationAction::Apply)?.error == Some(MigrationFailure::SourceChanged));
    ensure!(!plan.destination.exists());
    std::fs::write(audio, &original)?;
    std::fs::create_dir(&plan.destination)?;
    std::fs::write(plan.destination.join("unknown"), b"preserve")?;
    let report = perform(&plan, MigrationAction::Apply)?;
    ensure!(report.error == Some(MigrationFailure::DestinationChanged));
    ensure!(std::fs::read(plan.destination.join("unknown"))? == b"preserve");
    ensure!(!plan.destination.join(".voqx-migration").exists());
    Ok(())
}

#[test]
fn dry_validation_and_active_writer_locks_do_not_start_migration() -> Result<()> {
    let (_dir, plan) = setup()?;
    let report = perform(&plan, MigrationAction::Validate)?;
    ensure!(report.outcome == MigrationOutcome::Validated);
    ensure!(!report.destination_may_have_changed && !plan.destination.exists());
    let lock = voqx_core::atomic::LockGuard::acquire(
        &plan.source.root.join(".lock"),
        std::time::Duration::ZERO,
    )?;
    let report = perform(&plan, MigrationAction::Apply)?;
    ensure!(report.error == Some(MigrationFailure::Busy));
    ensure!(!plan.destination.exists());
    drop(lock);
    ensure!(perform(&plan, MigrationAction::Apply)?.outcome == MigrationOutcome::Completed);
    Ok(())
}

#[test]
fn late_destination_collision_is_never_overwritten() -> Result<()> {
    let (_dir, plan) = setup()?;
    let mut injected = false;
    let report = execute_migration(
        &plan,
        plan.revision.as_deref().context("missing revision")?,
        MigrationAction::Apply,
        |point| {
            if !injected && matches!(point, MigrationCheckpoint::Staged(_)) {
                injected = std::fs::write(
                    plan.destination.join("assets/audio.bin"),
                    b"external writer's data",
                )
                .is_ok();
            }
            ControlFlow::Continue(())
        },
    );
    ensure!(
        injected && report.error == Some(MigrationFailure::DestinationChanged),
        "{report:?}"
    );
    ensure!(std::fs::read(plan.destination.join("assets/audio.bin"))? == b"external writer's data");
    ensure!(!plan.destination.join("registry.json").exists());
    Ok(())
}

#[test]
fn missing_references_and_reserved_source_data_never_become_executable_plans() -> Result<()> {
    let (_dir, plan) = setup()?;
    std::fs::remove_file(plan.source.root.join("assets/text.txt"))?;
    let broken = MigrationPlan::inspect(plan.source.root.clone(), plan.destination.clone());
    ensure!(perform(&broken, MigrationAction::Apply)?.error == Some(MigrationFailure::InvalidPlan));
    ensure!(!plan.destination.exists());
    std::fs::write(
        plan.source.root.join("assets/text.txt"),
        b"Synthetic transcript\n",
    )?;
    std::fs::create_dir(plan.source.root.join(".voqx-migration"))?;
    let reserved = MigrationPlan::inspect(plan.source.root, plan.destination.clone());
    ensure!(
        perform(&reserved, MigrationAction::Apply)?.error == Some(MigrationFailure::InvalidPlan)
    );
    ensure!(!plan.destination.exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn symlink_destination_cannot_redirect_a_reviewed_copy() -> Result<()> {
    let (dir, plan) = setup()?;
    let elsewhere = dir.0.join("elsewhere");
    std::fs::create_dir(&elsewhere)?;
    std::os::unix::fs::symlink(&elsewhere, &plan.destination)?;
    ensure!(perform(&plan, MigrationAction::Apply)?.error == Some(MigrationFailure::InvalidPlan));
    ensure!(std::fs::read_dir(elsewhere)?.count() == 0);
    Ok(())
}
