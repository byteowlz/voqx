//! Synthetic filesystem tests; no model or personal registry is used.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use super::*;
use crate::{RecognitionProfile, SourceRecording, VoiceIdentity, VoiceRegistry};

fn check_eq<T: std::fmt::Debug + PartialEq>(actual: &T, expected: &T) -> Result<()> {
    anyhow::ensure!(
        actual == expected,
        "actual: {actual:?}; expected: {expected:?}"
    );
    Ok(())
}

#[derive(Debug)]
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Result<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let path = std::env::temp_dir().join(format!(
            "voqx-discovery-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&path)?;
        Ok(Self(path))
    }
    fn root(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn fixture(root: &Path) -> Result<()> {
    let paths = crate::RegistryPaths::from_root(root.to_path_buf());
    let mut identity = VoiceIdentity::new("vox-synthetic", "Synthetic subject");
    identity.sources.push(SourceRecording {
        id: "src-1".into(),
        audio: "assets/source.wav".into(),
        transcript: Some("assets/source.txt".into()),
        language: Some("en".into()),
        note: String::new(),
        created_at_unix: 1,
    });
    identity.recognition_profiles.push(RecognitionProfile::new(
        "fixture-engine",
        "fixture-model",
        vec![vec![1.0, 0.0]],
        vec!["assets/source.wav".into()],
    ));
    let registry = VoiceRegistry {
        version: 1,
        identities: vec![identity],
    };
    std::fs::create_dir_all(root.join("assets/empty"))?;
    std::fs::write(
        root.join("assets/source.wav"),
        b"synthetic frozen PCM bytes",
    )?;
    std::fs::write(root.join("assets/source.txt"), b"Synthetic transcript\n")?;
    registry.save(&paths)?;
    Ok(())
}

fn copy_fixture(source: &Path, target: &Path) -> Result<()> {
    std::fs::create_dir_all(target.join("assets/empty"))?;
    for file in ["registry.json", "assets/source.wav", "assets/source.txt"] {
        std::fs::copy(source.join(file), target.join(file))?;
    }
    Ok(())
}

#[test]
fn discovery_keeps_legacy_selected_in_all_root_combinations() -> Result<()> {
    let dir = Scratch::new()?;
    let legacy = dir.root("legacy");
    let proposed = dir.root("proposed");
    for (have_legacy, have_proposed) in [(false, false), (false, true), (true, true), (true, false)]
    {
        if have_legacy {
            fixture(&legacy)?;
        } else if legacy.exists() {
            std::fs::remove_dir_all(&legacy)?;
        }
        if have_proposed {
            fixture(&proposed)?;
        } else if proposed.exists() {
            std::fs::remove_dir_all(&proposed)?;
        }
        check_eq(
            &RegistryDiscovery::inspect(legacy.clone(), proposed.clone(), None),
            &RegistryDiscovery {
                protocol: "voqx-registry-discovery/1".into(),
                selection: RootSelection::LegacyDefault,
                effective: RegistryProbe::inspect(legacy.clone()),
                proposed: Some(RegistryProbe::inspect(proposed.clone())),
                ready_to_filter: false,
            },
        )?;
    }
    let explicit = dir.root("explicit");
    fixture(&explicit)?;
    check_eq(
        &RegistryDiscovery::inspect(legacy, proposed, Some(explicit.clone())),
        &RegistryDiscovery {
            protocol: "voqx-registry-discovery/1".into(),
            selection: RootSelection::Explicit,
            effective: RegistryProbe::inspect(explicit),
            proposed: None,
            ready_to_filter: false,
        },
    )?;
    Ok(())
}

#[test]
fn probe_distinguishes_missing_invalid_unsupported_busy_and_unsafe() -> Result<()> {
    let dir = Scratch::new()?;
    let root = dir.root("registry");
    check_eq(
        &RegistryProbe::inspect(root.clone()).state,
        &RegistryState::Missing,
    )?;
    std::fs::create_dir(&root)?;
    for (bytes, state, version) in [
        ("not JSON", RegistryState::Invalid, None),
        (
            "{\"version\":2}",
            RegistryState::UnsupportedVersion,
            Some(2),
        ),
        (
            "{\"version\":0}",
            RegistryState::UnsupportedVersion,
            Some(0),
        ),
        (
            "{\"version\":1,\"identities\":[{}]}",
            RegistryState::Invalid,
            Some(1),
        ),
    ] {
        std::fs::write(root.join("registry.json"), bytes)?;
        check_eq(
            &RegistryProbe::inspect(root.clone()),
            &RegistryProbe {
                root: root.clone(),
                state,
                version,
                revision: None,
                identity_count: None,
            },
        )?;
    }
    fixture(&root)?;
    std::fs::write(root.join(".lock"), b"synthetic lock")?;
    check_eq(
        &RegistryProbe::inspect(root.clone()).state,
        &RegistryState::Busy,
    )?;
    std::fs::remove_file(root.join(".lock"))?;
    std::fs::remove_file(root.join("registry.json"))?;
    std::fs::create_dir(root.join("registry.json"))?;
    check_eq(
        &RegistryProbe::inspect(root).state,
        &RegistryState::UnsafePath,
    )?;
    Ok(())
}

#[test]
fn registry_size_is_bounded_before_parsing() -> Result<()> {
    let dir = Scratch::new()?;
    let root = dir.root("registry");
    std::fs::create_dir(&root)?;
    std::fs::File::create(root.join("registry.json"))?.set_len(MAX_REGISTRY_BYTES + 1)?;
    check_eq(
        &RegistryProbe::inspect(root).state,
        &RegistryState::LimitExceeded,
    )?;
    Ok(())
}

#[test]
fn plan_preserves_exact_bytes_hashes_and_empty_directories_without_writes() -> Result<()> {
    let dir = Scratch::new()?;
    let source = dir.root("source");
    let target = dir.root("target");
    fixture(&source)?;
    let before = inventory::scan(&source).map_err(|issue| anyhow::anyhow!("{issue:?}"))?;
    let plan = MigrationPlan::inspect(source.clone(), target.clone());
    check_eq(&plan.state, &PlanState::CopyRequired)?;
    check_eq(&plan.blockers, &vec![])?;
    anyhow::ensure!(plan.apply_supported);
    check_eq(
        &plan.entries,
        &before
            .values()
            .cloned()
            .map(|entry| PlanEntry {
                source: Some(entry),
                destination: None,
                state: TransferState::Copy,
            })
            .collect::<Vec<_>>(),
    )?;
    check_eq(
        &before["assets/source.txt"].sha256,
        &Some(digest(b"Synthetic transcript\n")),
    )?;
    anyhow::ensure!(!target.exists());
    check_eq(&inventory::scan(&source), &Ok(before))?;
    let encoded = serde_json::to_string(&plan)?;
    anyhow::ensure!(!encoded.contains("Synthetic subject"));
    anyhow::ensure!(!encoded.contains("fixture-model"));
    Ok(())
}

#[test]
fn exact_copy_and_partial_copy_are_distinct_and_deterministic() -> Result<()> {
    let dir = Scratch::new()?;
    let source = dir.root("source");
    let target = dir.root("target");
    fixture(&source)?;
    copy_fixture(&source, &target)?;
    let exact = MigrationPlan::inspect(source.clone(), target.clone());
    check_eq(&exact.state, &PlanState::AlreadyPresent)?;
    anyhow::ensure!(
        exact
            .entries
            .iter()
            .all(|entry| entry.state == TransferState::AlreadyPresent)
    );
    check_eq(
        &MigrationPlan::inspect(source.clone(), target.clone()),
        &exact,
    )?;
    std::fs::remove_file(target.join("assets/source.txt"))?;
    let partial = MigrationPlan::inspect(source.clone(), target.clone());
    check_eq(&partial.state, &PlanState::CopyRequired)?;
    check_eq(
        &partial
            .entries
            .iter()
            .filter(|entry| entry.state == TransferState::Copy)
            .count(),
        &1,
    )?;
    anyhow::ensure!(!target.join("assets/source.txt").exists());
    check_eq(&MigrationPlan::inspect(source, target), &partial)?;
    Ok(())
}

#[test]
fn conflicts_and_destination_only_entries_block_without_overwrite_or_delete() -> Result<()> {
    let dir = Scratch::new()?;
    let source = dir.root("source");
    let target = dir.root("target");
    fixture(&source)?;
    copy_fixture(&source, &target)?;
    std::fs::write(target.join("assets/source.txt"), b"different")?;
    std::fs::write(target.join("extra"), b"preserve me")?;
    let plan = MigrationPlan::inspect(source, target.clone());
    check_eq(
        &(plan.state, plan.blockers),
        &(PlanState::Blocked, vec![PlanBlocker::DestinationConflict]),
    )?;
    anyhow::ensure!(
        plan.entries
            .iter()
            .any(|entry| entry.state == TransferState::Conflict)
    );
    anyhow::ensure!(
        plan.entries
            .iter()
            .any(|entry| entry.state == TransferState::DestinationOnly)
    );
    check_eq(
        &std::fs::read(target.join("assets/source.txt"))?,
        &b"different".to_vec(),
    )?;
    check_eq(
        &std::fs::read(target.join("extra"))?,
        &b"preserve me".to_vec(),
    )?;
    Ok(())
}

#[test]
fn absent_source_and_overlapping_roots_block() -> Result<()> {
    let dir = Scratch::new()?;
    let source = dir.root("source");
    check_eq(
        &MigrationPlan::inspect(source.clone(), dir.root("target")).blockers,
        &vec![PlanBlocker::SourceNotReady],
    )?;
    fixture(&source)?;
    for destination in [source.clone(), source.join("nested"), dir.0.clone()] {
        check_eq(
            &MigrationPlan::inspect(source.clone(), destination).blockers,
            &vec![PlanBlocker::OverlappingRoots],
        )?;
    }
    Ok(())
}

#[test]
fn missing_external_and_nonfile_asset_references_block() -> Result<()> {
    let dir = Scratch::new()?;
    let source = dir.root("source");
    let paths = crate::RegistryPaths::from_root(source.clone());
    fixture(&source)?;
    for reference in [
        "absent.wav",
        "../outside.wav",
        "/outside.wav",
        "C:\\outside.wav",
        "assets/empty",
    ] {
        let mut registry = VoiceRegistry::load(&paths)?;
        registry
            .identities
            .first_mut()
            .context("missing fixture identity")?
            .sources[0]
            .audio = reference.into();
        registry.save(&paths)?;
        check_eq(
            &MigrationPlan::inspect(source.clone(), dir.root("target")).blockers,
            &vec![PlanBlocker::InvalidReferences],
        )?;
    }
    Ok(())
}

#[test]
fn writer_lock_at_destination_blocks_even_without_registry() -> Result<()> {
    let dir = Scratch::new()?;
    let source = dir.root("source");
    let target = dir.root("target");
    fixture(&source)?;
    std::fs::create_dir(&target)?;
    std::fs::write(target.join(".lock"), b"lock")?;
    check_eq(
        &MigrationPlan::inspect(source, target).blockers,
        &vec![PlanBlocker::DestinationScan {
            issue: ScanIssue::Busy,
        }],
    )?;
    Ok(())
}

#[test]
fn inventory_depth_and_entry_count_are_bounded() -> Result<()> {
    let dir = Scratch::new()?;
    let deep = dir.root("deep");
    std::fs::create_dir_all(deep.join("d/".repeat(34)))?;
    check_eq(&inventory::scan(&deep), &Err(ScanIssue::LimitExceeded))?;
    let many = dir.root("many");
    std::fs::create_dir(&many)?;
    for i in 0..=inventory::MAX_ENTRIES {
        std::fs::write(many.join(format!("f{i}")), b"")?;
    }
    check_eq(&inventory::scan(&many), &Err(ScanIssue::LimitExceeded))?;
    Ok(())
}

#[test]
fn readiness_is_profile_specific_and_never_claims_model_or_filter_readiness() -> Result<()> {
    use crate::RecognitionPreprocessing as P;
    let dir = Scratch::new()?;
    let root = dir.root("registry");
    let report = RecognitionReadiness::inspect(root.clone(), P::Uniform);
    check_eq(
        &(
            report.registry.state,
            report.profile_state,
            report.matching_profiles,
        ),
        &(RegistryState::Missing, ProfileState::NotChecked, None),
    )?;
    fixture(&root)?;
    let paths = crate::RegistryPaths::from_root(root.clone());
    let mut registry = VoiceRegistry::load(&paths)?;
    let identity = registry
        .identities
        .first_mut()
        .context("missing identity")?;
    identity.recognition_profiles.push(RecognitionProfile::new(
        P::Uniform.engine_id(),
        P::Uniform.model_version(),
        vec![vec![1.0, 0.0]],
        vec![],
    ));
    registry.save(&paths)?;
    for (preprocessing, state, count) in [
        (P::Uniform, ProfileState::Present, 1),
        (P::ValidV1, ProfileState::Missing, 0),
    ] {
        let report = RecognitionReadiness::inspect(root.clone(), preprocessing);
        check_eq(
            &(report.profile_state, report.matching_profiles),
            &(state, Some(count)),
        )?;
        anyhow::ensure!(!report.ready_to_filter);
        check_eq(&report.runtime_state, &"not_checked".to_string())?;
        check_eq(&report.model_state, &"not_checked".to_string())?;
        check_eq(&report.calibration_state, &"not_checked".to_string())?;
    }
    registry
        .identities
        .first_mut()
        .context("missing identity")?
        .recognition_profiles[1]
        .model_version = "wrong-version".into();
    registry.save(&paths)?;
    let report = RecognitionReadiness::inspect(root, P::Uniform);
    check_eq(
        &(report.profile_state, report.matching_profiles),
        &(ProfileState::IncompatibleModel, Some(0)),
    )?;
    Ok(())
}

#[test]
fn committed_revisions_bind_exact_bytes_and_stale_updates_do_not_run() -> Result<()> {
    let dir = Scratch::new()?;
    let paths = crate::RegistryPaths::from_root(dir.root("registry"));
    let store = crate::RegistryStore::new(paths.clone());
    let first =
        store.edit_with_revision(None, |registry| Ok(registry.create_identity("Synthetic")))?;
    check_eq(
        &first.revision,
        &digest(&std::fs::read(&paths.registry_file)?),
    )?;
    check_eq(
        &RegistryProbe::inspect(paths.root.clone()).revision,
        &Some(first.revision.clone()),
    )?;
    let second = store.edit_with_revision(Some(&first.revision), |registry| {
        registry.rename(&first.value, "Renamed")
    })?;
    let before = std::fs::read(&paths.registry_file)?;
    anyhow::ensure!(second.revision != first.revision);
    let mut called = false;
    let failure = store.edit_with_revision(Some(&first.revision), |_| {
        called = true;
        Ok(())
    });
    anyhow::ensure!(
        failure
            .err()
            .is_some_and(|error| error.is::<crate::RegistryRevisionConflict>())
    );
    anyhow::ensure!(!called);
    check_eq(&std::fs::read(&paths.registry_file)?, &before)?;
    anyhow::ensure!(!paths.root.join(".lock").exists());
    Ok(())
}

#[cfg(unix)]
#[test]
fn symlink_roots_assets_and_dangling_links_are_rejected() -> Result<()> {
    use std::os::unix::fs::symlink;
    let dir = Scratch::new()?;
    let source = dir.root("source");
    fixture(&source)?;
    let alias = dir.root("alias");
    symlink(&source, &alias)?;
    check_eq(
        &RegistryProbe::inspect(alias).state,
        &RegistryState::UnsafePath,
    )?;
    symlink(dir.root("absent"), source.join("dangling"))?;
    check_eq(
        &MigrationPlan::inspect(source.clone(), dir.root("target")).blockers,
        &vec![PlanBlocker::SourceScan {
            issue: ScanIssue::UnsafeTree,
        }],
    )?;
    std::fs::remove_file(source.join("dangling"))?;
    symlink(source.join("assets/source.wav"), source.join("linked.wav"))?;
    check_eq(&inventory::scan(&source), &Err(ScanIssue::UnsafeTree))?;
    Ok(())
}
