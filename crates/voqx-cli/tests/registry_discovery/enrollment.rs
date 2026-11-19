//! Generated audio and isolated registry only. No inference or download.

use super::Scratch;
use anyhow::{Context, Result, ensure};

fn wav(dir: &Scratch) -> Result<String> {
    let path = dir.0.join("reviewed.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&path, spec)?;
    for _ in 0..16_000 {
        writer.write_sample(1000_i16)?;
    }
    writer.finalize()?;
    Ok(path.to_string_lossy().into_owned())
}

#[test]
fn enrollment_guards_and_preparation_failures_write_nothing() -> Result<()> {
    let dir = Scratch::new()?;
    let mut cases = vec![
        (
            vec![
                "enroll",
                "Sensitive fixture",
                "--audio",
                "missing.wav",
                "--json",
            ],
            "confirmation_required",
        ),
        (
            vec![
                "enroll",
                "Sensitive fixture",
                "--audio",
                "missing.wav",
                "--json",
                "--yes",
                "--dry-run",
            ],
            "dry_run_unsupported",
        ),
        (
            vec![
                "enroll",
                "Sensitive fixture",
                "--audio",
                "missing.wav",
                "--json",
                "--yes",
            ],
            "preparation_failed",
        ),
    ];
    if !cfg!(feature = "recognition") {
        cases.push((
            vec![
                "enroll",
                "Sensitive fixture",
                "--audio",
                "missing.wav",
                "--json",
                "--yes",
                "--recognition",
            ],
            "recognition_not_compiled",
        ));
    }
    for (args, code) in cases {
        let output = dir.invoke(&args)?;
        ensure!(!output.status.success());
        let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        ensure!(
            report
                == serde_json::json!({
                    "protocol":"voqx-enrollment/1", "outcome":"failed", "code":code,
                    "registry_effect":"not_started", "assets_may_have_changed":false,
                }),
            "unexpected receipt: {report}"
        );
        ensure!(!String::from_utf8(output.stderr)?.contains("Sensitive fixture"));
    }
    dir.unchanged()
}

#[test]
fn source_only_receipts_bind_actual_commits_and_stale_updates_write_no_assets() -> Result<()> {
    let dir = Scratch::new()?;
    let audio = wav(&dir)?;
    let root = dir.0.join("data/voice-registry");
    let paths = voqx_core::RegistryPaths::from_root(root.clone());
    let output = dir.run(&[
        "enroll",
        "Synthetic",
        "--audio",
        &audio,
        "--yes",
        "--json",
        "--quiet",
    ])?;
    let first: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    ensure!(first["outcome"] == "committed");
    let id = first["voice_id"].as_str().context("missing voice_id")?;
    let revision = first["registry_revision"]
        .as_str()
        .context("missing revision")?;
    let registry = voqx_core::VoiceRegistry::load(&paths)?;
    let identity = registry.find(id).context("receipt identity absent")?;
    ensure!(identity.sources.len() == 1);
    ensure!(identity.recognition_profiles.is_empty() && identity.synthesis_profiles.is_empty());
    ensure!(first["preprocessing"].is_null() && first["recognition_engine"].is_null());
    ensure!(first["synthesis_engine"].is_null());
    let probe = voqx_core::registry_discovery::RegistryProbe::inspect(root.clone());
    ensure!(probe.revision.as_deref() == Some(revision));
    let args = [
        "enroll",
        "Ignored",
        "--id",
        id,
        "--audio",
        &audio,
        "--yes",
        "--yaml",
        "--quiet",
        "--expected-revision",
        revision,
    ];
    let output = dir.run(&args)?;
    let second: serde_yaml::Value = serde_yaml::from_slice(&output.stdout)?;
    ensure!(second["outcome"] == "committed");
    ensure!(second["voice_id"] == id);
    ensure!(second["source_id"] == "src-2");
    let before = std::fs::read(&paths.registry_file)?;
    let output = dir.invoke(&args)?;
    ensure!(!output.status.success());
    let failed: serde_yaml::Value = serde_yaml::from_slice(&output.stdout)?;
    ensure!(failed["code"] == "stale_revision");
    ensure!(failed["registry_effect"] == "not_started");
    ensure!(failed["assets_may_have_changed"] == false);
    ensure!(std::fs::read(&paths.registry_file)? == before);
    ensure!(!paths.identity_dir(id).join("sources/src-3.wav").exists());
    ensure!(!root.join(".lock").exists());
    for name in ["config", "state", "cache", "hf"] {
        ensure!(!dir.0.join(name).exists());
    }
    Ok(())
}

#[test]
fn supplied_unreadable_transcript_is_not_silently_ignored() -> Result<()> {
    let dir = Scratch::new()?;
    let audio = wav(&dir)?;
    let output = dir.invoke(&[
        "enroll",
        "Synthetic",
        "--audio",
        &audio,
        "--transcript-file",
        "missing.txt",
        "--yes",
        "--json",
        "--quiet",
    ])?;
    ensure!(!output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    ensure!(report["code"] == "preparation_failed");
    ensure!(report["assets_may_have_changed"] == false);
    ensure!(!dir.0.join("data").exists());
    Ok(())
}

#[test]
fn failed_asset_phase_is_reported_as_uncertain_without_false_commit() -> Result<()> {
    let dir = Scratch::new()?;
    let audio = wav(&dir)?;
    let root = dir.0.join("data/voice-registry");
    let paths = voqx_core::RegistryPaths::from_root(root.clone());
    let mut registry = voqx_core::VoiceRegistry::default();
    let id = registry.create_identity("Synthetic");
    registry.save(&paths)?;
    let before = std::fs::read(&paths.registry_file)?;
    let synthesis_dir = paths.identity_dir(&id).join("synthesis");
    std::fs::create_dir_all(&synthesis_dir)?;
    std::fs::write(
        synthesis_dir.join("fixture-engine"),
        b"block directory creation",
    )?;
    let output = dir.invoke(&[
        "enroll",
        "Ignored",
        "--id",
        &id,
        "--audio",
        &audio,
        "--transcript",
        "Sensitive synthetic text",
        "--synthesis",
        "fixture-engine",
        "--yes",
        "--json",
        "--quiet",
    ])?;
    ensure!(!output.status.success());
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    ensure!(
        report
            == serde_json::json!({"protocol":"voqx-enrollment/1", "outcome":"failed", "code":"registry_update_failed", "registry_effect":"uncertain", "assets_may_have_changed":true})
    );
    ensure!(std::fs::read(&paths.registry_file)? == before);
    ensure!(paths.identity_dir(&id).join("sources/src-1.wav").exists());
    ensure!(!root.join(".lock").exists());
    ensure!(!String::from_utf8(output.stderr)?.contains("Sensitive synthetic text"));
    Ok(())
}

#[test]
fn readiness_reports_unchecked_runtime_model_and_calibration_without_state() -> Result<()> {
    let dir = Scratch::new()?;
    let output = dir.run(&[
        "registry",
        "readiness",
        "--preprocessing",
        "valid-v1",
        "--json",
    ])?;
    let report: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    ensure!(report["protocol"] == "voqx-recognition-readiness/1");
    ensure!(report["recognition_compiled"] == cfg!(feature = "recognition"));
    ensure!(report["registry"]["state"] == "missing");
    ensure!(report["profile_state"] == "not_checked");
    ensure!(report["engine"] == "speakrs-wespeaker-valid-v1");
    for key in ["runtime_state", "model_state", "calibration_state"] {
        ensure!(report[key] == "not_checked");
    }
    ensure!(report["ready_to_filter"] == false);
    dir.unchanged()
}
