//! Storage seam test with injected vectors, not an acoustic accuracy test.

use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

use super::{Audio, RecognitionPreprocessing, RegistryPaths, VoiceIdentity, attach_recognition};

#[test]
fn adding_masked_samples_preserves_uniform_profile_and_assets() -> Result<()> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = std::env::temp_dir().join(format!("voqx-pooling-{}-{nonce}", std::process::id()));
    std::fs::create_dir(&root)?;
    let paths = RegistryPaths::from_root(root.clone());
    let mut identity = VoiceIdentity::new("vox-fixture", "Fixture");
    let source = Audio {
        samples: vec![0.1; 1600],
        sample_rate: 16_000,
    };
    let uniform = RecognitionPreprocessing::Uniform;
    let valid = RecognitionPreprocessing::ValidV1;
    attach_recognition(&mut identity, &paths, &source, vec![1.0, 0.0], uniform)?;
    let original = serde_json::to_value(identity.recognition_profile(uniform.engine_id()))?;
    let uniform_path =
        root.join("identities/vox-fixture/recognition/speakrs-wespeaker/sample-1.wav");
    let original_audio = std::fs::read(&uniform_path)?;
    attach_recognition(&mut identity, &paths, &source, vec![0.0, 1.0], valid)?;
    // Real encoders emit unit vectors; the fixture must satisfy the same rule.
    let unit = std::f32::consts::FRAC_1_SQRT_2;
    attach_recognition(&mut identity, &paths, &source, vec![unit, unit], valid)?;
    assert_eq!(identity.recognition_profiles.len(), 2);
    assert_eq!(
        serde_json::to_value(identity.recognition_profile(uniform.engine_id()))?,
        original
    );
    assert_eq!(std::fs::read(&uniform_path)?, original_audio);
    let profile = identity
        .recognition_profile(valid.engine_id())
        .context("masked profile missing")?;
    assert_eq!(profile.model_version, valid.model_version());
    assert_eq!(profile.embeddings, vec![vec![0.0, 1.0], vec![unit, unit]]);
    assert_eq!(
        profile.samples,
        vec![
            "identities/vox-fixture/recognition/speakrs-wespeaker-valid-v1/sample-1.wav",
            "identities/vox-fixture/recognition/speakrs-wespeaker-valid-v1/sample-2.wav",
        ]
    );
    for path in &profile.samples {
        assert_eq!(std::fs::read(root.join(path))?, original_audio);
    }
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[test]
fn injected_recognition_receipt_preserves_other_profiles_and_binds_commit() -> Result<()> {
    use super::{EnrollmentOutcome, PreparedEnrollment, commit};
    use crate::{Cli, Command};
    use clap::Parser;
    use voqx_core::{RecognitionProfile, RegistryStore, SynthesisProfile, WakeProfile};

    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let root = std::env::temp_dir().join(format!("voqx-receipt-{}-{nonce}", std::process::id()));
    let paths = RegistryPaths::from_root(root.clone());
    let store = RegistryStore::new(paths.clone());
    let uniform = RecognitionPreprocessing::Uniform;
    let selected = RecognitionPreprocessing::ValidV1;
    let seed = store.edit_with_revision(None, |registry| {
        let id = registry.create_identity("Synthetic");
        let identity = registry.find_mut(&id).context("missing new identity")?;
        identity.recognition_profiles.push(RecognitionProfile::new(
            uniform.engine_id(),
            uniform.model_version(),
            vec![vec![1.0, 0.0]],
            vec![],
        ));
        identity
            .synthesis_profiles
            .push(SynthesisProfile::new("fixture-tts"));
        identity.wake_profiles.push(WakeProfile {
            id: "wake-fixture".into(),
            phrase: "Synthetic phrase".into(),
            backend: "fixture".into(),
            artifact: None,
            tags: vec![],
            created_at_unix: 1,
            updated_at_unix: 1,
        });
        Ok(id)
    })?;
    let before = store.load()?;
    let original = before.find(&seed.value).context("missing seed")?;
    let Command::Enroll(mut args) = Cli::try_parse_from([
        "voqx",
        "enroll",
        "Ignored",
        "--id",
        &seed.value,
        "--audio",
        "unused-injected.wav",
        "--recognition",
        "--preprocessing",
        "valid-v1",
        "--expected-revision",
        &seed.revision,
    ])?
    .command
    else {
        anyhow::bail!("expected enrollment");
    };
    let prepared = PreparedEnrollment {
        source: Audio {
            samples: vec![0.1; 16_000],
            sample_rate: 16_000,
        },
        transcript: None,
        embedding: Some(vec![0.0, 1.0]),
    };
    let outcome = commit(&args, Some(&root), &prepared)?;
    let value = serde_json::to_value(&outcome)?;
    let EnrollmentOutcome::Committed {
        voice_id,
        registry_revision,
        recognition_engine,
        ..
    } = outcome
    else {
        anyhow::bail!("expected commit");
    };
    anyhow::ensure!(voice_id == seed.value);
    anyhow::ensure!(recognition_engine.as_deref() == Some(selected.engine_id()));
    anyhow::ensure!(value["synthesis_engine"].is_null());
    anyhow::ensure!(
        voqx_core::registry_discovery::RegistryProbe::inspect(root.clone())
            .revision
            .as_deref()
            == Some(registry_revision.as_str())
    );
    let after = store.load()?;
    let identity = after.find(&voice_id).context("receipt identity absent")?;
    anyhow::ensure!(
        serde_json::to_value(&identity.synthesis_profiles)?
            == serde_json::to_value(&original.synthesis_profiles)?
    );
    anyhow::ensure!(
        serde_json::to_value(&identity.wake_profiles)?
            == serde_json::to_value(&original.wake_profiles)?
    );
    anyhow::ensure!(
        serde_json::to_value(identity.recognition_profile(uniform.engine_id()))?
            == serde_json::to_value(original.recognition_profile(uniform.engine_id()))?
    );
    anyhow::ensure!(
        identity
            .recognition_profile(selected.engine_id())
            .context("missing selected profile")?
            .embeddings
            == vec![vec![0.0, 1.0]]
    );
    let bytes = std::fs::read(&paths.registry_file)?;
    args.expected_revision = Some(seed.revision);
    let error = commit(&args, Some(&root), &prepared)
        .err()
        .context("stale update unexpectedly succeeded")?;
    anyhow::ensure!(error.is::<voqx_core::RegistryRevisionConflict>());
    anyhow::ensure!(std::fs::read(&paths.registry_file)? == bytes);
    anyhow::ensure!(
        !paths
            .identity_dir(&voice_id)
            .join("sources/src-2.wav")
            .exists()
    );
    store.edit(|registry| {
        let identity = registry
            .find_mut(&voice_id)
            .context("missing fixture identity")?;
        let profile = identity
            .recognition_profiles
            .iter_mut()
            .find(|profile| profile.engine == selected.engine_id())
            .context("missing selected profile")?;
        profile.model_version = "wrong-model-contract".into();
        Ok(())
    })?;
    let before_mismatch = std::fs::read(&paths.registry_file)?;
    args.expected_revision = None;
    anyhow::ensure!(commit(&args, Some(&root), &prepared).is_err());
    anyhow::ensure!(std::fs::read(&paths.registry_file)? == before_mismatch);
    anyhow::ensure!(
        !paths
            .identity_dir(&voice_id)
            .join("sources/src-2.wav")
            .exists()
    );
    std::fs::remove_dir_all(root)?;
    Ok(())
}
