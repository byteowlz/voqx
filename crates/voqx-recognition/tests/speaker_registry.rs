//! Registry adapters do not write files, mutate identities or merge profile spaces.
#![cfg(feature = "registry")]

#[path = "../src/test_support.rs"]
mod support;
use support::check_eq;

use voqx_core::{RecognitionProfile, VoiceIdentity, VoiceRegistry};
use voqx_recognition::{
    speaker::{EmbeddingSpace, SpeakerEmbedding},
    speaker_registry::{recognition_profile, reference_snapshot},
};

fn space() -> EmbeddingSpace {
    EmbeddingSpace {
        engine: "fixture".into(),
        model_version: "cold-v1".into(),
        dimensions: 2,
        sample_rate: 16_000,
        minimum_samples: 4,
        maximum_samples: 16,
    }
}

fn registry() -> VoiceRegistry {
    let mut voice = VoiceIdentity::new("vox-a", "Same display name");
    voice.recognition_profiles.push(RecognitionProfile::new(
        "fixture",
        "cold-v1",
        vec![vec![1.0, 0.0]],
        vec![],
    ));
    let second = VoiceIdentity::new("vox-b", "Same display name");
    VoiceRegistry {
        identities: vec![voice, second],
        ..VoiceRegistry::default()
    }
}

#[test]
fn snapshot_selection_is_by_id_and_does_not_mutate_registry() -> anyhow::Result<()> {
    let registry = registry();
    let before = serde_json::to_vec(&registry)?;
    let snapshot = reference_snapshot(&registry, &space(), &["vox-a"])?;
    check_eq(&(snapshot.voices.len()), &(1))?;
    check_eq(&(snapshot.voices[0].voice_id), &("vox-a"))?;
    check_eq(&(serde_json::to_vec(&registry)?), &(before))?;
    anyhow::ensure!(
        reference_snapshot(&registry, &space(), &[])?
            .voices
            .is_empty()
    );
    anyhow::ensure!(reference_snapshot(&registry, &space(), &["Same display name"]).is_err());
    anyhow::ensure!(reference_snapshot(&registry, &space(), &["vox-a", "vox-b"]).is_err());
    anyhow::ensure!(reference_snapshot(&registry, &space(), &["vox-a", "vox-a"]).is_err());
    Ok(())
}

#[test]
fn stale_version_duplicate_slot_and_wrong_dimensions_are_rejected() {
    for change in 0..4 {
        let mut registry = registry();
        let voice = &mut registry.identities[0];
        match change {
            0 => voice.recognition_profiles[0].model_version = "continuous-v1".into(),
            1 => voice
                .recognition_profiles
                .push(voice.recognition_profiles[0].clone()),
            2 => voice.recognition_profiles[0].embeddings = vec![vec![1.0; 256]],
            _ => voice.recognition_profiles[0].embeddings.clear(),
        }
        assert!(reference_snapshot(&registry, &space(), &["vox-a"]).is_err());
    }
}

#[test]
fn profile_builder_requires_exact_compatible_vectors_and_does_not_add_synthesis()
-> anyhow::Result<()> {
    let a = SpeakerEmbedding::new(space(), vec![1.0, 0.0])?;
    let b = SpeakerEmbedding::new(space(), vec![0.0, 1.0])?;
    let profile = recognition_profile(&[a.clone(), b])?;
    check_eq(
        &(
            &profile.engine,
            &profile.model_version,
            profile.embeddings.len(),
        ),
        &(&space().engine, &space().model_version, 2),
    )?;
    anyhow::ensure!(profile.samples.is_empty());
    anyhow::ensure!(recognition_profile(&[]).is_err());
    let mut other = space();
    other.model_version.push_str("-other");
    let incompatible = SpeakerEmbedding::new(other, vec![1.0, 0.0])?;
    anyhow::ensure!(recognition_profile(&[a, incompatible]).is_err());
    Ok(())
}
