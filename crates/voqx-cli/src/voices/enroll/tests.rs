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
    attach_recognition(&mut identity, &paths, &source, vec![0.5, 0.5], valid)?;
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
    assert_eq!(profile.embeddings, vec![vec![0.0, 1.0], vec![0.5, 0.5]]);
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
