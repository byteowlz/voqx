//! Short-duration contracts and actual graph execution, not speaker-accuracy evidence.
#![cfg(feature = "fastdiar")]

use anyhow::{Context, Result};
use voqx_recognition::{
    fastdiar::{
        DIMENSIONS, ENGINE_ID, FastdiarEncoder, FastdiarWindowEncoder, MODEL_SHA256,
        PHRASE_ENGINE_ID, STEP_SAMPLES, embedding_space, phrase_embedding_space,
    },
    speaker::{
        EmbeddingSpace, ReferenceSnapshot, ReferenceVoice, SpeakerCalibration, SpeakerEmbedding,
        SpeakerMatcher, SpeakerObservation, SpeakerStatus, SpeechActivity, WindowEncoder,
    },
};

fn snapshot(space: EmbeddingSpace) -> ReferenceSnapshot {
    let mut reference = vec![0.0; DIMENSIONS];
    reference[0] = 1.0;
    ReferenceSnapshot {
        space,
        voices: vec![ReferenceVoice {
            voice_id: "vox-test".into(),
            embeddings: vec![reference],
        }],
    }
}

const fn observation(samples: usize) -> SpeakerObservation {
    SpeakerObservation {
        generation: 1,
        start_sample: 32_000,
        end_sample: 32_000 + samples as u64,
        activity: SpeechActivity::Speech,
    }
}

fn policy(matcher: &SpeakerMatcher, confident: f64) -> SpeakerCalibration {
    SpeakerCalibration {
        id: "synthetic-tuning-not-qualified".into(),
        reference_revision: matcher.reference_revision().into(),
        minimum_samples: matcher.space().minimum_samples,
        maximum_samples: matcher.space().maximum_samples,
        confident,
        tentative: 0.4,
        minimum_margin: 0.05,
        numerical_guard: 0.0,
    }
}

#[test]
fn legacy_descriptor_is_unchanged() -> Result<()> {
    anyhow::ensure!(
        embedding_space()
            == EmbeddingSpace {
                engine: ENGINE_ID.into(),
                model_version: format!("cold80-mean-f64-l2-f32-v1:{MODEL_SHA256}"),
                dimensions: 192,
                sample_rate: 16_000,
                minimum_samples: 24_320,
                maximum_samples: 160_000,
            }
    );
    Ok(())
}

#[test]
fn phrase_bounds_are_explicit_bounded_and_versioned() -> Result<()> {
    for (min, max) in [
        (0, 160_000),
        (1279, 160_000),
        (9600, 9599),
        (1280, 160_001),
        (usize::MAX, usize::MAX),
    ] {
        anyhow::ensure!(phrase_embedding_space(min, max).is_err());
    }
    for (min, max) in [
        (1280, 1280),
        (4800, 160_000),
        (9600, 160_000),
        (12800, 64000),
    ] {
        let space = phrase_embedding_space(min, max)?;
        anyhow::ensure!(space.engine == PHRASE_ENGINE_ID);
        anyhow::ensure!(space.minimum_samples == min && space.maximum_samples == max);
        anyhow::ensure!(
            space
                .model_version
                .ends_with(&format!(":phrase-v1:{min}-{max}"))
        );
        anyhow::ensure!(
            serde_json::from_str::<EmbeddingSpace>(&serde_json::to_string(&space)?)? == space
        );
    }
    Ok(())
}

#[test]
fn short_duration_boundaries_and_missing_policy_abstain() -> Result<()> {
    let bank = snapshot(phrase_embedding_space(4800, 12_800)?);
    let query = SpeakerEmbedding::new(bank.space.clone(), bank.voices[0].embeddings[0].clone())?;
    let matcher = SpeakerMatcher::new(bank)?;
    for samples in [4800, 9600, 12_800] {
        let evidence = matcher.evaluate(observation(samples), Some(&query))?;
        anyhow::ensure!(evidence.status == SpeakerStatus::Uncalibrated);
        anyhow::ensure!(evidence.voice_id.is_none());
    }
    let below = matcher.evaluate(observation(4799), None)?;
    anyhow::ensure!(below.status == SpeakerStatus::InsufficientAudio && below.voice_id.is_none());
    anyhow::ensure!(matcher.evaluate(observation(12_801), Some(&query)).is_err());
    Ok(())
}

#[test]
fn changed_duration_band_cannot_reuse_vectors_or_calibration() -> Result<()> {
    let space = phrase_embedding_space(4800, 160_000)?;
    let original = SpeakerMatcher::new(snapshot(space.clone()))?;
    let calibration = policy(&original, 0.9);
    for other in [
        embedding_space(),
        phrase_embedding_space(9600, 160_000)?,
        phrase_embedding_space(4800, 64_000)?,
    ] {
        let matcher = SpeakerMatcher::new(snapshot(other))?;
        anyhow::ensure!(original.reference_revision() != matcher.reference_revision());
        let query = SpeakerEmbedding::new(space.clone(), vec![1.0; DIMENSIONS])?;
        let evidence = matcher.evaluate(observation(24_320), Some(&query))?;
        anyhow::ensure!(
            evidence.status == SpeakerStatus::IncompatibleModel && evidence.voice_id.is_none()
        );
        anyhow::ensure!(matcher.with_calibration(calibration.clone()).is_err());
    }
    Ok(())
}

#[test]
fn explicit_strictness_changes_decisions_and_policy_revision() -> Result<()> {
    let bank = snapshot(phrase_embedding_space(4800, 160_000)?);
    let mut values = vec![0.0; DIMENSIONS];
    values[0] = 0.8;
    values[1] = 0.6;
    let query = SpeakerEmbedding::new(bank.space.clone(), values)?;
    let matcher = SpeakerMatcher::new(bank.clone())?;
    let permissive = policy(&matcher, 0.7);
    let strict = policy(&matcher, 0.9);
    let accepted = matcher
        .with_calibration(permissive)?
        .evaluate(observation(4800), Some(&query))?;
    let abstained = SpeakerMatcher::new(bank)?
        .with_calibration(strict)?
        .evaluate(observation(4800), Some(&query))?;
    anyhow::ensure!(
        accepted.status == SpeakerStatus::Confident
            && accepted.voice_id.as_deref() == Some("vox-test")
    );
    anyhow::ensure!(abstained.status == SpeakerStatus::Tentative && abstained.voice_id.is_none());
    anyhow::ensure!(accepted.calibration_revision != abstained.calibration_revision);
    Ok(())
}

#[cfg(feature = "registry")]
#[test]
fn registry_profiles_cannot_silently_cross_phrase_bands() -> Result<()> {
    use voqx_core::{VoiceIdentity, VoiceRegistry};
    use voqx_recognition::speaker_registry::{recognition_profile, reference_snapshot};
    let bank = snapshot(phrase_embedding_space(4800, 160_000)?);
    let reference =
        SpeakerEmbedding::new(bank.space.clone(), bank.voices[0].embeddings[0].clone())?;
    let mut voice = VoiceIdentity::new("vox-test", "Synthetic fixture");
    voice
        .recognition_profiles
        .push(recognition_profile(&[reference])?);
    let registry = VoiceRegistry {
        identities: vec![voice],
        ..VoiceRegistry::default()
    };
    reference_snapshot(&registry, &bank.space, &["vox-test"])?;
    for other in [embedding_space(), phrase_embedding_space(9600, 160_000)?] {
        anyhow::ensure!(reference_snapshot(&registry, &other, &["vox-test"]).is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires VOQX_FASTDIAR_MODEL; generated tones prove inference, not speaker accuracy"]
fn real_graph_supports_short_spans_with_unchanged_pooling_and_reset() -> Result<()> {
    let model = std::env::var_os("VOQX_FASTDIAR_MODEL").context("explicit local graph required")?;
    let path = std::path::Path::new(&model);
    let mut window = FastdiarWindowEncoder::load_phrase(path, STEP_SAMPLES, 16_000)?;
    let mut raw = FastdiarEncoder::load(path)?;
    for length in [1280_usize, 4800, 9600, 12_800, 16_000] {
        let pcm: Vec<_> = (0..length).map(|i| 0.1 * (i as f32 * 0.08).sin()).collect();
        let started = std::time::Instant::now();
        let actual = window.encode(&pcm)?;
        let elapsed = started.elapsed();
        raw.reset()?;
        let mut frames = raw.push(&pcm)?;
        if raw.pending_samples() != 0 {
            frames.extend(raw.push(&vec![0.0; STEP_SAMPLES - raw.pending_samples()])?);
        }
        let mut oracle = vec![0.0_f64; DIMENSIONS];
        for frame in &frames {
            for (sum, value) in oracle.iter_mut().zip(frame.values()) {
                *sum += f64::from(*value);
            }
        }
        for value in &mut oracle {
            *value /= frames.len() as f64;
        }
        let norm = oracle.iter().map(|value| value * value).sum::<f64>().sqrt();
        let expected: Vec<_> = oracle
            .into_iter()
            .map(|value| (value / norm) as f32)
            .collect();
        anyhow::ensure!(actual.values() == expected);
        anyhow::ensure!(window.encode(&pcm)?.values() == actual.values());
        println!(
            "short-span generated-signal proof: samples={length} source_ms={} frames={} cpu_ms={:.3}",
            length as f64 / 16.0,
            frames.len(),
            elapsed.as_secs_f64() * 1000.0
        );
    }
    for invalid in [
        vec![0.1; 1279],
        vec![0.1; 16_001],
        vec![0.0; 4800],
        vec![f32::NAN; 4800],
    ] {
        anyhow::ensure!(window.encode(&invalid).is_err());
    }
    Ok(())
}
