//! Actual local graph proof with generated-signal fixtures; no speaker-accuracy claim.
#![cfg(feature = "fastdiar")]

#[path = "../src/test_support.rs"]
mod support;
use support::check_eq;

use std::{fs, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use voqx_recognition::{
    fastdiar::{
        DIMENSIONS, FastdiarEncoder, FastdiarWindowEncoder, MODEL_BYTES, STEP_SAMPLES,
        embedding_space,
    },
    speaker::{SpeakerError, WindowEncoder},
};

fn floats(path: PathBuf) -> Result<Vec<f32>> {
    let bytes = fs::read(path)?;
    anyhow::ensure!(bytes.len().is_multiple_of(4), "fixture must be exact f32le");
    bytes
        .chunks_exact(4)
        .map(|value| Ok(f32::from_le_bytes(value.try_into()?)))
        .collect()
}

fn parity(actual: &[f32], expected: &[f32]) -> Result<(f64, f64)> {
    anyhow::ensure!(
        actual.len() == expected.len(),
        "component coverage mismatch"
    );
    let errors: Vec<_> = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| f64::from((a - b).abs()))
        .collect();
    let max = errors.iter().copied().fold(0.0_f64, f64::max);
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    anyhow::ensure!(
        max <= 1e-4 && mean <= 1e-5,
        "unchanged parity gate: max={max}, mean={mean}"
    );
    Ok((max, mean))
}

#[test]
fn bad_graphs_are_rejected_before_loading_onnx() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("not-model.onnx");
    anyhow::ensure!(FastdiarEncoder::load(&path).is_err());
    fs::write(&path, [0_u8; 16])?;
    anyhow::ensure!(matches!(
        FastdiarEncoder::load(&path),
        Err(SpeakerError::Backend(_))
    ));
    fs::write(&path, vec![0_u8; MODEL_BYTES])?;
    anyhow::ensure!(matches!(
        FastdiarEncoder::load(&path),
        Err(SpeakerError::Backend(_))
    ));
    Ok(())
}

#[test]
#[ignore = "requires VOQX_FASTDIAR_MODEL and VOQX_FASTDIAR_FIXTURE containing existing synthetic parity files"]
fn actual_cpu_graph_matches_torch_frames_resets_and_window_pooling() -> Result<()> {
    let model = PathBuf::from(
        std::env::var_os("VOQX_FASTDIAR_MODEL").context("explicit model path required")?,
    );
    let fixture = PathBuf::from(
        std::env::var_os("VOQX_FASTDIAR_FIXTURE").context("explicit fixture directory required")?,
    );
    let input = floats(fixture.join("input.f32"))?;
    let expected = floats(fixture.join("expected.f32"))?;
    anyhow::ensure!(
        input.len() >= 24_320 && input.len().is_multiple_of(STEP_SAMPLES),
        "fixture input coverage"
    );
    anyhow::ensure!(
        expected.len() == input.len() / STEP_SAMPLES * DIMENSIONS,
        "fixture frame coverage"
    );

    let mut encoder = FastdiarEncoder::load(&model)?;
    let started = Instant::now();
    let mut frames = Vec::new();
    for chunk in input.chunks(13_001) {
        frames.extend(encoder.push(chunk)?);
    }
    let elapsed = started.elapsed();
    check_eq(&(encoder.processed_samples()), &(input.len() as u64))?;
    check_eq(&(encoder.pending_samples()), &(0))?;
    let actual: Vec<_> = frames
        .iter()
        .flat_map(|frame| frame.values().iter().copied())
        .collect();
    let (max, mean) = parity(&actual, &expected)?;
    for (index, frame) in frames.iter().enumerate() {
        check_eq(&(frame.end_sample), &(((index + 1) * STEP_SAMPLES) as u64))?;
    }

    encoder.reset()?;
    anyhow::ensure!(encoder.push(&input[..997])?.is_empty());
    anyhow::ensure!(encoder.push(&[f32::NAN]).is_err());
    check_eq(
        &(encoder.processed_samples(), encoder.pending_samples()),
        &(0, 997),
    )?;
    let first = encoder.push(&input[997..STEP_SAMPLES])?;
    check_eq(&(first[0].values()), &(frames[0].values()))?;
    encoder.reset()?;
    let partial = encoder.push(&input[..24_321])?;
    check_eq(
        &(
            partial.len(),
            encoder.processed_samples(),
            encoder.pending_samples(),
        ),
        &(19, 24_320, 1),
    )?;
    let final_frame = encoder.push(&vec![0.0; STEP_SAMPLES - 1])?;
    check_eq(
        &(
            final_frame.len(),
            encoder.processed_samples(),
            encoder.pending_samples(),
        ),
        &(1, 25_600, 0),
    )?;

    let mut window = FastdiarWindowEncoder::load(&model)?;
    check_eq(&(window.space()), &(&embedding_space()))?;
    for invalid in [
        vec![0.1; 24_319],
        vec![0.1; 160_001],
        vec![0.0; 24_320],
        vec![f32::NAN; 24_320],
    ] {
        anyhow::ensure!(window.encode(&invalid).is_err());
    }
    let pooled = window.encode(&input[..24_320])?;
    let mut oracle = vec![0.0_f64; DIMENSIONS];
    for frame in expected.chunks_exact(DIMENSIONS).take(19) {
        for (sum, value) in oracle.iter_mut().zip(frame) {
            *sum += f64::from(*value);
        }
    }
    for value in &mut oracle {
        *value /= 19.0;
    }
    let norm = oracle.iter().map(|value| value * value).sum::<f64>().sqrt();
    let oracle: Vec<_> = oracle
        .into_iter()
        .map(|value| (value / norm) as f32)
        .collect();
    let (pool_max, pool_mean) = parity(pooled.values(), &oracle)?;
    window.encode(&input[..24_321])?;
    check_eq(
        &(window.encode(&input[..24_320])?.values()),
        &(pooled.values()),
    )?;
    #[cfg(feature = "registry")]
    check_native_composition(window, &pooled, &input[..24_320])?;
    println!(
        "CPU generated-signal proof: {} frames, max={max:e}, mean={mean:e}, pooled_max={pool_max:e}, pooled_mean={pool_mean:e}, fragmented_stream_ms={}",
        frames.len(),
        elapsed.as_millis()
    );
    Ok(())
}

#[cfg(feature = "registry")]
fn check_native_composition(
    encoder: FastdiarWindowEncoder,
    reference: &voqx_recognition::speaker::SpeakerEmbedding,
    pcm: &[f32],
) -> Result<()> {
    use voqx_core::{VoiceIdentity, VoiceRegistry};
    use voqx_recognition::{
        speaker::{
            SpeakerAttributor, SpeakerMatcher, SpeakerObservation, SpeakerStatus, SpeechActivity,
        },
        speaker_registry::{recognition_profile, reference_snapshot},
    };
    let mut voice = VoiceIdentity::new("vox-generated-signal", "Generated signal");
    voice
        .recognition_profiles
        .push(recognition_profile(std::slice::from_ref(reference))?);
    let registry = VoiceRegistry {
        identities: vec![voice],
        ..VoiceRegistry::default()
    };
    let before = serde_json::to_vec(&registry)?;
    let snapshot = reference_snapshot(&registry, encoder.space(), &["vox-generated-signal"])?;
    let matcher = SpeakerMatcher::new(snapshot)?;
    let mut attribution = SpeakerAttributor::new(encoder, matcher, 1, 16_000)?;
    let observation = SpeakerObservation {
        generation: 1,
        start_sample: 64_000,
        end_sample: 64_000 + pcm.len() as u64,
        activity: SpeechActivity::Speech,
    };
    let evidence = attribution.process(observation, pcm)?;
    check_eq(
        &(evidence.status, evidence.voice_id),
        &(SpeakerStatus::Uncalibrated, None),
    )?;
    check_eq(&evidence.observation, &observation)?;
    anyhow::ensure!(
        evidence
            .candidates
            .first()
            .context("missing candidate")?
            .nearest
            > 0.999_999
    );
    check_eq(&serde_json::to_vec(&registry)?, &before)?;
    Ok(())
}
