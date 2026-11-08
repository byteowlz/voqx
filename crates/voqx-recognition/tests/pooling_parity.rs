//! Real CPU graph parity on generated signals. Not speaker-accuracy evidence.

#![cfg(feature = "wespeaker")]

#[path = "../src/test_support.rs"]
mod support;
use support::check_eq;

use std::path::PathBuf;

use anyhow::{Context, Result};
use speakrs::{ExecutionMode, inference::EmbeddingModel};
use voqx_recognition::{Embedder, RecognitionPreprocessing, WINDOW_SAMPLES};

fn assert_near(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), 256);
    assert_eq!(expected.len(), 256);
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
}

#[test]
#[ignore = "requires VOQX_TEST_MODEL_DIR pointing to existing local model files"]
fn masked_and_legacy_paths_match_the_pinned_cpu_graph() -> Result<()> {
    let dir = PathBuf::from(
        std::env::var_os("VOQX_TEST_MODEL_DIR").context("set explicit local model dir")?,
    );
    let mut direct = EmbeddingModel::with_mode(
        dir.join("wespeaker-voxceleb-resnet34.onnx"),
        ExecutionMode::Cpu,
    )?;
    let mut legacy = Embedder::new("cpu", Some(&dir))?;
    let mut masked =
        Embedder::new("cpu", Some(&dir))?.with_preprocessing(RecognitionPreprocessing::ValidV1);
    check_eq(
        &(legacy.engine_id()),
        &(RecognitionPreprocessing::Uniform.engine_id()),
    )?;
    check_eq(
        &(masked.engine_id()),
        &(RecognitionPreprocessing::ValidV1.engine_id()),
    )?;
    let pcm: Vec<f32> = (0..WINDOW_SAMPLES)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            // Preserve the legacy fixture's unfused arithmetic and exact samples.
            let ramp = t * 30.0;
            let chirp = 0.2 * (t * (900.0 + ramp)).sin();
            let tone = 0.1 * (t * 1700.0).sin();
            chirp + tone
        })
        .collect();
    for (samples, valid_bins) in [(16_000, 59), (24_000, 89), (41_600, 154), (160_000, 589)] {
        let audio = &pcm[..samples];
        let expected_uniform = direct.embed(audio)?.to_vec();
        assert_near(&legacy.embed(audio)?, &expected_uniform);
        let mut mask = vec![0.0; 589];
        mask[..valid_bins].fill(1.0);
        let expected_masked = direct.embed_masked(audio, &mask, None)?.to_vec();
        assert_near(&masked.embed(audio)?, &expected_masked);
        if samples == WINDOW_SAMPLES {
            assert_near(&expected_uniform, &expected_masked);
        }
    }
    anyhow::ensure!(masked.embed(&pcm[..15_999]).is_err());
    anyhow::ensure!(masked.embed(&vec![0.0; 16_000]).is_err());
    Ok(())
}
