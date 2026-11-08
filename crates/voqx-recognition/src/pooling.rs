//! Pooling geometry pinned to speakrs 0.4.2, not an activity or speaker mask.

use anyhow::{Result, ensure};
use voqx_core::RecognitionPreprocessing;

use crate::{MIN_IDENTIFY_SAMPLES, WINDOW_SAMPLES};

const MASK_FRAMES: usize = 589;
const EMBEDDING_DIM: usize = 256;

pub fn valid_mask(samples: usize) -> [f32; MASK_FRAMES] {
    let observed = samples.min(WINDOW_SAMPLES);
    // Match the probe's floor(i * 160000 / 589) grid. The ONNX graph then
    // nearest-resizes weights for pooling; encoder padding itself is unchanged.
    std::array::from_fn(|index| {
        if index * WINDOW_SAMPLES / MASK_FRAMES < observed {
            1.0
        } else {
            0.0
        }
    })
}

pub fn input_window(
    samples: &[f32],
    model_minimum: usize,
    preprocessing: RecognitionPreprocessing,
) -> Result<&[f32]> {
    ensure!(
        samples.iter().all(|s| s.is_finite()),
        "recognition audio contains nonfinite samples"
    );
    let observed = &samples[..samples.len().min(WINDOW_SAMPLES)];
    let minimum = if preprocessing == RecognitionPreprocessing::ValidV1 {
        model_minimum.max(MIN_IDENTIFY_SAMPLES)
    } else {
        model_minimum
    };
    ensure!(
        observed.len() >= minimum,
        "too little audio to embed: {} samples, need at least {minimum}",
        observed.len()
    );
    ensure!(
        observed.iter().any(|s| *s != 0.0),
        "cannot enroll or embed digital silence"
    );
    Ok(observed)
}

pub fn validate_embedding(vector: &[f32]) -> Result<()> {
    ensure!(
        vector.len() == EMBEDDING_DIM,
        "embedding dimension is not {EMBEDDING_DIM}"
    );
    ensure!(
        vector.iter().all(|v| v.is_finite()) && vector.iter().any(|v| *v != 0.0),
        "embedding model returned a nonfinite or zero vector"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        RecognitionPreprocessing::{Uniform, ValidV1},
        test_support::check_eq,
    };

    #[test]
    fn valid_mask_matches_probe_geometry_and_excludes_only_padding() {
        for (samples, bins) in [
            (0, 0),
            (16_000, 59),
            (24_000, 89),
            (41_600, 154),
            (160_000, 589),
            (200_000, 589),
        ] {
            let mut expected = [0.0; 589];
            expected[..bins].fill(1.0);
            assert_eq!(
                valid_mask(samples).map(f32::to_bits),
                expected.map(f32::to_bits)
            );
        }
    }

    #[test]
    fn silence_invalid_audio_and_short_valid_windows_are_rejected() {
        for audio in [
            vec![],
            vec![0.0; 16_000],
            vec![f32::NAN; 16_000],
            vec![f32::INFINITY; 16_000],
            vec![0.1; 15_999],
        ] {
            assert!(input_window(&audio, 400, ValidV1).is_err());
        }
        assert!(input_window(&[0.1; 800], 400, Uniform).is_ok());
        assert!(input_window(&vec![0.1; 16_000], 32_000, ValidV1).is_err());
    }

    #[test]
    fn only_the_leading_window_is_observed_but_invalid_tails_error() -> Result<()> {
        let mut samples = vec![0.1; WINDOW_SAMPLES + 100];
        check_eq(
            input_window(&samples, 400, ValidV1)?,
            &samples[..WINDOW_SAMPLES],
        )?;
        samples[WINDOW_SAMPLES] = f32::NAN;
        anyhow::ensure!(input_window(&samples, 400, ValidV1).is_err());
        Ok(())
    }

    #[test]
    fn invalid_embeddings_cannot_reach_enrollment_or_matching() -> Result<()> {
        for vector in [
            vec![],
            vec![0.0; 256],
            vec![f32::NAN; 256],
            vec![f32::INFINITY; 256],
            vec![0.1; 96],
        ] {
            anyhow::ensure!(validate_embedding(&vector).is_err());
        }
        validate_embedding(&[0.1; 256])
    }
}
