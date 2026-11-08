//! `voqx identify` — recognition evidence for an audio file.
//!
//! Output is evidence only (`voice_id`, score, status). Authorization decisions
//! belong to the consuming application.

use std::fmt::Write;
use std::path::Path;

use anyhow::Result;
use voqx_core::{Recognition, RecognitionPreprocessing, Thresholds};

use crate::audio::{self, RECOGNITION_SAMPLE_RATE};

use super::open_store;

/// Run `voqx identify`.
///
/// # Errors
///
/// Returns an error if the audio cannot be read, the model cannot be loaded, or
/// the registry cannot be read.
pub fn run(
    path: &Path,
    root: Option<&Path>,
    mode: &str,
    preprocessing: RecognitionPreprocessing,
    thresholds: Thresholds,
    json: bool,
) -> Result<()> {
    let source = audio::read_wav(path)?;
    let mono16 = source.to_rate(RECOGNITION_SAMPLE_RATE)?;

    let store = open_store(root)?;
    let registry = store.load()?;

    let mut embedder =
        voqx_recognition::Embedder::new(mode, None)?.with_preprocessing(preprocessing);
    let evidence = embedder.identify(&registry, &mono16.samples, thresholds)?;

    println!("{}", render_evidence(&evidence, json)?);
    Ok(())
}

fn render_evidence(evidence: &Recognition, json: bool) -> Result<String> {
    if json {
        return Ok(serde_json::to_string_pretty(evidence)?);
    }
    let mut output = match (evidence.voice_id.as_deref(), evidence.score) {
        (Some(matched_id), Some(similarity)) => format!(
            "{matched_id} ({}) score {similarity:.3} [{:?}]",
            evidence.display_name.as_deref().unwrap_or("-"),
            evidence.status
        ),
        _ => format!("no match [{:?}]", evidence.status),
    };
    write!(
        output,
        "\nsample score {}; centroid score {} (uncalibrated)",
        render_score(evidence.score_sample),
        render_score(evidence.score_centroid)
    )?;
    if let Some(audio) = &evidence.audio {
        write!(
            output,
            "\naudio {:.3}s; observed {:.3}s; minimum {:.3}s; padded {}; truncated {}",
            audio.input_duration_seconds,
            audio.observed_duration_seconds,
            audio.minimum_duration_seconds,
            audio.padded,
            audio.truncated
        )?;
    }
    Ok(output)
}

fn render_score(score: Option<f32>) -> String {
    score.map_or_else(|| "-".into(), |value| format!("{value:.3}"))
}

#[cfg(test)]
mod tests {
    use super::render_evidence;
    use voqx_core::{AudioObservation, Recognition, RecognitionStatus};

    #[test]
    fn human_and_json_output_preserve_scores_and_scope() -> anyhow::Result<()> {
        let evidence = Recognition {
            voice_id: Some("vox-fixture".into()),
            display_name: Some("Fixture".into()),
            engine: "test-engine".into(),
            score: Some(0.9),
            score_sample: Some(0.9),
            score_centroid: Some(0.5),
            audio: Some(AudioObservation {
                sample_rate: 16_000,
                input_duration_seconds: 12.0,
                observed_duration_seconds: 10.0,
                minimum_duration_seconds: 1.0,
                padded: false,
                truncated: true,
                calibration_band: None,
            }),
            status: RecognitionStatus::Confident,
        };
        assert_eq!(
            render_evidence(&evidence, false)?,
            "vox-fixture (Fixture) score 0.900 [Confident]\nsample score 0.900; centroid score 0.500 (uncalibrated)\naudio 12.000s; observed 10.000s; minimum 1.000s; padded false; truncated true"
        );
        let json = render_evidence(&evidence, true)?;
        assert_eq!(serde_json::from_str::<Recognition>(&json)?, evidence);
        Ok(())
    }

    #[test]
    fn insufficient_audio_has_no_identity_or_score() -> anyhow::Result<()> {
        assert_eq!(
            render_evidence(&Recognition::insufficient_audio("test-engine"), false)?,
            "no match [InsufficientAudio]\nsample score -; centroid score - (uncalibrated)"
        );
        Ok(())
    }
}
