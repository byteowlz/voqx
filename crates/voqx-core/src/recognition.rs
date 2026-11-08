//! Recognition *evidence*: matching an embedding against the registry.
//!
//! This module returns who best matches and how confidently — never an
//! authorization decision. Command/authorization policy (fail-closed on unknown
//! speaker, confirmation for destructive actions) lives downstream in
//! eaRS/Foxline. Voice similarity is not authentication.

use serde::{Deserialize, Serialize};

/// Cosine-similarity thresholds for classifying a match.
///
/// Legacy defaults are confident `>= 0.60`, tentative `>= 0.45`.
/// They are not calibrated for nearest-sample scoring, enrollment count,
/// channel, or duration. Callers must qualify thresholds for their use case.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Thresholds {
    /// At or above this cosine, a match is [`RecognitionStatus::Confident`].
    pub confident: f32,
    /// At or above this cosine (but below `confident`), a match is
    /// [`RecognitionStatus::Tentative`].
    pub tentative: f32,
}

impl Default for Thresholds {
    fn default() -> Self {
        Self {
            confident: 0.60,
            tentative: 0.45,
        }
    }
}

impl Thresholds {
    /// Classify a raw cosine score against these thresholds.
    #[must_use]
    pub fn classify(&self, score: f32) -> RecognitionStatus {
        if score >= self.confident {
            RecognitionStatus::Confident
        } else if score >= self.tentative {
            RecognitionStatus::Tentative
        } else {
            RecognitionStatus::Unknown
        }
    }
}

/// The qualitative outcome of a recognition attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecognitionStatus {
    /// Score at or above the confident threshold.
    Confident,
    /// Score in the tentative band.
    Tentative,
    /// Score below the tentative threshold.
    Unknown,
    /// Not enough usable audio was submitted for identification.
    InsufficientAudio,
    /// No registry profile exists for the requested embedding engine.
    /// This does not attest checkpoint compatibility.
    IncompatibleModel,
}

/// Recognition evidence for one utterance: the best matching identity (if any),
/// its cosine score, and the classified status.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Recognition {
    /// Matched `voice_id`, or `None` when no candidate applied.
    pub voice_id: Option<String>,
    /// Matched display name, mirrored for convenience.
    pub display_name: Option<String>,
    /// Embedding engine the match was computed against.
    pub engine: String,
    /// Ranking/classification score: nearest sample, or centroid for a legacy
    /// profile with no per-sample embeddings. Not a probability or authentication.
    pub score: Option<f32>,
    /// Maximum cosine against the matched identity's enrolled samples.
    #[serde(default)]
    pub score_sample: Option<f32>,
    /// Cosine against that same identity's centroid (not the best other identity).
    #[serde(default)]
    pub score_centroid: Option<f32>,
    /// Audio scope, when supplied by an audio-aware producer. Embedding-only
    /// callers cannot infer duration or usable speech from a vector.
    #[serde(default)]
    pub audio: Option<AudioObservation>,
    /// Classified status.
    pub status: RecognitionStatus,
}

/// Audio scope of a single embedding, not a speech-duration measurement.
///
/// No VAD, duration/channel calibration, or full-input consistency is implied.
/// Durations refer to mono audio at `sample_rate` Hz.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AudioObservation {
    /// Sample rate after caller-side conversion.
    pub sample_rate: u32,
    /// Total supplied audio duration, including silence.
    pub input_duration_seconds: f64,
    /// Duration of the leading audio actually observed (excludes padding).
    pub observed_duration_seconds: f64,
    /// Operational minimum duration; not a calibrated accuracy guarantee.
    pub minimum_duration_seconds: f64,
    /// Whether the embedding window was zero-padded.
    pub padded: bool,
    /// Whether the unobserved tail was truncated.
    pub truncated: bool,
    /// Qualified calibration band, or `None` when no calibration exists.
    /// Missing means uncalibrated, never a default calibrated band.
    pub calibration_band: Option<String>,
}

impl Recognition {
    /// Evidence when no usable vector comparison exists for a known engine.
    #[must_use]
    pub fn unknown(engine: impl Into<String>) -> Self {
        Self::without_candidate(engine, RecognitionStatus::Unknown)
    }

    /// Evidence for the case where the registry has no profile for `engine`.
    #[must_use]
    pub fn incompatible(engine: impl Into<String>) -> Self {
        Self::without_candidate(engine, RecognitionStatus::IncompatibleModel)
    }

    /// Evidence for an utterance too short to identify.
    #[must_use]
    pub fn insufficient_audio(engine: impl Into<String>) -> Self {
        Self::without_candidate(engine, RecognitionStatus::InsufficientAudio)
    }

    fn without_candidate(engine: impl Into<String>, status: RecognitionStatus) -> Self {
        Self {
            voice_id: None,
            display_name: None,
            engine: engine.into(),
            score: None,
            score_sample: None,
            score_centroid: None,
            audio: None,
            status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Recognition, RecognitionStatus, Thresholds};

    #[test]
    fn legacy_evidence_has_no_invented_scores_or_audio() -> anyhow::Result<()> {
        let json = r#"{"voice_id":null,"display_name":null,"engine":"test","score":null,"status":"unknown"}"#;
        let evidence: Recognition = serde_json::from_str(json)?;
        assert_eq!(evidence, Recognition::unknown("test"));
        let encoded = serde_json::to_string(&evidence)?;
        assert_eq!(serde_json::from_str::<Recognition>(&encoded)?, evidence);
        Ok(())
    }

    #[test]
    fn classify_bands() {
        let t = Thresholds::default();
        assert_eq!(t.classify(0.8), RecognitionStatus::Confident);
        assert_eq!(t.classify(0.5), RecognitionStatus::Tentative);
        assert_eq!(t.classify(0.1), RecognitionStatus::Unknown);
    }
}
