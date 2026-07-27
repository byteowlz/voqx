//! Recognition *evidence*: matching an embedding against the registry.
//!
//! This module returns who best matches and how confidently — never an
//! authorization decision. Command/authorization policy (fail-closed on unknown
//! speaker, confirmation for destructive actions) lives downstream in
//! eaRS/Foxline. Voice similarity is not authentication.

use serde::{Deserialize, Serialize};

/// Cosine-similarity thresholds for classifying a match.
///
/// Defaults follow trnscrb's measured WeSpeaker guidance
/// (`docs/speaker-embeddings.md`): confident `>= 0.60`, tentative `>= 0.45`.
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
    /// The utterance did not carry enough speech to identify.
    InsufficientAudio,
    /// No registry profile exists for the requested embedding engine/model.
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
    /// Best cosine score in `[-1, 1]`, or `None` when no candidate applied.
    pub score: Option<f32>,
    /// Classified status.
    pub status: RecognitionStatus,
}

impl Recognition {
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
            status,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{RecognitionStatus, Thresholds};

    #[test]
    fn classify_bands() {
        let t = Thresholds::default();
        assert_eq!(t.classify(0.8), RecognitionStatus::Confident);
        assert_eq!(t.classify(0.5), RecognitionStatus::Tentative);
        assert_eq!(t.classify(0.1), RecognitionStatus::Unknown);
    }
}
