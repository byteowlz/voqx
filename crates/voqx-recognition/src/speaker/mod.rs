//! Portable, bounded known-speaker attribution; no inference or storage is required.
//!
//! A snapshot is a temporary view of explicitly selected canonical voice profiles,
//! not another identity registry. Thresholds must name its exact fingerprint.
//! Unknown speakers are not clustered or enrolled. Activity is supplied by the host;
//! this module does not detect speech, music or overlap.

mod bank;
mod engine;
mod tracker;

use serde::{Deserialize, Serialize};

pub use bank::SpeakerMatcher;
pub use engine::{SpeakerAttributor, WindowEncoder};
pub use tracker::SpeakerTracker;

/// Version of the timestamped evidence wire contract.
pub const EVIDENCE_FORMAT: &str = "voqx-speaker-evidence/1";

/// Failures never become low cosine scores or permission to bypass recognition.
#[derive(Debug, thiserror::Error)]
pub enum SpeakerError {
    /// Invalid samples, vector, timestamp or host-supplied metadata.
    #[error("invalid speaker input: {0}")]
    InvalidInput(&'static str),
    /// Invalid or incompatible reference profiles.
    #[error("invalid recognition profile: {0}")]
    InvalidProfile(&'static str),
    /// Missing or inconsistent decision-policy configuration.
    #[error("invalid speaker policy: {0}")]
    InvalidPolicy(&'static str),
    /// An observation belongs to an inactive capture, bank or policy revision.
    #[error("stale speaker observation")]
    StaleObservation,
    /// Timestamps did not advance within the current capture.
    #[error("speaker observation is out of order")]
    OutOfOrder,
    /// The optional encoder could not load or perform inference.
    #[error("speaker backend unavailable: {0}")]
    Backend(String),
}

/// Result returned by the portable and native speaker interfaces.
pub type SpeakerResult<T> = Result<T, SpeakerError>;

/// Exact compatibility descriptor, including checkpoint, preprocessing and storage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingSpace {
    /// Engine/profile slot in the canonical registry.
    pub engine: String,
    /// Immutable checkpoint/preprocessing/pooling/storage version.
    pub model_version: String,
    /// Number of vector components.
    pub dimensions: usize,
    /// Rate of the host-supplied mono PCM.
    pub sample_rate: u32,
    /// Operational observation floor, including any real silence.
    pub minimum_samples: usize,
    /// Maximum observation size; longer input is rejected, never truncated.
    pub maximum_samples: usize,
}

impl EmbeddingSpace {
    pub(crate) fn validate(&self) -> SpeakerResult<()> {
        if self.engine.trim().is_empty()
            || self.model_version.trim().is_empty()
            || self.engine.len() > 128
            || self.model_version.len() > 256
            || !(1..=1024).contains(&self.dimensions)
            || !(1..=192_000).contains(&self.sample_rate)
            || self.minimum_samples == 0
            || self.minimum_samples > self.maximum_samples
            || self.maximum_samples as u64 > u64::from(self.sample_rate) * 60
        {
            return Err(SpeakerError::InvalidProfile("invalid embedding space"));
        }
        Ok(())
    }
}

/// A validated embedding. Debug formatting deliberately omits voiceprint values.
#[derive(Clone)]
pub struct SpeakerEmbedding {
    space: EmbeddingSpace,
    values: Vec<f32>,
}

impl std::fmt::Debug for SpeakerEmbedding {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeakerEmbedding")
            .field("space", &self.space)
            .finish_non_exhaustive()
    }
}

impl SpeakerEmbedding {
    /// Validate a vector in an explicitly named space; never infer a model from size.
    ///
    /// # Errors
    /// Rejects invalid descriptors and zero, nonfinite or wrong-dimensional vectors.
    pub fn new(space: EmbeddingSpace, values: Vec<f32>) -> SpeakerResult<Self> {
        space.validate()?;
        bank::normalized(&values, space.dimensions)?;
        Ok(Self { space, values })
    }

    /// Descriptor that must exactly match the enrolled snapshot.
    #[must_use]
    pub const fn space(&self) -> &EmbeddingSpace {
        &self.space
    }

    /// Voiceprint values for explicit enrollment or same-host inference transport.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Explicit reference vectors belonging to an existing canonical identity.
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceVoice {
    /// Canonical `voice_id`, never a display-name lookup.
    pub voice_id: String,
    /// Reviewed references from one compatible recognition profile.
    pub embeddings: Vec<Vec<f32>>,
}

impl std::fmt::Debug for ReferenceVoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReferenceVoice")
            .field("voice_id", &self.voice_id)
            .field("references", &self.embeddings.len())
            .finish_non_exhaustive()
    }
}

/// Read-only, caller-selected view of profiles; no paths, consent mutation or I/O.
///
/// Serialization includes sensitive voiceprints and is an explicit caller operation.
/// It is never performed automatically or sent to a server by this crate.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReferenceSnapshot {
    /// Exact space shared by every reference.
    pub space: EmbeddingSpace,
    /// At most 64 identities, each with 1–32 references.
    pub voices: Vec<ReferenceVoice>,
}

/// Caller-declared calibration scope. Supplying an ID is not empirical qualification.
///
/// There are deliberately no default numeric thresholds. This record is bound to
/// exact reference bytes/order and an observation-duration band. The host owns
/// independent speaker/session qualification and its downstream admission policy.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerCalibration {
    /// Version of the externally qualified operating point.
    pub id: String,
    /// Fingerprint returned by [`SpeakerMatcher::reference_revision`].
    pub reference_revision: String,
    /// Inclusive duration-band floor, in the profile's sample clock.
    pub minimum_samples: usize,
    /// Inclusive duration-band ceiling.
    pub maximum_samples: usize,
    /// Nearest-reference cosine needed for confident evidence, before guard.
    pub confident: f64,
    /// Nearest-reference cosine needed for tentative evidence, before guard.
    pub tentative: f64,
    /// Winner must exceed the runner-up by strictly more than this plus guard.
    pub minimum_margin: f64,
    /// Explicit numerical safety margin; never silently chosen by the library.
    pub numerical_guard: f64,
}

/// Host-provided activity evidence. This library does not infer it from cosine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechActivity {
    /// Host has speech evidence for this interval, without claiming speaker purity.
    Speech,
    /// Host has no-speech evidence.
    NoSpeech,
    /// Host reports overlapping speakers; attribution must abstain.
    Overlap,
    /// No usable activity decision is available.
    Unknown,
}

/// Source interval and generation in the host's resampled audio clock.
///
/// Observation duration is not detected speech duration. Zero-length intervals
/// are valid insufficient-audio evidence. Integers must fit `JavaScript`'s exact range.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpeakerObservation {
    /// Monotonically increasing capture generation, fenced on stop/reset.
    pub generation: u64,
    /// Inclusive source sample position.
    pub start_sample: u64,
    /// Exclusive source sample position; padding is never included.
    pub end_sample: u64,
    /// Host-supplied activity, not an encoder/VAD result.
    pub activity: SpeechActivity,
}

impl SpeakerObservation {
    pub(crate) const fn validate(self) -> SpeakerResult<()> {
        const MAX_EXACT: u64 = (1_u64 << 53) - 1;
        if self.end_sample < self.start_sample
            || self.end_sample > MAX_EXACT
            || self.generation > MAX_EXACT
        {
            return Err(SpeakerError::InvalidInput("invalid observation clock"));
        }
        Ok(())
    }

    /// Observed samples, excluding final encoder padding.
    #[must_use]
    pub const fn samples(self) -> u64 {
        self.end_sample.saturating_sub(self.start_sample)
    }
}

/// Open-set attribution state, not an authentication or authorization decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeakerStatus {
    /// Calibrated winner; streaming consumers must also apply stabilization.
    Confident,
    /// Subthreshold or still stabilizing; no resolved `voice_id`.
    Tentative,
    /// No supported match or no usable activity decision.
    Unknown,
    /// Too little observed audio.
    InsufficientAudio,
    /// Query and reference spaces differ.
    IncompatibleModel,
    /// No calibration, or observation outside its declared duration band.
    Uncalibrated,
    /// Close competitors; no closest-person naming.
    Ambiguous,
    /// Host reported no speech.
    NoSpeech,
    /// Host reported overlapping speakers.
    Overlap,
}

/// Diagnostic scores for a candidate identity. Cosines are not probabilities.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerCandidate {
    /// Canonical identity; a candidate is not a resolved speaker.
    pub voice_id: String,
    /// Maximum cosine over this identity's references; the ranking score.
    pub nearest: f64,
    /// Cosine against this identity's normalized centroid; None if references cancel.
    pub centroid: Option<f64>,
}

/// Timestamped evidence. Audio and reference vectors are deliberately absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerEvidence {
    /// Wire-contract version; not a speaker-accuracy qualification.
    pub format: String,
    /// Exact capture generation, source interval and host activity.
    pub observation: SpeakerObservation,
    /// Exact compatible model and processing contract.
    pub space: EmbeddingSpace,
    /// Immutable fingerprint of the selected model/reference snapshot.
    pub reference_revision: String,
    /// Externally supplied calibration identifier, if configured.
    pub calibration_id: Option<String>,
    /// Exact policy-content fingerprint, including thresholds, guard and duration band.
    pub calibration_revision: Option<String>,
    /// Only populated for confident evidence, never tentative/unknown/ambiguous.
    pub voice_id: Option<String>,
    /// Classified outcome; not ASR admission, intent or tool authority.
    pub status: SpeakerStatus,
    /// Ranked diagnostic candidates; not forced labels.
    pub candidates: Vec<SpeakerCandidate>,
}
