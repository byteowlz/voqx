use std::collections::HashSet;

use sha2::{Digest, Sha256};

use super::{
    EmbeddingSpace, ReferenceSnapshot, SpeakerCalibration, SpeakerCandidate, SpeakerEmbedding,
    SpeakerError, SpeakerEvidence, SpeakerObservation, SpeakerResult, SpeakerStatus,
    SpeechActivity,
};

struct VoiceVectors {
    voice_id: String,
    references: Vec<Vec<f64>>,
    centroid: Option<Vec<f64>>,
}

/// Immutable scoring view of an explicitly selected reference bank.
///
/// At most 64 voices, 32 references per voice and 1,048,576 total components are
/// accepted. One query embedding is shared across all candidate comparisons.
pub struct SpeakerMatcher {
    space: EmbeddingSpace,
    voices: Vec<VoiceVectors>,
    revision: String,
    calibration: Option<SpeakerCalibration>,
    calibration_revision: Option<String>,
}

impl std::fmt::Debug for SpeakerMatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpeakerMatcher")
            .field("space", &self.space)
            .field("voices", &self.voices.len())
            .field("reference_revision", &self.revision)
            .field("calibration", &self.calibration)
            .finish_non_exhaustive()
    }
}

impl SpeakerMatcher {
    /// Load a bounded snapshot without enabling confident decisions.
    ///
    /// # Errors
    /// Rejects invalid spaces, duplicate/empty IDs, excessive banks and invalid vectors.
    pub fn new(snapshot: ReferenceSnapshot) -> SpeakerResult<Self> {
        snapshot.space.validate()?;
        if snapshot.voices.len() > 64 {
            return Err(SpeakerError::InvalidProfile(
                "more than 64 reference voices",
            ));
        }
        let mut ids = HashSet::new();
        let mut components = 0_usize;
        let mut voices = Vec::with_capacity(snapshot.voices.len());
        for voice in &snapshot.voices {
            if voice.voice_id.trim().is_empty()
                || voice.voice_id.len() > 128
                || !ids.insert(&voice.voice_id)
                || !(1..=32).contains(&voice.embeddings.len())
            {
                return Err(SpeakerError::InvalidProfile(
                    "invalid, duplicate or oversized reference voice",
                ));
            }
            components = components
                .checked_add(voice.embeddings.len() * snapshot.space.dimensions)
                .ok_or(SpeakerError::InvalidProfile("reference component overflow"))?;
            if components > 1_048_576 {
                return Err(SpeakerError::InvalidProfile(
                    "reference component limit exceeded",
                ));
            }
            let references = voice
                .embeddings
                .iter()
                .map(|values| normalized(values, snapshot.space.dimensions))
                .collect::<SpeakerResult<Vec<_>>>()?;
            let centroid = reference_centroid(&references, snapshot.space.dimensions);
            voices.push(VoiceVectors {
                voice_id: voice.voice_id.clone(),
                references,
                centroid,
            });
        }
        let bytes = serde_json::to_vec(&snapshot)
            .map_err(|_| SpeakerError::InvalidProfile("snapshot serialization failed"))?;
        let revision = format!("{:x}", Sha256::digest(bytes));
        Ok(Self {
            space: snapshot.space,
            voices,
            revision,
            calibration: None,
            calibration_revision: None,
        })
    }

    /// Apply caller-declared thresholds bound to the exact model/reference snapshot.
    ///
    /// # Errors
    /// Rejects a different snapshot fingerprint, invalid values or duration scope.
    pub fn with_calibration(mut self, calibration: SpeakerCalibration) -> SpeakerResult<Self> {
        self.validate_calibration(&calibration)?;
        let bytes = serde_json::to_vec(&calibration)
            .map_err(|_| SpeakerError::InvalidPolicy("policy serialization failed"))?;
        self.calibration_revision = Some(format!("{:x}", Sha256::digest(bytes)));
        self.calibration = Some(calibration);
        Ok(self)
    }

    /// Exact preprocessing/model/input contract.
    #[must_use]
    pub const fn space(&self) -> &EmbeddingSpace {
        &self.space
    }

    /// SHA-256 of compact serde JSON for the exact space and ordered f32 references.
    #[must_use]
    pub fn reference_revision(&self) -> &str {
        &self.revision
    }

    /// External policy revision, not evidence that a real-world FAR gate passed.
    #[must_use]
    pub fn calibration_id(&self) -> Option<&str> {
        self.calibration
            .as_ref()
            .map(|calibration| calibration.id.as_str())
    }

    /// Exact calibration content, not just its caller-supplied name.
    #[must_use]
    pub fn calibration_revision(&self) -> Option<&str> {
        self.calibration_revision.as_deref()
    }

    /// Whether this observation needs an embedding; use before expensive inference.
    ///
    /// # Errors
    /// Rejects invalid clocks and observations exceeding the model's input bound.
    pub fn needs_embedding(&self, observation: SpeakerObservation) -> SpeakerResult<bool> {
        Ok(self.skip_status(observation)?.is_none())
    }

    /// Score a vector for its exact source interval. Activity/short-input abstentions
    /// need no vector. This method does not apply temporal stabilization.
    ///
    /// # Errors
    /// Rejects invalid clocks, oversized windows or missing required embeddings.
    pub fn evaluate(
        &self,
        observation: SpeakerObservation,
        embedding: Option<&SpeakerEmbedding>,
    ) -> SpeakerResult<SpeakerEvidence> {
        if let Some(status) = self.skip_status(observation)? {
            return Ok(self.evidence(observation, status));
        }
        let embedding = embedding.ok_or(SpeakerError::InvalidInput(
            "speech observation needs an embedding",
        ))?;
        if embedding.space() != &self.space {
            return Ok(self.evidence(observation, SpeakerStatus::IncompatibleModel));
        }
        let query = normalized(embedding.values(), self.space.dimensions)?;
        let mut candidates: Vec<_> = self
            .voices
            .iter()
            .map(|voice| SpeakerCandidate {
                voice_id: voice.voice_id.clone(),
                nearest: voice
                    .references
                    .iter()
                    .map(|reference| dot(&query, reference))
                    .fold(-1.0, f64::max),
                centroid: voice
                    .centroid
                    .as_ref()
                    .map(|centroid| dot(&query, centroid)),
            })
            .collect();
        candidates.sort_by(|left, right| {
            right
                .nearest
                .total_cmp(&left.nearest)
                .then_with(|| left.voice_id.cmp(&right.voice_id))
        });
        let status = self.classify(observation, &candidates);
        let mut evidence = self.evidence(observation, status);
        if status == SpeakerStatus::Confident {
            evidence.voice_id = candidates
                .first()
                .map(|candidate| candidate.voice_id.clone());
        }
        evidence.candidates = candidates;
        Ok(evidence)
    }

    pub(super) fn evidence(
        &self,
        observation: SpeakerObservation,
        status: SpeakerStatus,
    ) -> SpeakerEvidence {
        SpeakerEvidence {
            format: super::EVIDENCE_FORMAT.to_owned(),
            observation,
            space: self.space.clone(),
            reference_revision: self.revision.clone(),
            calibration_id: self.calibration_id().map(str::to_owned),
            calibration_revision: self.calibration_revision.clone(),
            voice_id: None,
            status,
            candidates: Vec::new(),
        }
    }

    fn skip_status(&self, observation: SpeakerObservation) -> SpeakerResult<Option<SpeakerStatus>> {
        observation.validate()?;
        if observation.samples() > self.space.maximum_samples as u64 {
            return Err(SpeakerError::InvalidInput(
                "observation exceeds model window; never truncated",
            ));
        }
        let status = match observation.activity {
            SpeechActivity::NoSpeech => Some(SpeakerStatus::NoSpeech),
            SpeechActivity::Overlap => Some(SpeakerStatus::Overlap),
            SpeechActivity::Unknown => Some(SpeakerStatus::Unknown),
            SpeechActivity::Speech if observation.samples() < self.space.minimum_samples as u64 => {
                Some(SpeakerStatus::InsufficientAudio)
            }
            SpeechActivity::Speech if self.voices.is_empty() => Some(SpeakerStatus::Unknown),
            SpeechActivity::Speech => None,
        };
        Ok(status)
    }

    fn classify(
        &self,
        observation: SpeakerObservation,
        candidates: &[SpeakerCandidate],
    ) -> SpeakerStatus {
        let Some(calibration) = &self.calibration else {
            return SpeakerStatus::Uncalibrated;
        };
        if !(calibration.minimum_samples as u64..=calibration.maximum_samples as u64)
            .contains(&observation.samples())
        {
            return SpeakerStatus::Uncalibrated;
        }
        let Some(best) = candidates.first() else {
            return SpeakerStatus::Unknown;
        };
        if best.nearest < calibration.tentative + calibration.numerical_guard {
            return SpeakerStatus::Unknown;
        }
        if candidates.get(1).is_some_and(|other| {
            best.nearest - other.nearest <= calibration.minimum_margin + calibration.numerical_guard
        }) {
            return SpeakerStatus::Ambiguous;
        }
        if best.nearest < calibration.confident + calibration.numerical_guard {
            return SpeakerStatus::Tentative;
        }
        SpeakerStatus::Confident
    }

    fn validate_calibration(&self, calibration: &SpeakerCalibration) -> SpeakerResult<()> {
        if calibration.id.trim().is_empty()
            || calibration.id.len() > 128
            || calibration.reference_revision != self.revision
        {
            return Err(SpeakerError::InvalidPolicy(
                "calibration must name this exact bank and a nonempty revision",
            ));
        }
        if calibration.minimum_samples < self.space.minimum_samples
            || calibration.minimum_samples > calibration.maximum_samples
            || calibration.maximum_samples > self.space.maximum_samples
        {
            return Err(SpeakerError::InvalidPolicy(
                "calibration duration band is outside the model contract",
            ));
        }
        if !(-1.0..=1.0).contains(&calibration.confident)
            || !(-1.0..=1.0).contains(&calibration.tentative)
            || calibration.tentative > calibration.confident
            || !(0.0..=2.0).contains(&calibration.minimum_margin)
            || !(0.0..=1.0).contains(&calibration.numerical_guard)
            || calibration.confident + calibration.numerical_guard > 1.0
            || calibration.minimum_margin + calibration.numerical_guard > 2.0
        {
            return Err(SpeakerError::InvalidPolicy(
                "nonfinite or inconsistent cosine thresholds",
            ));
        }
        Ok(())
    }
}

pub(super) fn normalized(values: &[f32], dimensions: usize) -> SpeakerResult<Vec<f64>> {
    if values.len() != dimensions || values.iter().any(|value| !value.is_finite()) {
        return Err(SpeakerError::InvalidProfile(
            "nonfinite or wrong-dimensional vector",
        ));
    }
    let norm = values
        .iter()
        .map(|value| f64::from(*value).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return Err(SpeakerError::InvalidProfile("zero vector"));
    }
    Ok(values
        .iter()
        .map(|value| f64::from(*value) / norm)
        .collect())
}

fn reference_centroid(references: &[Vec<f64>], dimensions: usize) -> Option<Vec<f64>> {
    let mut centroid = vec![0.0_f64; dimensions];
    for reference in references {
        for (sum, value) in centroid.iter_mut().zip(reference) {
            *sum += value;
        }
    }
    let norm = centroid
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if norm == 0.0 {
        return None;
    }
    for value in &mut centroid {
        *value /= norm;
    }
    Some(centroid)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| a * b)
        .sum::<f64>()
        .clamp(-1.0, 1.0)
}
