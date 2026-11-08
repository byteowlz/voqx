use super::{
    SpeakerError, SpeakerEvidence, SpeakerMatcher, SpeakerObservation, SpeakerResult, SpeakerStatus,
};

/// Causal, conservative stabilization over advancing observation endpoints.
///
/// Repeated/overlapping windows do not each contribute a full window of dwell.
/// A competing speaker immediately removes the old label; uncertainty, missing
/// coverage and activity abstentions clear continuity. This is not diarization,
/// overlap detection or proof that every sample in a window has one speaker.
#[derive(Debug)]
pub struct SpeakerTracker {
    generation: u64,
    space: super::EmbeddingSpace,
    reference_revision: String,
    calibration_id: Option<String>,
    calibration_revision: Option<String>,
    minimum_stable_samples: u64,
    previous: Option<SpeakerObservation>,
    pending: Option<(String, u64)>,
    stable: Option<String>,
}

impl SpeakerTracker {
    /// Bind a tracker to an immutable bank/policy and explicit capture generation.
    ///
    /// # Errors
    /// Rejects a zero dwell interval or invalid generation.
    pub fn new(
        matcher: &SpeakerMatcher,
        generation: u64,
        minimum_stable_samples: u64,
    ) -> SpeakerResult<Self> {
        SpeakerObservation {
            generation,
            start_sample: 0,
            end_sample: 0,
            activity: super::SpeechActivity::Unknown,
        }
        .validate()?;
        if minimum_stable_samples == 0 {
            return Err(SpeakerError::InvalidPolicy(
                "stabilization must require advancing samples",
            ));
        }
        Ok(Self {
            generation,
            space: matcher.space().clone(),
            reference_revision: matcher.reference_revision().to_owned(),
            calibration_id: matcher.calibration_id().map(str::to_owned),
            calibration_revision: matcher.calibration_revision().map(str::to_owned),
            minimum_stable_samples,
            previous: None,
            pending: None,
            stable: None,
        })
    }

    /// Fence unfinished work and start a newer capture; never accept an old generation.
    ///
    /// # Errors
    /// Rejects non-increasing or nonportable generation identifiers.
    pub fn reset(&mut self, generation: u64) -> SpeakerResult<()> {
        if generation <= self.generation {
            return Err(SpeakerError::StaleObservation);
        }
        SpeakerObservation {
            generation,
            start_sample: 0,
            end_sample: 0,
            activity: super::SpeechActivity::Unknown,
        }
        .validate()?;
        self.generation = generation;
        self.previous = None;
        self.invalidate();
        Ok(())
    }

    /// Check freshness before a caller performs expensive inference.
    ///
    /// # Errors
    /// Rejects inactive generations and non-advancing/reordered windows.
    pub fn validate_observation(&self, observation: SpeakerObservation) -> SpeakerResult<()> {
        observation.validate()?;
        if observation.generation != self.generation {
            return Err(SpeakerError::StaleObservation);
        }
        if self.previous.is_some_and(|previous| {
            observation.start_sample < previous.start_sample
                || observation.end_sample <= previous.end_sample
        }) {
            return Err(SpeakerError::OutOfOrder);
        }
        Ok(())
    }

    /// Clear speaker continuity after an encoder/input failure without changing the clock.
    pub fn invalidate(&mut self) {
        self.pending = None;
        self.stable = None;
    }

    /// Stabilize matched evidence, preserving source timestamps and diagnostic scores.
    ///
    /// # Errors
    /// Rejects stale bank/policy/capture revisions, reordered intervals or malformed
    /// confident evidence. Rejected observations never replace current state.
    pub fn observe(&mut self, mut evidence: SpeakerEvidence) -> SpeakerResult<SpeakerEvidence> {
        self.validate_observation(evidence.observation)?;
        if evidence.format != super::EVIDENCE_FORMAT
            || evidence.space != self.space
            || evidence.reference_revision != self.reference_revision
            || evidence.calibration_id != self.calibration_id
            || evidence.calibration_revision != self.calibration_revision
        {
            return Err(SpeakerError::StaleObservation);
        }
        if evidence.status == SpeakerStatus::Confident
            && (evidence.voice_id.is_none()
                || self.calibration_id.is_none()
                || evidence.voice_id.as_ref()
                    != evidence
                        .candidates
                        .first()
                        .map(|candidate| &candidate.voice_id))
        {
            return Err(SpeakerError::InvalidInput(
                "confident evidence needs calibration and the winning canonical voice_id",
            ));
        }
        if self
            .previous
            .is_some_and(|previous| evidence.observation.start_sample > previous.end_sample)
        {
            self.invalidate();
        }
        self.previous = Some(evidence.observation);
        if evidence.status != SpeakerStatus::Confident {
            evidence.voice_id = None;
            self.invalidate();
            return Ok(evidence);
        }
        self.stabilize(&mut evidence);
        Ok(evidence)
    }

    fn stabilize(&mut self, evidence: &mut SpeakerEvidence) {
        let Some(voice_id) = &evidence.voice_id else {
            return;
        };
        if self.stable.as_ref() == Some(voice_id) {
            return;
        }
        self.stable = None;
        match &self.pending {
            Some((pending, since)) if pending == voice_id => {
                if evidence.observation.end_sample.saturating_sub(*since)
                    >= self.minimum_stable_samples
                {
                    self.stable = Some(voice_id.clone());
                    self.pending = None;
                    return;
                }
            }
            _ => self.pending = Some((voice_id.clone(), evidence.observation.end_sample)),
        }
        evidence.status = SpeakerStatus::Tentative;
        evidence.voice_id = None;
    }
}
