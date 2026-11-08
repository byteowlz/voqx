use super::{
    EmbeddingSpace, SpeakerEmbedding, SpeakerError, SpeakerEvidence, SpeakerMatcher,
    SpeakerObservation, SpeakerResult, SpeakerStatus, SpeakerTracker,
};

/// In-process encoder seam. It neither captures PCM nor reads/writes a voice registry.
pub trait WindowEncoder: std::fmt::Debug {
    /// Exact descriptor for independently reset, bounded observations.
    fn space(&self) -> &EmbeddingSpace;

    /// Encode exactly the supplied mono PCM, with the descriptor's reset/pooling rules.
    ///
    /// # Errors
    /// Rejects invalid/unsupported audio and propagates backend failures.
    fn encode(&mut self, samples: &[f32]) -> SpeakerResult<SpeakerEmbedding>;
}

/// Shared native orchestration: one encoder call per observation, then all identities.
///
/// Observation windows are bounded and independently reset. Repeated sliding-window
/// inference has a real CPU cost; this is not the qualified continuous-state backend.
/// No microphone, VAD, ASR policy, automatic enrollment or fallback is introduced.
#[derive(Debug)]
pub struct SpeakerAttributor<E> {
    encoder: E,
    matcher: SpeakerMatcher,
    tracker: SpeakerTracker,
}

impl<E: WindowEncoder> SpeakerAttributor<E> {
    /// Compose an encoder, read-only bank and conservative tracker without I/O.
    ///
    /// # Errors
    /// Rejects incompatible encoder/profile spaces and invalid tracker configuration.
    pub fn new(
        encoder: E,
        matcher: SpeakerMatcher,
        generation: u64,
        minimum_stable_samples: u64,
    ) -> SpeakerResult<Self> {
        if encoder.space() != matcher.space() {
            return Err(SpeakerError::InvalidProfile(
                "encoder does not match reference snapshot",
            ));
        }
        let tracker = SpeakerTracker::new(&matcher, generation, minimum_stable_samples)?;
        Ok(Self {
            encoder,
            matcher,
            tracker,
        })
    }

    /// Fingerprint to bind calibration or inspection to this immutable bank.
    #[must_use]
    pub fn reference_revision(&self) -> &str {
        self.matcher.reference_revision()
    }

    /// Cancel old observation results by advancing the capture generation.
    ///
    /// # Errors
    /// Rejects a non-increasing or invalid generation.
    pub fn reset(&mut self, generation: u64) -> SpeakerResult<()> {
        self.tracker.reset(generation)
    }

    /// Process exactly one host-provided PCM interval. Skips inference for activity,
    /// short/empty-bank abstentions and digital silence. Errors clear continuity.
    ///
    /// # Errors
    /// Rejects invalid samples, mismatched clocks, stale/out-of-order work and backend
    /// failures. The host must surface failure, never silently bypass recognition.
    pub fn process(
        &mut self,
        observation: SpeakerObservation,
        samples: &[f32],
    ) -> SpeakerResult<SpeakerEvidence> {
        self.tracker.validate_observation(observation)?;
        let result = self.process_current(observation, samples);
        if result.is_err() {
            self.tracker.invalidate();
        }
        result
    }

    fn process_current(
        &mut self,
        observation: SpeakerObservation,
        samples: &[f32],
    ) -> SpeakerResult<SpeakerEvidence> {
        if observation.samples() != samples.len() as u64
            || samples.iter().any(|value| !value.is_finite())
        {
            return Err(SpeakerError::InvalidInput(
                "PCM and observation must have exact finite coverage",
            ));
        }
        let evidence = if !self.matcher.needs_embedding(observation)? {
            self.matcher.evaluate(observation, None)?
        } else if samples.iter().all(|sample| *sample == 0.0) {
            self.matcher
                .evidence(observation, SpeakerStatus::InsufficientAudio)
        } else {
            let embedding = self.encoder.encode(samples)?;
            self.matcher.evaluate(observation, Some(&embedding))?
        };
        self.tracker.observe(evidence)
    }
}
