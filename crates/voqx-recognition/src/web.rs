//! Thin `JavaScript` transport for the same portable matcher/tracker used natively.

use wasm_bindgen::prelude::*;

use crate::speaker::{
    ReferenceSnapshot, SpeakerCalibration, SpeakerEmbedding, SpeakerMatcher, SpeakerObservation,
    SpeakerTracker,
};

/// WASM known-speaker scorer/tracker. Embeddings come from a separately loaded,
/// compatible CPU/WASM encoder; this object neither fetches assets nor captures audio.
#[wasm_bindgen(js_name = SpeakerAttributor)]
#[derive(Debug)]
pub struct WebSpeakerAttributor {
    matcher: SpeakerMatcher,
    tracker: SpeakerTracker,
}

fn js_error(error: impl std::fmt::Display) -> JsError {
    JsError::new(&error.to_string())
}

#[wasm_bindgen(js_class = SpeakerAttributor)]
impl WebSpeakerAttributor {
    /// Construct from an explicitly selected reference snapshot and optional
    /// calibration JSON. Capture/sample counters use `JavaScript` `BigInt` at the FFI.
    ///
    /// # Errors
    /// Rejects oversized/invalid JSON, incompatible profiles and invalid policy.
    #[wasm_bindgen(constructor)]
    pub fn new(
        snapshot_json: &str,
        calibration_json: Option<String>,
        generation: u64,
        stable_samples: u64,
    ) -> Result<Self, JsError> {
        if snapshot_json.len() > 8 * 1024 * 1024
            || calibration_json
                .as_ref()
                .is_some_and(|text| text.len() > 4096)
        {
            return Err(js_error("speaker configuration exceeds transport bound"));
        }
        let snapshot: ReferenceSnapshot = serde_json::from_str(snapshot_json).map_err(js_error)?;
        let mut matcher = SpeakerMatcher::new(snapshot).map_err(js_error)?;
        if let Some(text) = calibration_json {
            matcher = matcher
                .with_calibration(
                    serde_json::from_str::<SpeakerCalibration>(&text).map_err(js_error)?,
                )
                .map_err(js_error)?;
        }
        let tracker =
            SpeakerTracker::new(&matcher, generation, stable_samples).map_err(js_error)?;
        Ok(Self { matcher, tracker })
    }

    /// Exact reference/model fingerprint; contains no reference vectors.
    #[must_use]
    #[wasm_bindgen(js_name = referenceRevision)]
    pub fn reference_revision(&self) -> String {
        self.matcher.reference_revision().to_owned()
    }

    /// Advance capture generation, fencing delayed encoder results.
    ///
    /// # Errors
    /// Rejects non-increasing or invalid generations.
    pub fn reset(&mut self, generation: u64) -> Result<(), JsError> {
        self.tracker.reset(generation).map_err(js_error)
    }

    /// Return timestamped JSON evidence from one already encoded observation.
    /// Observation JSON uses the native `SpeakerObservation` schema and exact-safe
    /// integer clocks. Activity is host-provided, not inferred from the embedding.
    /// Empty vectors are allowed only when no embedding is needed.
    ///
    /// # Errors
    /// Rejects stale clocks, unknown activity, malformed vectors and incompatible scope.
    pub fn observe(&mut self, values: Vec<f32>, observation_json: &str) -> Result<String, JsError> {
        if observation_json.len() > 1024 {
            return Err(js_error("observation exceeds transport bound"));
        }
        let observation: SpeakerObservation =
            serde_json::from_str(observation_json).map_err(js_error)?;
        self.tracker
            .validate_observation(observation)
            .map_err(js_error)?;
        let result = self.observe_current(observation, values);
        if result.is_err() {
            self.tracker.invalidate();
        }
        result.map_err(js_error)
    }

    /// One-shot evidence for a single bounded span (for example one utterance or a
    /// wake-word span), without temporal stabilization or tracker state. Use this
    /// when the host already delimits the span; continuous labels should use `observe`.
    ///
    /// # Errors
    /// Rejects invalid clocks, unknown activity, malformed vectors and oversized spans.
    pub fn evaluate(&self, values: Vec<f32>, observation_json: &str) -> Result<String, JsError> {
        if observation_json.len() > 1024 {
            return Err(js_error("observation exceeds transport bound"));
        }
        let observation: SpeakerObservation =
            serde_json::from_str(observation_json).map_err(js_error)?;
        let embedding = if self
            .matcher
            .needs_embedding(observation)
            .map_err(js_error)?
        {
            Some(SpeakerEmbedding::new(self.matcher.space().clone(), values).map_err(js_error)?)
        } else {
            None
        };
        let evidence = self
            .matcher
            .evaluate(observation, embedding.as_ref())
            .map_err(js_error)?;
        serde_json::to_string(&evidence).map_err(js_error)
    }
}

impl WebSpeakerAttributor {
    fn observe_current(
        &mut self,
        observation: SpeakerObservation,
        values: Vec<f32>,
    ) -> Result<String, crate::speaker::SpeakerError> {
        let embedding = if self.matcher.needs_embedding(observation)? {
            Some(SpeakerEmbedding::new(self.matcher.space().clone(), values)?)
        } else {
            None
        };
        let evidence = self.matcher.evaluate(observation, embedding.as_ref())?;
        let evidence = self.tracker.observe(evidence)?;
        serde_json::to_string(&evidence).map_err(|_| {
            crate::speaker::SpeakerError::InvalidInput("evidence serialization failed")
        })
    }
}
