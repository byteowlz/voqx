//! WeSpeaker embeddings for the voqx Voice Identity registry.
//!
//! This crate is the optional, ML-dependent half of recognition: it turns mono
//! 16 kHz audio into speaker embeddings and matches them against enrolled
//! identities. Consumers that only resolve a `voice_id` should depend on
//! `voqx-core` alone.
//!
//! Embeddings are tagged [`EMBEDDING_MODEL_ID`] so recognition profiles can be
//! invalidated or migrated when the model changes. Results are **evidence**, not
//! authorization — see `voqx_core::recognition`.

use std::path::Path;

use anyhow::{Context, Result, anyhow};
use speakrs::inference::EmbeddingModel;
use speakrs::{ExecutionMode, ModelBundle};
use voqx_core::recognition::{Recognition, Thresholds};
use voqx_core::registry::VoiceRegistry;

/// Engine identifier stored on recognition profiles produced by this crate.
pub const EMBEDDING_MODEL_ID: &str = "speakrs-wespeaker";

/// Sample rate every embedding expects for its input audio.
pub const SAMPLE_RATE: u32 = 16_000;

/// Produces `WeSpeaker` speaker embeddings from mono 16 kHz audio.
pub struct Embedder {
    model: EmbeddingModel,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

impl Embedder {
    /// Load the embedding model, downloading from Hugging Face when `model_dir`
    /// is absent.
    ///
    /// `mode` is one of `cpu`, `coreml`, `coreml-fast`, `cuda`, `cuda-fast`;
    /// unknown values fall back to CPU. Modes without the matching build feature
    /// degrade to CPU inside speakrs.
    ///
    /// # Errors
    ///
    /// Returns an error if the models cannot be resolved or loaded.
    pub fn new(mode: &str, model_dir: Option<&Path>) -> Result<Self> {
        let execution = parse_mode(mode);
        let bundle = match model_dir {
            Some(dir) => ModelBundle::from_dir(dir),
            None => ModelBundle::from_pretrained(execution)
                .map_err(|err| anyhow!("resolving speakrs models: {err}"))?,
        };
        let model = EmbeddingModel::with_mode(bundle.embedding_path(), execution)
            .map_err(|err| anyhow!("loading embedding model: {err}"))?;
        Ok(Self { model })
    }

    /// The minimum number of samples the model can embed.
    #[must_use]
    pub fn min_samples(&self) -> usize {
        self.model.min_num_samples()
    }

    /// Embed mono 16 kHz `samples` into a speaker vector.
    ///
    /// # Errors
    ///
    /// Returns an error if there is too little audio, or inference fails.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let minimum = self.min_samples();
        if samples.len() < minimum {
            return Err(anyhow!(
                "too little audio to embed: {} samples, need at least {minimum}",
                samples.len()
            ));
        }
        let vector = self
            .model
            .embed(samples)
            .map_err(|err| anyhow!("embedding failed: {err}"))?;
        Ok(vector.to_vec())
    }

    /// Read a mono 16 kHz WAV file and embed it.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or inference fails.
    pub fn embed_wav(&mut self, path: &Path) -> Result<Vec<f32>> {
        let samples = read_wav_samples(path)?;
        self.embed(&samples)
    }

    /// Identify `samples` against `registry`, returning recognition evidence.
    ///
    /// Audio shorter than [`Self::min_samples`] yields
    /// [`voqx_core::RecognitionStatus::InsufficientAudio`] rather than an error,
    /// so short command utterances degrade gracefully.
    ///
    /// # Errors
    ///
    /// Returns an error if inference fails.
    pub fn identify(
        &mut self,
        registry: &VoiceRegistry,
        samples: &[f32],
        thresholds: Thresholds,
    ) -> Result<Recognition> {
        if samples.len() < self.min_samples() {
            return Ok(Recognition::insufficient_audio(EMBEDDING_MODEL_ID));
        }
        let embedding = self.embed(samples)?;
        Ok(registry.identify(&embedding, EMBEDDING_MODEL_ID, thresholds))
    }
}

/// Parse an execution-mode string; unknown values fall back to CPU.
fn parse_mode(mode: &str) -> ExecutionMode {
    match mode.to_ascii_lowercase().as_str() {
        "coreml" => ExecutionMode::CoreMl,
        "coreml-fast" => ExecutionMode::CoreMlFast,
        "cuda" => ExecutionMode::Cuda,
        "cuda-fast" => ExecutionMode::CudaFast,
        _ => ExecutionMode::Cpu,
    }
}

/// Read a mono 16 kHz WAV file into `f32` samples.
///
/// Other layouts are read as-is with a warning; callers are responsible for
/// converting to mono 16 kHz beforehand.
///
/// # Errors
///
/// Returns an error if the file cannot be read or decoded.
pub fn read_wav_samples(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening WAV {}", path.display()))?;
    let spec = reader.spec();
    if spec.channels != 1 || spec.sample_rate != SAMPLE_RATE {
        log::warn!(
            "recognition expects mono {SAMPLE_RATE} Hz; got {} channel(s) @ {} Hz",
            spec.channels,
            spec.sample_rate
        );
    }
    match spec.sample_format {
        hound::SampleFormat::Int => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| f32::from(value) / 32_768.0))
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("reading 16-bit PCM WAV samples"),
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("reading float WAV samples"),
    }
}

#[cfg(test)]
mod tests {
    use super::{EMBEDDING_MODEL_ID, parse_mode};
    use speakrs::ExecutionMode;

    #[test]
    fn unknown_mode_falls_back_to_cpu() {
        assert!(matches!(parse_mode("nonsense"), ExecutionMode::Cpu));
        assert!(matches!(parse_mode("CPU"), ExecutionMode::Cpu));
    }

    #[test]
    fn engine_id_is_stable() {
        assert_eq!(EMBEDDING_MODEL_ID, "speakrs-wespeaker");
    }
}
