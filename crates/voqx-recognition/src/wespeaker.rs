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
use voqx_core::recognition::{AudioObservation, Recognition, Thresholds};
use voqx_core::registry::VoiceRegistry;

use crate::{SAMPLE_RATE, pooling};
pub use voqx_core::RecognitionPreprocessing;

/// Engine identifier stored on recognition profiles produced by this crate.
pub const EMBEDDING_MODEL_ID: &str = "speakrs-wespeaker";

/// Single-window input size of the pinned speakrs 0.4.2 embedding implementation.
/// Audio beyond this leading 10-second window is not observed.
pub const WINDOW_SAMPLES: usize = 160_000;

/// One-second operational floor for identification, in mono 16 kHz samples.
///
/// This rejects very short utterances rather than interpreting padding-dominated
/// embeddings as negative identity evidence. It is not an accuracy calibration;
/// the actual minimum also respects the loaded model's minimum.
pub const MIN_IDENTIFY_SAMPLES: usize = SAMPLE_RATE as usize;

/// Produces `WeSpeaker` speaker embeddings from mono 16 kHz audio.
pub struct Embedder {
    model: EmbeddingModel,
    preprocessing: RecognitionPreprocessing,
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
        Ok(Self {
            model,
            preprocessing: RecognitionPreprocessing::Uniform,
        })
    }

    /// Select the pooling contract. Does not modify any stored recognition profile.
    ///
    /// Valid-frame masking requires separately enrolled profiles under its own
    /// engine key. The default constructor keeps legacy uniform pooling.
    #[must_use]
    pub const fn with_preprocessing(mut self, preprocessing: RecognitionPreprocessing) -> Self {
        self.preprocessing = preprocessing;
        self
    }

    /// Engine/profile key for the selected preprocessing; never mix variants.
    #[must_use]
    pub const fn engine_id(&self) -> &'static str {
        self.preprocessing.engine_id()
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
    /// Returns an error for invalid/silent audio, insufficient input, inference
    /// failure or invalid model output. Valid-v1 requires at least one second;
    /// that is an operational floor, not an accuracy guarantee.
    pub fn embed(&mut self, samples: &[f32]) -> Result<Vec<f32>> {
        let audio = pooling::input_window(samples, self.min_samples(), self.preprocessing)?;
        let vector = match self.preprocessing {
            RecognitionPreprocessing::Uniform => self.model.embed(audio),
            RecognitionPreprocessing::ValidV1 => {
                self.model
                    .embed_masked(audio, &pooling::valid_mask(audio.len()), None)
            }
        }
        .map_err(|err| anyhow!("embedding failed: {err}"))?
        .to_vec();
        pooling::validate_embedding(&vector)?;
        Ok(vector)
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
    /// Audio shorter than the larger of [`Self::min_samples`] and
    /// [`MIN_IDENTIFY_SAMPLES`] yields `InsufficientAudio` without inference.
    /// Scope metadata exposes leading-window truncation and zero padding.
    /// Durations include silence: no speech detection or calibrated band exists.
    /// Exact digital silence yields `InsufficientAudio`; nonfinite input errors.
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
        identify_audio(
            registry,
            samples,
            self.min_samples(),
            self.engine_id(),
            thresholds,
            |audio| self.embed(audio),
        )
    }
}

/// Shared orchestration seam: tests inject embeddings without loading a model.
fn identify_audio(
    registry: &VoiceRegistry,
    samples: &[f32],
    model_minimum: usize,
    engine: &str,
    thresholds: Thresholds,
    embed: impl FnOnce(&[f32]) -> Result<Vec<f32>>,
) -> Result<Recognition> {
    let minimum = model_minimum.max(MIN_IDENTIFY_SAMPLES);
    let observed = samples.len().min(WINDOW_SAMPLES);
    if !samples.iter().all(|sample| sample.is_finite()) {
        return Err(anyhow!("recognition audio contains nonfinite samples"));
    }
    let has_signal = samples[..observed].iter().any(|sample| *sample != 0.0);
    let mut evidence = if observed < minimum || !has_signal {
        Recognition::insufficient_audio(engine)
    } else {
        let embedding = embed(&samples[..observed])?;
        registry.identify(&embedding, engine, thresholds)
    };
    // Insufficient audio is not submitted to inference, so nothing was observed
    // and no padding/truncation operation actually took place.
    let inferred = evidence.status != voqx_core::RecognitionStatus::InsufficientAudio;
    evidence.audio = Some(AudioObservation {
        sample_rate: SAMPLE_RATE,
        input_duration_seconds: samples.len() as f64 / f64::from(SAMPLE_RATE),
        observed_duration_seconds: if inferred {
            observed as f64 / f64::from(SAMPLE_RATE)
        } else {
            0.0
        },
        minimum_duration_seconds: minimum as f64 / f64::from(SAMPLE_RATE),
        padded: inferred && observed < WINDOW_SAMPLES,
        truncated: inferred && samples.len() > WINDOW_SAMPLES,
        calibration_band: None,
    });
    Ok(evidence)
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
    use super::{
        EMBEDDING_MODEL_ID, MIN_IDENTIFY_SAMPLES, SAMPLE_RATE, WINDOW_SAMPLES,
        identify_audio as identify_with_engine, parse_mode,
    };
    use crate::test_support::check_eq;
    use anyhow::Result;
    use speakrs::ExecutionMode;
    use voqx_core::{
        AudioObservation, Recognition, RecognitionProfile, RecognitionStatus, Thresholds,
        VoiceIdentity, VoiceRegistry,
    };

    fn identify_audio(
        registry: &VoiceRegistry,
        samples: &[f32],
        model_minimum: usize,
        thresholds: Thresholds,
        embed: impl FnOnce(&[f32]) -> Result<Vec<f32>>,
    ) -> Result<Recognition> {
        identify_with_engine(
            registry,
            samples,
            model_minimum,
            EMBEDDING_MODEL_ID,
            thresholds,
            embed,
        )
    }

    #[test]
    fn masked_identification_keeps_its_engine_on_abstention_and_incompatibility() -> Result<()> {
        let engine = super::RecognitionPreprocessing::ValidV1.engine_id();
        let short = identify_with_engine(
            &registry(),
            &[0.1; 800],
            400,
            engine,
            Thresholds::default(),
            |_| Err(anyhow::anyhow!("must not infer")),
        )?;
        check_eq(&(short.engine), &(engine))?;
        check_eq(&(short.status), &(RecognitionStatus::InsufficientAudio))?;
        let mismatched = identify_with_engine(
            &registry(),
            &vec![0.1; 16_000],
            400,
            engine,
            Thresholds::default(),
            |_| Ok(vec![1.0, 0.0]),
        )?;
        check_eq(&(mismatched.engine), &(engine))?;
        check_eq(
            &(mismatched.status),
            &(RecognitionStatus::IncompatibleModel),
        )?;
        check_eq(&(mismatched.score), &(None))?;
        Ok(())
    }

    fn registry() -> VoiceRegistry {
        let mut identity = VoiceIdentity::new("vox-fixture", "Fixture");
        identity.recognition_profiles.push(RecognitionProfile::new(
            EMBEDDING_MODEL_ID,
            "test-only",
            vec![vec![1.0, 0.0]],
            vec![],
        ));
        VoiceRegistry {
            identities: vec![identity],
            ..VoiceRegistry::default()
        }
    }

    #[test]
    fn duration_boundaries_report_actual_scope_without_calibration() -> Result<()> {
        // Generated PCM tests orchestration, not recognition accuracy. No model,
        // voice recording, download, or derived evaluation corpus is involved.
        for seconds in [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 10.0, 12.0] {
            let count = (seconds * f64::from(SAMPLE_RATE)) as usize;
            let mut samples = vec![0.25; count];
            if count > WINDOW_SAMPLES {
                samples[WINDOW_SAMPLES..].fill(-0.5);
            }
            let mut called = false;
            let evidence =
                identify_audio(&registry(), &samples, 400, Thresholds::default(), |audio| {
                    called = true;
                    check_eq(&(audio), &(&samples[..count.min(WINDOW_SAMPLES)]))?;
                    Ok(vec![1.0, 0.0])
                })?;
            let inferred = count >= MIN_IDENTIFY_SAMPLES;
            check_eq(&(called), &(inferred))?;
            let mut expected = if inferred {
                Recognition {
                    voice_id: Some("vox-fixture".into()),
                    display_name: Some("Fixture".into()),
                    engine: EMBEDDING_MODEL_ID.into(),
                    score: Some(1.0),
                    score_sample: Some(1.0),
                    score_centroid: Some(1.0),
                    audio: None,
                    status: RecognitionStatus::Confident,
                }
            } else {
                Recognition::insufficient_audio(EMBEDDING_MODEL_ID)
            };
            expected.audio = Some(AudioObservation {
                sample_rate: SAMPLE_RATE,
                input_duration_seconds: seconds,
                observed_duration_seconds: if inferred { seconds.min(10.0) } else { 0.0 },
                minimum_duration_seconds: 1.0,
                padded: inferred && count < WINDOW_SAMPLES,
                truncated: inferred && count > WINDOW_SAMPLES,
                calibration_band: None,
            });
            check_eq(&(evidence), &(expected))?;
        }
        Ok(())
    }

    #[test]
    fn loaded_model_minimum_is_respected_without_inference() -> Result<()> {
        let evidence = identify_audio(
            &registry(),
            &vec![0.25; MIN_IDENTIFY_SAMPLES],
            MIN_IDENTIFY_SAMPLES * 2,
            Thresholds::default(),
            |_| Err(anyhow::anyhow!("must not run inference")),
        )?;
        let mut expected = Recognition::insufficient_audio(EMBEDDING_MODEL_ID);
        expected.audio = Some(AudioObservation {
            sample_rate: SAMPLE_RATE,
            input_duration_seconds: 1.0,
            observed_duration_seconds: 0.0,
            minimum_duration_seconds: 2.0,
            padded: false,
            truncated: false,
            calibration_band: None,
        });
        check_eq(&(evidence), &(expected))?;
        Ok(())
    }

    #[test]
    fn silence_is_insufficient_and_nonfinite_audio_is_rejected() -> Result<()> {
        let evidence = identify_audio(
            &registry(),
            &vec![0.0; WINDOW_SAMPLES],
            400,
            Thresholds::default(),
            |_| Err(anyhow::anyhow!("must not run inference")),
        )?;
        let mut expected = Recognition::insufficient_audio(EMBEDDING_MODEL_ID);
        expected.audio = Some(AudioObservation {
            sample_rate: SAMPLE_RATE,
            input_duration_seconds: 10.0,
            observed_duration_seconds: 0.0,
            minimum_duration_seconds: 1.0,
            padded: false,
            truncated: false,
            calibration_band: None,
        });
        check_eq(&(evidence), &(expected))?;
        for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let result =
                identify_audio(&registry(), &[invalid], 400, Thresholds::default(), |_| {
                    Err(anyhow::anyhow!("must not run inference"))
                });
            check_eq(
                &(result.err().map(|error| error.to_string())),
                &(Some("recognition audio contains nonfinite samples".into())),
            )?;
        }
        Ok(())
    }

    #[test]
    fn inference_failure_is_not_recognition_evidence() {
        let result = identify_audio(
            &registry(),
            &vec![0.25; WINDOW_SAMPLES],
            400,
            Thresholds::default(),
            |_| Err(anyhow::anyhow!("fixture provider error")),
        );
        assert_eq!(
            result.err().map(|error| error.to_string()),
            Some("fixture provider error".into())
        );
    }

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
