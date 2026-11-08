//! Optional in-process, one-thread CPU adapter for the pinned explicit-state graph.
//!
//! The graph is supplied by the application, never fetched or redistributed here.
//! Raw stateful frames and independently reset pooled windows are distinct contracts.
//! There is no VAD, speaker calibration, microphone, subprocess or GPU fallback.

use std::{fs, io::Read, path::Path};

use ort::{
    ep,
    session::{Session, SessionInputValue},
    value::{DynValue, Tensor, TensorElementType, ValueType},
};
use sha2::{Digest, Sha256};

use crate::speaker::{
    EmbeddingSpace, SpeakerEmbedding, SpeakerError, SpeakerResult, WindowEncoder,
};

/// SHA-256 of the locally verified, self-contained explicit-state ONNX graph.
pub const MODEL_SHA256: &str = "e7e3878d00afb742094eb22a8edc3a7f756d8ae1a10ec16200d95e56cf6ec3b1";
/// Exact graph length; model assets are managed by the host application.
pub const MODEL_BYTES: usize = 10_665_298;
/// Fixed mono 16 kHz frame length (80 ms).
pub const STEP_SAMPLES: usize = 1280;
/// FASTDIAR-small vector dimension, incompatible with WeSpeaker's 256.
pub const DIMENSIONS: usize = 192;
/// Independent-window profile slot, not a continuous-stream calibration mode.
pub const ENGINE_ID: &str = "fastdiar-small-cold80-mean-f32-v1";

/// Descriptor including graph hash, reset cadence, FP64 pooling and f32 storage.
///
/// Windows include all final zero-padded frames in the FP64 mean, followed by L2
/// normalization and f32 storage. Their scope excludes padding. No onset trim/VAD
/// is performed. The 1.52-second floor is operational, not a qualified FAR claim.
#[must_use]
pub fn embedding_space() -> EmbeddingSpace {
    EmbeddingSpace {
        engine: ENGINE_ID.to_owned(),
        model_version: format!("cold80-mean-f64-l2-f32-v1:{MODEL_SHA256}"),
        dimensions: DIMENSIONS,
        sample_rate: crate::SAMPLE_RATE,
        minimum_samples: 24_320,
        maximum_samples: 160_000,
    }
}

struct StateSlot {
    input: String,
    output: String,
    dtype: ValueType,
    value: DynValue,
}

/// An actual encoder frame; Debug formatting omits the voiceprint.
#[derive(Clone)]
pub struct EncoderFrame {
    /// Exclusive frame endpoint, including any explicitly submitted padding.
    pub end_sample: u64,
    values: Vec<f32>,
}

impl std::fmt::Debug for EncoderFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderFrame")
            .field("end_sample", &self.end_sample)
            .finish_non_exhaustive()
    }
}

impl EncoderFrame {
    /// Read the 192 model components; raw frames are not pooled recognition profiles.
    #[must_use]
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// Fixed-step, explicit-state CPU encoder with bounded fragmented-input buffering.
///
/// Up to ten seconds can be submitted per push. Partial frames remain buffered;
/// no samples are silently padded or dropped. Reset is explicit. Inference errors
/// fault the encoder until reset, rather than silently continuing a broken stream.
pub struct FastdiarEncoder {
    session: Session,
    states: Vec<StateSlot>,
    pending: Vec<f32>,
    processed_samples: u64,
    faulted: bool,
}

impl std::fmt::Debug for FastdiarEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastdiarEncoder")
            .field("processed_samples", &self.processed_samples)
            .field("pending_samples", &self.pending.len())
            .field("faulted", &self.faulted)
            .finish_non_exhaustive()
    }
}

fn backend(error: impl std::fmt::Display) -> SpeakerError {
    SpeakerError::Backend(error.to_string())
}

impl FastdiarEncoder {
    /// Load and hash a local graph once, with only the CPU execution provider.
    ///
    /// The already verified bytes are loaded from memory to avoid path-replacement
    /// races. No model download or external-weight resolution is performed.
    ///
    /// # Errors
    /// Rejects missing, wrong-sized, wrong-hash or incompatible graphs/runtime.
    pub fn load(path: &Path) -> SpeakerResult<Self> {
        let file = fs::File::open(path).map_err(backend)?;
        if file.metadata().map_err(backend)?.len() != MODEL_BYTES as u64 {
            return Err(backend("unexpected FASTDIAR graph length"));
        }
        let mut bytes = Vec::with_capacity(MODEL_BYTES);
        file.take(MODEL_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(backend)?;
        if bytes.len() != MODEL_BYTES || format!("{:x}", Sha256::digest(&bytes)) != MODEL_SHA256 {
            return Err(backend("FASTDIAR graph SHA-256 mismatch"));
        }
        let session = Session::builder()
            .map_err(backend)?
            .with_no_environment_execution_providers()
            .map_err(backend)?
            .with_execution_providers([ep::CPU::default()
                .with_arena_allocator(true)
                .build()
                .error_on_failure()])
            .map_err(backend)?
            .with_intra_threads(1)
            .map_err(backend)?
            .with_inter_threads(1)
            .map_err(backend)?
            .with_parallel_execution(false)
            .map_err(backend)?
            .commit_from_memory(&bytes)
            .map_err(backend)?;
        let states = model_states(&session)?;
        Ok(Self {
            session,
            states,
            pending: Vec::new(),
            processed_samples: 0,
            faulted: false,
        })
    }

    /// Clear all 50 state slots, counters and partial PCM; keep model weights loaded.
    ///
    /// # Errors
    /// Propagates tensor allocation/runtime failures and leaves the encoder faulted.
    pub fn reset(&mut self) -> SpeakerResult<()> {
        self.faulted = true;
        for state in &mut self.states {
            state.value = zeros(&state.dtype)?;
        }
        self.pending.clear();
        self.processed_samples = 0;
        self.faulted = false;
        Ok(())
    }

    /// Number of actual full frames processed, in samples.
    #[must_use]
    pub const fn processed_samples(&self) -> u64 {
        self.processed_samples
    }

    /// Buffered real input that has not yet completed an 80 ms frame.
    #[must_use]
    pub const fn pending_samples(&self) -> usize {
        self.pending.len()
    }

    /// Consume finite, mono 16 kHz f32 PCM without implicit padding.
    ///
    /// # Errors
    /// Rejects oversized/nonfinite input before state mutation. Backend failures
    /// require an explicit reset and never become partial successful evidence.
    pub fn push(&mut self, samples: &[f32]) -> SpeakerResult<Vec<EncoderFrame>> {
        if self.faulted {
            return Err(backend("encoder is faulted; reset required"));
        }
        if samples.len() > 160_000 || samples.iter().any(|value| !value.is_finite()) {
            return Err(SpeakerError::InvalidInput(
                "expected at most ten seconds of finite mono PCM",
            ));
        }
        let mut data = std::mem::take(&mut self.pending);
        data.extend_from_slice(samples);
        let mut chunks = data.chunks_exact(STEP_SAMPLES);
        let mut frames = Vec::with_capacity(chunks.len());
        for chunk in &mut chunks {
            match self.step(chunk) {
                Ok(frame) => frames.push(frame),
                Err(error) => {
                    self.faulted = true;
                    return Err(error);
                }
            }
        }
        self.pending.extend_from_slice(chunks.remainder());
        Ok(frames)
    }

    fn step(&mut self, samples: &[f32]) -> SpeakerResult<EncoderFrame> {
        let end_sample = self
            .processed_samples
            .checked_add(STEP_SAMPLES as u64)
            .filter(|value| i64::try_from(*value).is_ok())
            .ok_or_else(|| backend("sample clock overflow"))?;
        let audio = Tensor::from_array(([STEP_SAMPLES], samples.to_vec())).map_err(backend)?;
        let mut inputs = vec![("audio", SessionInputValue::from(audio))];
        inputs.extend(
            self.states
                .iter()
                .map(|slot| (slot.input.as_str(), SessionInputValue::from(&slot.value))),
        );
        let mut outputs = self.session.run(inputs).map_err(backend)?;
        let embedding = outputs
            .get("embedding")
            .ok_or_else(|| backend("missing embedding output"))?
            .try_extract_tensor::<f32>()
            .map_err(backend)?
            .1
            .to_vec();
        if embedding.len() != DIMENSIONS || embedding.iter().any(|value| !value.is_finite()) {
            return Err(backend("invalid embedding output"));
        }
        for (name, expected) in [
            ("next_state_4", end_sample),
            ("next_state_6", end_sample / 160),
        ] {
            let value = outputs
                .get(name)
                .ok_or_else(|| backend("missing streaming counter"))?
                .try_extract_tensor::<i64>()
                .map_err(backend)?
                .1;
            if value != [expected as i64] {
                return Err(backend("frozen or invalid streaming counter"));
            }
        }
        for slot in &mut self.states {
            let value = outputs
                .remove(&slot.output)
                .ok_or_else(|| backend("missing state output"))?;
            if value.dtype() != &slot.dtype {
                return Err(backend("state output type changed"));
            }
            if slot.dtype.tensor_type() == Some(TensorElementType::Float32)
                && value
                    .try_extract_tensor::<f32>()
                    .map_err(backend)?
                    .1
                    .iter()
                    .any(|value| !value.is_finite())
            {
                return Err(backend("nonfinite streaming state"));
            }
            slot.value = value;
        }
        self.processed_samples = end_sample;
        Ok(EncoderFrame {
            end_sample,
            values: embedding,
        })
    }
}

fn zeros(dtype: &ValueType) -> SpeakerResult<DynValue> {
    let ValueType::Tensor { ty, shape, .. } = dtype else {
        return Err(backend("non-tensor model state"));
    };
    let count = shape
        .iter()
        .try_fold(1_usize, |count, dimension| {
            usize::try_from(*dimension)
                .ok()
                .and_then(|dimension| count.checked_mul(dimension))
        })
        .filter(|count| *count <= 132_943)
        .ok_or_else(|| backend("invalid model state shape"))?;
    match ty {
        TensorElementType::Float32 => {
            Ok(Tensor::from_array((shape.to_vec(), vec![0.0_f32; count]))
                .map_err(backend)?
                .into_dyn())
        }
        TensorElementType::Int64 => Ok(Tensor::from_array((shape.to_vec(), vec![0_i64; count]))
            .map_err(backend)?
            .into_dyn()),
        _ => Err(backend("unsupported model state type")),
    }
}

fn model_states(session: &Session) -> SpeakerResult<Vec<StateSlot>> {
    if session.inputs().len() != 51
        || session.outputs().len() != 51
        || session
            .inputs()
            .first()
            .is_none_or(|input| input.name() != "audio")
        || session
            .outputs()
            .first()
            .is_none_or(|output| output.name() != "embedding")
    {
        return Err(backend(
            "model tensor names/counts do not match explicit-state graph",
        ));
    }
    let mut states = Vec::with_capacity(50);
    for (index, input) in session.inputs().iter().skip(1).enumerate() {
        let output = session
            .outputs()
            .get(index + 1)
            .ok_or_else(|| backend("missing state metadata"))?;
        if input.name() != format!("state_{index}")
            || output.name() != format!("next_state_{index}")
            || input.dtype() != output.dtype()
        {
            return Err(backend("model state metadata mismatch"));
        }
        states.push(StateSlot {
            input: input.name().to_owned(),
            output: output.name().to_owned(),
            dtype: input.dtype().clone(),
            value: zeros(input.dtype())?,
        });
    }
    Ok(states)
}

/// Independently reset, mean-pooled speaker windows using one loaded CPU encoder.
///
/// This intentionally preserves cold-window semantics rather than applying pilot
/// thresholds to continuous-state embeddings. Live hosts choose an affordable
/// observation cadence and must qualify continuous processing separately.
#[derive(Debug)]
pub struct FastdiarWindowEncoder {
    encoder: FastdiarEncoder,
    space: EmbeddingSpace,
}

impl FastdiarWindowEncoder {
    /// Load the pinned graph without a model service or Python.
    ///
    /// # Errors
    /// Propagates graph/runtime/hash verification failures.
    pub fn load(path: &Path) -> SpeakerResult<Self> {
        Ok(Self {
            encoder: FastdiarEncoder::load(path)?,
            space: embedding_space(),
        })
    }
}

impl WindowEncoder for FastdiarWindowEncoder {
    fn space(&self) -> &EmbeddingSpace {
        &self.space
    }

    fn encode(&mut self, samples: &[f32]) -> SpeakerResult<SpeakerEmbedding> {
        if samples.len() < self.space.minimum_samples
            || samples.len() > self.space.maximum_samples
            || samples.iter().any(|value| !value.is_finite())
            || samples.iter().all(|value| *value == 0.0)
        {
            return Err(SpeakerError::InvalidInput(
                "expected a finite, nonsilent 1.52–10 second observation",
            ));
        }
        self.encoder.reset()?;
        let mut frames = self.encoder.push(samples)?;
        if self.encoder.pending_samples() != 0 {
            let padding = vec![0.0_f32; STEP_SAMPLES - self.encoder.pending_samples()];
            frames.extend(self.encoder.push(&padding)?);
        }
        let mut pooled = vec![0.0_f64; DIMENSIONS];
        for frame in &frames {
            for (sum, value) in pooled.iter_mut().zip(frame.values()) {
                *sum += f64::from(*value);
            }
        }
        for value in &mut pooled {
            *value /= frames.len() as f64;
        }
        let norm = pooled.iter().map(|value| value * value).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(backend("invalid pooled model output"));
        }
        let values = pooled
            .into_iter()
            .map(|value| (value / norm) as f32)
            .collect();
        SpeakerEmbedding::new(self.space.clone(), values)
    }
}
