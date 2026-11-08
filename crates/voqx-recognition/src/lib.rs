//! Embeddable speaker recognition: optional native encoders and portable attribution.
//!
//! Default builds preserve the existing WeSpeaker interface. For FASTDIAR alone use
//! `default-features = false, features = ["fastdiar", "registry"]`. With no features,
//! only pure-Rust scoring/tracking is built; `wasm` adds `JavaScript` bindings.
//!
//! No microphone, subprocess, network request, registry write or action is owned by
//! this crate. Speaker evidence is not authentication or ASR-admission policy.

/// Sample rate of the supported mono recognition waveforms.
pub const SAMPLE_RATE: u32 = 16_000;

pub mod speaker;

#[cfg(all(test, feature = "wespeaker"))]
mod test_support;

#[cfg(feature = "fastdiar")]
pub mod fastdiar;
#[cfg(feature = "registry")]
pub mod speaker_registry;
#[cfg(all(feature = "wasm", target_arch = "wasm32"))]
mod web;

#[cfg(feature = "wespeaker")]
mod pooling;
#[cfg(feature = "wespeaker")]
mod wespeaker;
#[cfg(feature = "wespeaker")]
pub use wespeaker::{
    EMBEDDING_MODEL_ID, Embedder, MIN_IDENTIFY_SAMPLES, RecognitionPreprocessing, WINDOW_SAMPLES,
    read_wav_samples,
};
