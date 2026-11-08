//! voqx-core: the shared **Voice Identity** registry, plus the standard
//! byteowlz workspace substrate (config, XDG paths, schema generation).
//!
//! A [`VoiceIdentity`] is a stable, engine-agnostic entity with one display
//! name, aliases, reusable source recordings, and any number of engine-scoped
//! profiles: [`RecognitionProfile`] ("who spoke"), [`SynthesisProfile`] ("speak
//! as them"), and [`WakeProfile`] ("they said the wake word"). Identity is by
//! stable `voice_id`, never by display name.
//!
//! The registry layer is pure Rust with **no ML dependencies**: it owns the
//! on-disk format, cosine matching, XDG path resolution, and atomic +
//! lock-guarded persistence. Producing embeddings from audio is a separate,
//! optional concern.
//!
//! See `docs/adr/0001-voice-identity-model.md` and `CONTEXT.md`.

// Workspace substrate.
pub mod config;
pub mod error;
pub mod paths;
pub mod schema;

// Voice Identity registry.
pub mod analysis;
pub mod atomic;
pub mod identity;
pub mod ids;
pub mod math;
pub mod preprocessing;
pub mod recognition;
pub mod registry;
pub mod registry_paths;
mod time;

pub use config::{AppConfig, LoggingConfig, PathsConfig, RuntimeConfig};
pub use error::{CoreError, Result};
pub use paths::{AppPaths, default_cache_dir};
pub use schema::{generate_example_config, generate_schema, write_generated_files};

pub use analysis::{
    AnalysisOptions, IdentityAnalysis, SampleAnalysis, SimilarityMatrix, SimilarityRow,
    analyze_identity, analyze_registry, similarity_matrix,
};
pub use identity::{
    Capabilities, RecognitionProfile, SourceRecording, SynthesisProfile, SynthesisReference,
    VoiceIdentity, WakeProfile,
};
pub use math::{centroid, cosine};
pub use preprocessing::RecognitionPreprocessing;
pub use recognition::{AudioObservation, Recognition, RecognitionStatus, Thresholds};
pub use registry::{REGISTRY_VERSION, RegistryStore, VoiceRegistry};
pub use registry_paths::{RegistryPaths, default_registry_root};

/// Application name used for config directories and environment prefix.
pub const APP_NAME: &str = "voqx";

/// Returns the environment variable prefix for this application.
#[must_use]
pub fn env_prefix() -> String {
    APP_NAME
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

/// Returns the default parallelism based on available CPU cores.
#[must_use]
pub fn default_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
}
