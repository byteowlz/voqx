//! The Voice Identity model: identities and their engine-scoped profiles.
//!
//! See `docs/adr/0001-voice-identity-model.md`. Identity is by stable
//! [`VoiceIdentity::id`], never by display name. All heavy assets (audio, model
//! artifacts) are referenced by **registry-relative** paths so the registry
//! directory is portable.

use serde::{Deserialize, Serialize};

use crate::math::centroid;
use crate::time::now_unix;

/// A durable, engine-agnostic voice entity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceIdentity {
    /// Stable internal id (`vox-<slug>`). Never derived from the display name at
    /// read time.
    pub id: String,
    /// Human-facing display name.
    pub display_name: String,
    /// Alternate names previously used for this identity.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Free-form operator note.
    #[serde(default)]
    pub note: String,
    /// Reusable, engine-agnostic source recordings.
    #[serde(default)]
    pub sources: Vec<SourceRecording>,
    /// "Who spoke" profiles (e.g. WeSpeaker embeddings).
    #[serde(default)]
    pub recognition_profiles: Vec<RecognitionProfile>,
    /// "Speak as them" profiles (e.g. spqx references).
    #[serde(default)]
    pub synthesis_profiles: Vec<SynthesisProfile>,
    /// "They said the wake word" profiles.
    #[serde(default)]
    pub wake_profiles: Vec<WakeProfile>,
    /// Creation time (unix seconds).
    pub created_at_unix: u64,
    /// Last-updated time (unix seconds).
    pub updated_at_unix: u64,
}

impl VoiceIdentity {
    /// Create a new identity with the given id and display name and no profiles.
    #[must_use]
    pub fn new(id: impl Into<String>, display_name: impl Into<String>) -> Self {
        let now = now_unix();
        Self {
            id: id.into(),
            display_name: display_name.into(),
            aliases: Vec::new(),
            note: String::new(),
            sources: Vec::new(),
            recognition_profiles: Vec::new(),
            synthesis_profiles: Vec::new(),
            wake_profiles: Vec::new(),
            created_at_unix: now,
            updated_at_unix: now,
        }
    }

    /// Whether `name` matches this identity's display name or any alias.
    #[must_use]
    pub fn matches_name(&self, name: &str) -> bool {
        self.display_name == name || self.aliases.iter().any(|a| a == name)
    }

    /// The capabilities this identity currently has.
    #[must_use]
    pub const fn capabilities(&self) -> Capabilities {
        Capabilities {
            recognition: !self.recognition_profiles.is_empty(),
            synthesis: !self.synthesis_profiles.is_empty(),
            wake: !self.wake_profiles.is_empty(),
        }
    }

    /// The recognition profile for `engine`, if present.
    #[must_use]
    pub fn recognition_profile(&self, engine: &str) -> Option<&RecognitionProfile> {
        self.recognition_profiles
            .iter()
            .find(|p| p.engine == engine)
    }

    /// The synthesis profile for `engine`, if present.
    #[must_use]
    pub fn synthesis_profile(&self, engine: &str) -> Option<&SynthesisProfile> {
        self.synthesis_profiles.iter().find(|p| p.engine == engine)
    }

    /// Mark the identity as updated now. Call after any mutation.
    pub fn touch(&mut self) {
        self.updated_at_unix = now_unix();
    }
}

/// Which capabilities a [`VoiceIdentity`] currently has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Capabilities {
    /// Has at least one recognition profile.
    pub recognition: bool,
    /// Has at least one synthesis profile.
    pub synthesis: bool,
    /// Has at least one wake profile.
    pub wake: bool,
}

/// An engine-agnostic reusable source recording kept on the identity so
/// engine-specific derivatives can be regenerated later.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceRecording {
    /// Profile-local id.
    pub id: String,
    /// Registry-relative path to the audio file.
    pub audio: String,
    /// Registry-relative path to a transcript file, if any.
    #[serde(default)]
    pub transcript: Option<String>,
    /// BCP-47-ish language tag of the recording (e.g. `de-DE`), if known.
    #[serde(default)]
    pub language: Option<String>,
    /// Free-form operator note.
    #[serde(default)]
    pub note: String,
    /// Creation time (unix seconds).
    pub created_at_unix: u64,
}

/// A "who spoke" profile: per-sample embeddings and their centroid, scoped to
/// one embedding engine/model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognitionProfile {
    /// Embedding engine identifier (e.g. `speakrs-wespeaker`).
    pub engine: String,
    /// Model/version identifier, so the profile can be invalidated on change.
    #[serde(default)]
    pub model_version: String,
    /// L2-normalized centroid of [`Self::embeddings`].
    #[serde(default)]
    pub centroid: Vec<f32>,
    /// Per-sample embedding vectors.
    #[serde(default)]
    pub embeddings: Vec<Vec<f32>>,
    /// Registry-relative sample audio paths, aligned with [`Self::embeddings`].
    #[serde(default)]
    pub samples: Vec<String>,
    /// Free-form namespaced tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Creation time (unix seconds).
    pub created_at_unix: u64,
    /// Last-updated time (unix seconds).
    pub updated_at_unix: u64,
}

impl RecognitionProfile {
    /// Create a recognition profile from embeddings and aligned sample paths,
    /// computing the centroid.
    #[must_use]
    pub fn new(
        engine: impl Into<String>,
        model_version: impl Into<String>,
        embeddings: Vec<Vec<f32>>,
        samples: Vec<String>,
    ) -> Self {
        let now = now_unix();
        Self {
            engine: engine.into(),
            model_version: model_version.into(),
            centroid: centroid(&embeddings),
            embeddings,
            samples,
            tags: Vec::new(),
            created_at_unix: now,
            updated_at_unix: now,
        }
    }

    /// Append embeddings and sample paths and recompute the centroid.
    pub fn add_samples(&mut self, mut embeddings: Vec<Vec<f32>>, mut samples: Vec<String>) {
        self.embeddings.append(&mut embeddings);
        self.samples.append(&mut samples);
        self.centroid = centroid(&self.embeddings);
        self.updated_at_unix = now_unix();
    }

    /// Drop the sample at `index` and recompute the centroid.
    ///
    /// Returns the removed sample's registry-relative audio path, if it had one.
    /// Pruning a contaminated sample changes what this profile matches, which is
    /// the point: the centroid is re-derived from what remains.
    pub fn remove_sample(&mut self, index: usize) -> Option<String> {
        if index >= self.embeddings.len() {
            return None;
        }
        self.embeddings.remove(index);
        let audio = (index < self.samples.len()).then(|| self.samples.remove(index));
        self.centroid = centroid(&self.embeddings);
        self.updated_at_unix = now_unix();
        audio
    }
}

/// A "speak as them" profile, scoped to one TTS engine, holding one or more
/// reference performances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisProfile {
    /// TTS engine identifier (e.g. `spqx-qwen3`).
    pub engine: String,
    /// Reference performances. The registry never selects one.
    #[serde(default)]
    pub references: Vec<SynthesisReference>,
    /// Creation time (unix seconds).
    pub created_at_unix: u64,
    /// Last-updated time (unix seconds).
    pub updated_at_unix: u64,
}

impl SynthesisProfile {
    /// Create an empty synthesis profile for `engine`.
    #[must_use]
    pub fn new(engine: impl Into<String>) -> Self {
        let now = now_unix();
        Self {
            engine: engine.into(),
            references: Vec::new(),
            created_at_unix: now,
            updated_at_unix: now,
        }
    }

    /// The reference marked `default`, else the first reference, if any.
    #[must_use]
    pub fn default_reference(&self) -> Option<&SynthesisReference> {
        self.references
            .iter()
            .find(|r| r.default)
            .or_else(|| self.references.first())
    }
}

/// One reference recording ("performance") within a [`SynthesisProfile`].
///
/// The registry stores every reference and exposes them all. Selection — by id,
/// by `reference_language` + `tags`, by a model interpreting a natural-language
/// style request, or by [`SynthesisProfile::default_reference`] — is the
/// caller's job. Tags are free-form, preferably namespaced `dimension:value`
/// (e.g. `emotion:happy`, `register:casual`); the registry defines no closed
/// emotion taxonomy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisReference {
    /// Profile-local id.
    pub id: String,
    /// Registry-relative path to the reference audio.
    pub audio: String,
    /// Registry-relative path to the exact reference transcript.
    pub transcript: String,
    /// BCP-47-ish language spoken in the reference (e.g. `de-DE`), if known.
    /// Independent of the synthesis target language.
    #[serde(default)]
    pub reference_language: Option<String>,
    /// Whether this is the profile's default reference.
    #[serde(default)]
    pub default: bool,
    /// Free-form namespaced tags describing style/use.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Human-authored description of the delivery.
    #[serde(default)]
    pub description: String,
    /// Reference duration in seconds, if known.
    #[serde(default)]
    pub duration_seconds: f64,
    /// Audio sample rate in Hz, if known.
    #[serde(default)]
    pub sample_rate: u32,
}

/// A "they said the wake word" profile. The wake tool owns the model artifact;
/// the registry only records that the profile exists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WakeProfile {
    /// Profile-local id.
    pub id: String,
    /// The wake/command phrase.
    pub phrase: String,
    /// Detection backend identifier (e.g. `rustpotter`).
    pub backend: String,
    /// Registry-relative path to the trained/reference artifact, if any.
    #[serde(default)]
    pub artifact: Option<String>,
    /// Free-form namespaced tags.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Creation time (unix seconds).
    pub created_at_unix: u64,
    /// Last-updated time (unix seconds).
    pub updated_at_unix: u64,
}

#[cfg(test)]
mod tests {
    use super::{RecognitionProfile, SynthesisProfile, SynthesisReference, VoiceIdentity};

    #[test]
    fn capabilities_reflect_profiles() {
        let mut id = VoiceIdentity::new("vox-tommy", "Tommy");
        assert!(!id.capabilities().recognition);
        id.recognition_profiles.push(RecognitionProfile::new(
            "speakrs-wespeaker",
            "v1",
            vec![vec![1.0, 0.0]],
            vec![],
        ));
        let caps = id.capabilities();
        assert!(caps.recognition);
        assert!(!caps.synthesis);
    }

    #[test]
    fn default_reference_prefers_flagged() {
        let mut profile = SynthesisProfile::new("spqx-qwen3");
        profile.references.push(SynthesisReference {
            id: "a".into(),
            audio: "a.wav".into(),
            transcript: "a.txt".into(),
            reference_language: None,
            default: false,
            tags: vec![],
            description: String::new(),
            duration_seconds: 0.0,
            sample_rate: 0,
        });
        profile.references.push(SynthesisReference {
            id: "b".into(),
            audio: "b.wav".into(),
            transcript: "b.txt".into(),
            reference_language: None,
            default: true,
            tags: vec![],
            description: String::new(),
            duration_seconds: 0.0,
            sample_rate: 0,
        });
        assert_eq!(profile.default_reference().map(|r| r.id.as_str()), Some("b"));
    }
}
