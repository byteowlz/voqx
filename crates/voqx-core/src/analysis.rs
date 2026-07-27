//! Enrollment-quality analysis for recognition profiles.
//!
//! Enrolling a mislabeled review snippet drags an identity's centroid toward
//! another speaker, which quietly degrades every later match. This module scores
//! each enrolled sample against the registry so contaminated enrollments can be
//! found and pruned.
//!
//! Everything here is pure, deterministic cosine math over stored embeddings —
//! no audio and no model are needed. Thresholds are inputs, not policy: the
//! module reports evidence and never decides what to delete.

use serde::{Deserialize, Serialize};

use crate::identity::VoiceIdentity;
use crate::math::{centroid, cosine};
use crate::registry::VoiceRegistry;

/// Tunables for flagging a sample as suspicious.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnalysisOptions {
    /// Leave-one-out cosine below this marks a sample `weak`.
    pub weak_below: f32,
}

impl Default for AnalysisOptions {
    fn default() -> Self {
        Self { weak_below: 0.60 }
    }
}

/// How one enrolled sample scores against the registry.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SampleAnalysis {
    /// Position within the profile's embeddings.
    pub index: usize,
    /// Registry-relative audio path, when the sample audio was kept.
    pub audio: Option<String>,
    /// Cosine against its own identity's stored centroid (what runtime matching
    /// sees). Inflated for small profiles, since the sample is part of it.
    pub own_score: f32,
    /// Cosine against its own centroid recomputed *without* this sample. `None`
    /// when the profile has only one sample. This is the honest signal.
    pub leave_one_out: Option<f32>,
    /// Best-matching identity across the whole registry.
    pub best_voice_id: String,
    /// Display name of [`Self::best_voice_id`].
    pub best_display_name: String,
    /// Cosine for the best match.
    pub best_score: f32,
    /// The best match is a *different* identity — likely mislabeled.
    pub misattributed: bool,
    /// Leave-one-out similarity is below [`AnalysisOptions::weak_below`].
    pub weak: bool,
}

impl SampleAnalysis {
    /// Whether this sample is worth reviewing.
    #[must_use]
    pub const fn suspicious(&self) -> bool {
        self.misattributed || self.weak
    }
}

/// Per-identity enrollment quality for one recognition engine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IdentityAnalysis {
    /// Stable identity id.
    pub voice_id: String,
    /// Display name.
    pub display_name: String,
    /// Embedding engine analysed.
    pub engine: String,
    /// Mean leave-one-out score across samples that have one.
    pub mean_leave_one_out: Option<f32>,
    /// Per-sample detail.
    pub samples: Vec<SampleAnalysis>,
}

impl IdentityAnalysis {
    /// Number of samples worth reviewing.
    #[must_use]
    pub fn suspicious_count(&self) -> usize {
        self.samples.iter().filter(|s| s.suspicious()).count()
    }
}

/// One row of the identity-to-identity centroid similarity matrix.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimilarityRow {
    /// Identity this row describes.
    pub voice_id: String,
    /// Display name.
    pub display_name: String,
    /// Cosine against every identity in [`SimilarityMatrix::voice_ids`] order.
    pub scores: Vec<f32>,
}

/// Centroid similarity across every identity holding the engine's profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SimilarityMatrix {
    /// Column order.
    pub voice_ids: Vec<String>,
    /// Rows, in the same order as [`Self::voice_ids`].
    pub rows: Vec<SimilarityRow>,
}

/// Analyse one identity's enrolled samples for `engine`.
///
/// Returns `None` when the identity has no profile for that engine.
#[must_use]
pub fn analyze_identity(
    registry: &VoiceRegistry,
    identity: &VoiceIdentity,
    engine: &str,
    options: AnalysisOptions,
) -> Option<IdentityAnalysis> {
    let profile = identity.recognition_profile(engine)?;
    let samples = profile
        .embeddings
        .iter()
        .enumerate()
        .map(|(index, embedding)| {
            let leave_one_out = leave_one_out_score(&profile.embeddings, index);
            let (best_voice_id, best_display_name, best_score) =
                best_match(registry, engine, embedding);
            SampleAnalysis {
                index,
                audio: profile.samples.get(index).cloned(),
                own_score: cosine(&profile.centroid, embedding),
                leave_one_out,
                misattributed: best_voice_id != identity.id,
                weak: leave_one_out.is_some_and(|score| score < options.weak_below),
                best_voice_id,
                best_display_name,
                best_score,
            }
        })
        .collect::<Vec<_>>();

    let scored: Vec<f32> = samples.iter().filter_map(|s| s.leave_one_out).collect();
    let mean_leave_one_out = if scored.is_empty() {
        None
    } else {
        Some(scored.iter().sum::<f32>() / scored.len() as f32)
    };

    Some(IdentityAnalysis {
        voice_id: identity.id.clone(),
        display_name: identity.display_name.clone(),
        engine: engine.to_string(),
        mean_leave_one_out,
        samples,
    })
}

/// Analyse every identity holding a profile for `engine`.
#[must_use]
pub fn analyze_registry(
    registry: &VoiceRegistry,
    engine: &str,
    options: AnalysisOptions,
) -> Vec<IdentityAnalysis> {
    registry
        .identities
        .iter()
        .filter_map(|identity| analyze_identity(registry, identity, engine, options))
        .collect()
}

/// Centroid similarity matrix across identities holding `engine`.
#[must_use]
pub fn similarity_matrix(registry: &VoiceRegistry, engine: &str) -> SimilarityMatrix {
    let members: Vec<&VoiceIdentity> = registry
        .identities
        .iter()
        .filter(|identity| identity.recognition_profile(engine).is_some())
        .collect();

    let voice_ids = members.iter().map(|i| i.id.clone()).collect::<Vec<_>>();
    let rows = members
        .iter()
        .map(|identity| {
            let own = identity
                .recognition_profile(engine)
                .map_or(&[] as &[f32], |p| p.centroid.as_slice());
            SimilarityRow {
                voice_id: identity.id.clone(),
                display_name: identity.display_name.clone(),
                scores: members
                    .iter()
                    .map(|other| {
                        other
                            .recognition_profile(engine)
                            .map_or(0.0, |p| cosine(own, &p.centroid))
                    })
                    .collect(),
            }
        })
        .collect();

    SimilarityMatrix { voice_ids, rows }
}

/// Cosine of `embeddings[index]` against the centroid of all the others.
fn leave_one_out_score(embeddings: &[Vec<f32>], index: usize) -> Option<f32> {
    if embeddings.len() < 2 {
        return None;
    }
    let rest: Vec<Vec<f32>> = embeddings
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != index)
        .map(|(_, e)| e.clone())
        .collect();
    let others = centroid(&rest);
    embeddings.get(index).map(|e| cosine(&others, e))
}

/// Best-matching identity for an embedding across the registry.
fn best_match(registry: &VoiceRegistry, engine: &str, embedding: &[f32]) -> (String, String, f32) {
    registry
        .identities
        .iter()
        .filter_map(|identity| {
            identity
                .recognition_profile(engine)
                .map(|profile| (identity, cosine(&profile.centroid, embedding)))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or_else(
            || (String::new(), String::new(), 0.0),
            |(identity, score)| (identity.id.clone(), identity.display_name.clone(), score),
        )
}

#[cfg(test)]
mod tests {
    use super::{AnalysisOptions, analyze_identity, similarity_matrix};
    use crate::identity::RecognitionProfile;
    use crate::registry::VoiceRegistry;

    const ENGINE: &str = "speakrs-wespeaker";

    fn registry_with_contamination() -> VoiceRegistry {
        let mut registry = VoiceRegistry::default();

        let a = registry.create_identity("Alice");
        if let Some(identity) = registry.find_mut(&a) {
            identity.recognition_profiles.push(RecognitionProfile::new(
                ENGINE,
                "v1",
                vec![
                    vec![1.0, 0.0, 0.0],
                    vec![0.98, 0.02, 0.0],
                    // Contaminant: actually Bob's voice enrolled under Alice.
                    vec![0.0, 1.0, 0.0],
                ],
                vec![],
            ));
        }

        let b = registry.create_identity("Bob");
        if let Some(identity) = registry.find_mut(&b) {
            identity.recognition_profiles.push(RecognitionProfile::new(
                ENGINE,
                "v1",
                vec![vec![0.0, 1.0, 0.0], vec![0.02, 0.98, 0.0]],
                vec![],
            ));
        }
        registry
    }

    #[test]
    fn flags_the_contaminated_sample() {
        let registry = registry_with_contamination();
        let alice = registry.find("vox-alice").expect("alice enrolled");
        let analysis = analyze_identity(&registry, alice, ENGINE, AnalysisOptions::default())
            .expect("alice has a profile");

        assert_eq!(analysis.samples.len(), 3);
        assert!(!analysis.samples[0].suspicious());
        assert!(!analysis.samples[1].suspicious());

        let bad = &analysis.samples[2];
        assert!(bad.suspicious(), "contaminant should be flagged");
        assert!(bad.misattributed, "contaminant should match Bob");
        assert_eq!(bad.best_voice_id, "vox-bob");
        assert_eq!(analysis.suspicious_count(), 1);
    }

    #[test]
    fn leave_one_out_is_absent_for_single_sample() {
        let mut registry = VoiceRegistry::default();
        let id = registry.create_identity("Solo");
        if let Some(identity) = registry.find_mut(&id) {
            identity.recognition_profiles.push(RecognitionProfile::new(
                ENGINE,
                "v1",
                vec![vec![1.0, 0.0]],
                vec![],
            ));
        }
        let solo = registry.find(&id).expect("solo enrolled");
        let analysis = analyze_identity(&registry, solo, ENGINE, AnalysisOptions::default())
            .expect("profile present");
        assert!(analysis.samples[0].leave_one_out.is_none());
        assert!(!analysis.samples[0].weak);
    }

    #[test]
    fn similarity_matrix_is_square_and_self_similar() {
        let registry = registry_with_contamination();
        let matrix = similarity_matrix(&registry, ENGINE);
        assert_eq!(matrix.voice_ids.len(), 2);
        assert_eq!(matrix.rows.len(), 2);
        for (row_index, row) in matrix.rows.iter().enumerate() {
            assert_eq!(row.scores.len(), 2);
            assert!((row.scores[row_index] - 1.0).abs() < 1e-5);
        }
    }
}
