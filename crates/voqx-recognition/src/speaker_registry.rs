//! Read-only adapters for the canonical registry; no parallel identity store.
//!
//! Selection is by exact `voice_id` and both engine and `model_version`. No old
//! WeSpeaker vector, centroid-only fallback or display-name matching is used.

use voqx_core::{RecognitionProfile, VoiceRegistry};

use crate::speaker::{
    EmbeddingSpace, ReferenceSnapshot, ReferenceVoice, SpeakerEmbedding, SpeakerError,
    SpeakerMatcher, SpeakerResult,
};

/// Select explicitly requested identities into a bounded inference snapshot.
///
/// An empty selection remains empty; it never silently means every identity.
/// All requested voices must have exactly one compatible, nonempty profile.
///
/// # Errors
/// Rejects unknown IDs, missing/duplicate/incompatible profiles and invalid vectors.
pub fn reference_snapshot(
    registry: &VoiceRegistry,
    space: &EmbeddingSpace,
    voice_ids: &[&str],
) -> SpeakerResult<ReferenceSnapshot> {
    let mut voices = Vec::with_capacity(voice_ids.len().min(64));
    if voice_ids.len() > 64 {
        return Err(SpeakerError::InvalidProfile("more than 64 selected voices"));
    }
    for voice_id in voice_ids {
        let identity = registry
            .find(voice_id)
            .ok_or(SpeakerError::InvalidProfile("unknown canonical voice_id"))?;
        let mut profiles = identity
            .recognition_profiles
            .iter()
            .filter(|profile| profile.engine == space.engine);
        let profile = profiles.next().ok_or(SpeakerError::InvalidProfile(
            "missing compatible recognition profile",
        ))?;
        if profiles.next().is_some() || profile.model_version != space.model_version {
            return Err(SpeakerError::InvalidProfile(
                "duplicate slot or incompatible model/preprocessing version",
            ));
        }
        voices.push(ReferenceVoice {
            voice_id: identity.id.clone(),
            embeddings: profile.embeddings.clone(),
        });
    }
    let snapshot = ReferenceSnapshot {
        space: space.clone(),
        voices,
    };
    // Validate the full selection; never silently drop malformed profiles.
    SpeakerMatcher::new(snapshot.clone())?;
    Ok(snapshot)
}

/// Prepare a recognition-only profile for a caller-reviewed enrollment operation.
///
/// Does not create an identity, write the registry, retain audio, infer consent or
/// enable synthesis. Samples are initially empty; reviewed source paths stay owned
/// by the host's enrollment transaction.
///
/// # Errors
/// Rejects empty/excessive references and any mixed model/preprocessing spaces.
pub fn recognition_profile(embeddings: &[SpeakerEmbedding]) -> SpeakerResult<RecognitionProfile> {
    let first = embeddings
        .first()
        .ok_or(SpeakerError::InvalidProfile("enrollment needs references"))?;
    if embeddings.len() > 32
        || embeddings
            .iter()
            .any(|embedding| embedding.space() != first.space())
    {
        return Err(SpeakerError::InvalidProfile(
            "too many references or mixed embedding spaces",
        ));
    }
    Ok(RecognitionProfile::new(
        first.space().engine.clone(),
        first.space().model_version.clone(),
        embeddings
            .iter()
            .map(|embedding| embedding.values().to_vec())
            .collect(),
        Vec::new(),
    ))
}
