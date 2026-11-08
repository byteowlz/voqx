//! Registry inspection, enrollment-quality analysis, and cleanup endpoints.
//!
//! The API is a thin transport over `voqx-core`: analysis is pure cosine math
//! from the stored embeddings, and every mutation goes through the same
//! lock-guarded `RegistryStore::edit` the CLI uses.

use std::path::{Path, PathBuf};

use axum::extract::{Path as AxumPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use voqx_core::analysis::{AnalysisOptions, IdentityAnalysis, SimilarityMatrix};
use voqx_core::registry::RegistryStore;
use voqx_core::{VoiceIdentity, analyze_registry, similarity_matrix};

/// Recognition engine the UI analyses.
const DEFAULT_ENGINE: &str = "speakrs-wespeaker";

/// Shared state for the registry endpoints.
#[derive(Clone)]
pub struct RegistryState {
    /// Store bound to the registry root being served.
    pub store: RegistryStore,
}

impl std::fmt::Debug for RegistryState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegistryState").finish_non_exhaustive()
    }
}

/// Build the `/api` router.
pub fn router(state: RegistryState) -> Router {
    Router::new()
        .route("/api/registry", get(list_identities))
        .route("/api/identities/{voice_id}", get(get_identity))
        .route("/api/identities/{voice_id}/rename", post(rename_identity))
        .route(
            "/api/identities/{voice_id}/samples/{index}",
            post(drop_sample),
        )
        .route("/api/analysis", get(get_analysis))
        .route("/api/audio", get(get_audio))
        .with_state(state)
}

/// An API failure with a stable shape.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn new(status: StatusCode, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    fn internal(err: &anyhow::Error) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, err.to_string())
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(serde_json::json!({ "error": self.message })),
        )
            .into_response()
    }
}

/// Summary row for the identity list.
#[derive(Debug, Serialize)]
struct IdentitySummary {
    voice_id: String,
    display_name: String,
    aliases: Vec<String>,
    recognition: bool,
    synthesis: bool,
    wake: bool,
    recognition_samples: usize,
    synthesis_references: usize,
}

/// Response for `GET /api/registry`.
#[derive(Debug, Serialize)]
struct RegistryResponse {
    root: String,
    identities: Vec<IdentitySummary>,
}

async fn list_identities(
    State(state): State<RegistryState>,
) -> Result<Json<RegistryResponse>, ApiError> {
    let registry = state.store.load().map_err(|e| ApiError::internal(&e))?;
    let identities = registry
        .identities
        .iter()
        .map(|identity| {
            let caps = identity.capabilities();
            IdentitySummary {
                voice_id: identity.id.clone(),
                display_name: identity.display_name.clone(),
                aliases: identity.aliases.clone(),
                recognition: caps.recognition,
                synthesis: caps.synthesis,
                wake: caps.wake,
                recognition_samples: identity
                    .recognition_profiles
                    .iter()
                    .map(|p| p.embeddings.len())
                    .sum(),
                synthesis_references: identity
                    .synthesis_profiles
                    .iter()
                    .map(|p| p.references.len())
                    .sum(),
            }
        })
        .collect();

    Ok(Json(RegistryResponse {
        root: state.store.paths().root.display().to_string(),
        identities,
    }))
}

async fn get_identity(
    State(state): State<RegistryState>,
    AxumPath(voice_id): AxumPath<String>,
) -> Result<Json<VoiceIdentity>, ApiError> {
    let registry = state.store.load().map_err(|e| ApiError::internal(&e))?;
    registry.find(&voice_id).map_or_else(
        || Err(ApiError::new(StatusCode::NOT_FOUND, "unknown voice_id")),
        |identity| Ok(Json(identity.clone())),
    )
}

/// Query for `GET /api/analysis`.
#[derive(Debug, Deserialize)]
struct AnalysisQuery {
    engine: Option<String>,
    weak_below: Option<f32>,
}

/// Response for `GET /api/analysis`.
#[derive(Debug, Serialize)]
struct AnalysisResponse {
    engine: String,
    analyses: Vec<IdentityAnalysis>,
    matrix: SimilarityMatrix,
}

async fn get_analysis(
    State(state): State<RegistryState>,
    Query(query): Query<AnalysisQuery>,
) -> Result<Json<AnalysisResponse>, ApiError> {
    let registry = state.store.load().map_err(|e| ApiError::internal(&e))?;
    let engine = query.engine.unwrap_or_else(|| DEFAULT_ENGINE.to_string());
    let options = query
        .weak_below
        .map_or_else(AnalysisOptions::default, |weak_below| AnalysisOptions {
            weak_below,
        });

    Ok(Json(AnalysisResponse {
        analyses: analyze_registry(&registry, &engine, options),
        matrix: similarity_matrix(&registry, &engine),
        engine,
    }))
}

/// Body for renaming an identity.
#[derive(Debug, Deserialize)]
struct RenameRequest {
    display_name: String,
}

async fn rename_identity(
    State(state): State<RegistryState>,
    AxumPath(voice_id): AxumPath<String>,
    Json(body): Json<RenameRequest>,
) -> Result<Json<serde_json::Value>, ApiError> {
    if body.display_name.trim().is_empty() {
        return Err(ApiError::new(
            StatusCode::BAD_REQUEST,
            "display_name must not be empty",
        ));
    }
    state
        .store
        .edit(|registry| registry.rename(&voice_id, body.display_name.trim()))
        .map_err(|e| ApiError::internal(&e))?;
    Ok(Json(serde_json::json!({ "ok": true })))
}

/// Query for dropping a sample.
#[derive(Debug, Deserialize)]
struct DropSampleQuery {
    engine: Option<String>,
    /// Also delete the sample's audio file from disk.
    delete_audio: Option<bool>,
}

async fn drop_sample(
    State(state): State<RegistryState>,
    AxumPath((voice_id, index)): AxumPath<(String, usize)>,
    Query(query): Query<DropSampleQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let engine = query.engine.unwrap_or_else(|| DEFAULT_ENGINE.to_string());
    let root = state.store.paths().root.clone();

    let removed = state
        .store
        .edit(|registry| {
            let identity = registry
                .find_mut(&voice_id)
                .ok_or_else(|| anyhow::anyhow!("unknown voice_id: {voice_id}"))?;
            let profile = identity
                .recognition_profiles
                .iter_mut()
                .find(|p| p.engine == engine)
                .ok_or_else(|| anyhow::anyhow!("no {engine} profile on {voice_id}"))?;
            let audio = profile.remove_sample(index);
            identity.touch();
            Ok(audio)
        })
        .map_err(|e| ApiError::internal(&e))?;

    if query.delete_audio.unwrap_or(false)
        && let Some(relative) = removed.as_deref()
        && let Some(path) = safe_join(&root, relative)
    {
        let _ = std::fs::remove_file(path);
    }

    Ok(Json(
        serde_json::json!({ "ok": true, "removed_audio": removed }),
    ))
}

/// Query for `GET /api/audio`.
#[derive(Debug, Deserialize)]
struct AudioQuery {
    /// Registry-relative path, as stored on a profile.
    path: String,
}

async fn get_audio(
    State(state): State<RegistryState>,
    Query(query): Query<AudioQuery>,
) -> Result<Response, ApiError> {
    let root = state.store.paths().root.clone();
    let path = safe_join(&root, &query.path)
        .ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "path escapes the registry root"))?;
    let bytes = std::fs::read(&path)
        .map_err(|err| ApiError::new(StatusCode::NOT_FOUND, err.to_string()))?;
    Ok(([(header::CONTENT_TYPE, "audio/wav")], bytes).into_response())
}

/// Join a registry-relative path to `root`, refusing anything that escapes it.
///
/// Audio paths come from the registry, but they are attacker-shaped input on the
/// wire, so traversal is rejected before touching the filesystem.
fn safe_join(root: &Path, relative: &str) -> Option<PathBuf> {
    if relative.is_empty() || Path::new(relative).is_absolute() {
        return None;
    }
    let candidate = root.join(relative);
    let canonical_root = root.canonicalize().ok()?;
    let canonical = candidate.canonicalize().ok()?;
    canonical.starts_with(&canonical_root).then_some(canonical)
}

#[cfg(test)]
mod tests {
    use super::safe_join;

    #[test]
    fn rejects_traversal_and_absolute_paths() {
        let dir = std::env::temp_dir();
        assert!(safe_join(&dir, "../etc/passwd").is_none());
        assert!(safe_join(&dir, "/etc/passwd").is_none());
        assert!(safe_join(&dir, "").is_none());
    }

    #[test]
    fn accepts_a_real_child_path() {
        let dir = std::env::temp_dir().join(format!("voqx-api-{}", std::process::id()));
        let file = dir.join("ok.wav");
        std::fs::create_dir_all(&dir).expect("creating dir");
        std::fs::write(&file, b"x").expect("writing file");
        assert!(safe_join(&dir, "ok.wav").is_some());
        std::fs::remove_dir_all(&dir).ok();
    }
}
