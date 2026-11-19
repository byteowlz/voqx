//! Metadata-only recognition readiness, never model preparation or qualification.

use std::path::PathBuf;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::{RegistryProbe, RegistryState, digest, read_registry};
use crate::RecognitionPreprocessing;

/// Presence of the requested engine/model profile, not vector usability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ProfileState {
    /// Registry was not readable or changed during inspection.
    NotChecked,
    /// No profile for the selected engine.
    Missing,
    /// Selected engine exists, but no profile has the required model-version tag.
    IncompatibleModel,
    /// At least one exact engine/model-version profile exists.
    Present,
}

/// Registry-side readiness inspection. Native runtime/model state stays unchecked.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct RecognitionReadiness {
    /// `voqx-recognition-readiness/1`.
    pub protocol: String,
    /// Exact registry inspection to which the counts apply.
    pub registry: RegistryProbe,
    /// Requested preprocessing contract.
    pub preprocessing: String,
    /// Requested engine slot.
    pub engine: String,
    /// Required profile model-version tag, not loaded-checkpoint provenance.
    pub model_version: String,
    /// Registry profile presence.
    pub profile_state: ProfileState,
    /// Exact engine/model-version count, only for a consistent registry read.
    pub matching_profiles: Option<usize>,
    /// Always `not_checked`: this probe never opens a native ML runtime.
    pub runtime_state: String,
    /// Always `not_checked`: no model resolution, download or loading.
    pub model_state: String,
    /// Always `not_checked`: no calibration attestation is performed.
    pub calibration_state: String,
    /// Always false. Matching profile metadata is not permission/qualification.
    pub ready_to_filter: bool,
}

impl RecognitionReadiness {
    /// Inspect registry metadata without inference, mutation or revealing identities.
    #[must_use]
    pub fn inspect(root: PathBuf, preprocessing: RecognitionPreprocessing) -> Self {
        let registry = RegistryProbe::inspect(root);
        let mut report = Self {
            protocol: "voqx-recognition-readiness/1".into(),
            registry,
            preprocessing: preprocessing.to_string(),
            engine: preprocessing.engine_id().into(),
            model_version: preprocessing.model_version().into(),
            profile_state: ProfileState::NotChecked,
            matching_profiles: None,
            runtime_state: "not_checked".into(),
            model_state: "not_checked".into(),
            calibration_state: "not_checked".into(),
            ready_to_filter: false,
        };
        report.inspect_profiles();
        report
    }

    fn inspect_profiles(&mut self) {
        if self.registry.state != RegistryState::Ready {
            return;
        }
        let Ok((registry, bytes)) = read_registry(&self.registry.root) else {
            return;
        };
        if self.registry.revision.as_ref() != Some(&digest(&bytes)) {
            return;
        }
        let profiles: Vec<_> = registry
            .identities
            .iter()
            .flat_map(|identity| &identity.recognition_profiles)
            .filter(|profile| profile.engine == self.engine)
            .collect();
        let matching = profiles
            .iter()
            .filter(|profile| profile.model_version == self.model_version)
            .count();
        self.matching_profiles = Some(matching);
        self.profile_state = if matching > 0 {
            ProfileState::Present
        } else if profiles.is_empty() {
            ProfileState::Missing
        } else {
            ProfileState::IncompatibleModel
        };
    }
}
