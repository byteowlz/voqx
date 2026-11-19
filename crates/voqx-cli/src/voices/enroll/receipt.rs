//! Enrollment effect receipts. Errors never echo transcripts, names or paths.

use schemars::JsonSchema;
use serde::Serialize;

/// Versioned enrollment response for both success and failure.
#[derive(Debug, Serialize, JsonSchema)]
pub struct EnrollmentReport {
    /// `voqx-enrollment/1`.
    pub protocol: &'static str,
    /// Effect outcome, not recognition qualification or consent beyond this operation.
    #[serde(flatten)]
    pub outcome: EnrollmentOutcome,
}

/// Structured enrollment outcome; uncertain effects must not be blindly retried.
#[derive(Debug, Serialize, JsonSchema)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum EnrollmentOutcome {
    /// Registry replacement succeeded under the writer lock.
    Committed {
        /// Canonical identity, never inferred from CLI text.
        voice_id: String,
        /// Attached reusable source.
        source_id: String,
        /// SHA-256 of exact JSON bytes at this commit, not a bank fingerprint.
        registry_revision: String,
        /// Selected preprocessing only when recognition was requested.
        preprocessing: Option<String>,
        /// Selected recognition slot, if any.
        recognition_engine: Option<String>,
        /// Profile version tag, not loaded-checkpoint attestation.
        recognition_model_version: Option<String>,
        /// Synthesis engine only when explicitly requested.
        synthesis_engine: Option<String>,
    },
    /// Request failed or was refused. No identity is invented.
    Failed {
        /// Stable, bounded failure class.
        code: FailureCode,
        /// Whether a registry write might have happened.
        registry_effect: RegistryEffect,
        /// Asset writes may precede a failed registry replacement.
        assets_may_have_changed: bool,
    },
}

/// Stable enrollment failure classes.
#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FailureCode {
    /// Explicit `--yes` was not supplied. No preparation or write started.
    ConfirmationRequired,
    /// Enrollment dry-run is not implemented; no operation was started.
    DryRunUnsupported,
    /// Recognition was requested from a registry-only build.
    RecognitionNotCompiled,
    /// Audio/transcript/embedding preparation failed before registry mutation.
    PreparationFailed,
    /// Registry/asset update failed; inspect state before retrying.
    RegistryUpdateFailed,
    /// Expected registry revision did not match, before the mutation closure ran.
    StaleRevision,
}

/// Effect certainty, deliberately conservative on a failed write phase.
#[derive(Debug, Clone, Copy, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RegistryEffect {
    /// No registry mutation started.
    NotStarted,
    /// The caller must inspect current state; never automatically retry enrollment.
    Uncertain,
}

impl EnrollmentReport {
    pub(super) const fn failed(code: FailureCode, effect: RegistryEffect, assets: bool) -> Self {
        Self {
            protocol: "voqx-enrollment/1",
            outcome: EnrollmentOutcome::Failed {
                code,
                registry_effect: effect,
                assets_may_have_changed: assets,
            },
        }
    }
}
