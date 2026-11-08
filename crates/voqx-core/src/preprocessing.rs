//! Versioned recognition preprocessing. Different variants occupy different profile slots.

use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

/// Pooling contract for the pinned WeSpeaker waveform/weights graph.
///
/// This describes preprocessing, not a calibrated speaker filter. Profiles from
/// different variants must never be compared or combined.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecognitionPreprocessing {
    /// Legacy all-one pooling weights, including waveform padding.
    #[default]
    #[serde(rename = "uniform")]
    Uniform,
    /// Exclude padded time bins from pooling; keep real silence and encoder context.
    #[serde(rename = "valid-v1")]
    ValidV1,
}

impl RecognitionPreprocessing {
    /// Engine/profile key. A preprocessing change requires a different key.
    #[must_use]
    pub const fn engine_id(self) -> &'static str {
        match self {
            Self::Uniform => "speakrs-wespeaker",
            Self::ValidV1 => "speakrs-wespeaker-valid-v1",
        }
    }

    /// Version stored alongside enrollment vectors.
    #[must_use]
    pub const fn model_version(self) -> &'static str {
        match self {
            Self::Uniform => "speakrs-0.4",
            Self::ValidV1 => "speakrs-0.4.2-valid-v1",
        }
    }
}

impl fmt::Display for RecognitionPreprocessing {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Uniform => "uniform",
            Self::ValidV1 => "valid-v1",
        })
    }
}

impl FromStr for RecognitionPreprocessing {
    type Err = &'static str;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "uniform" => Ok(Self::Uniform),
            "valid-v1" => Ok(Self::ValidV1),
            _ => Err("preprocessing must be uniform or valid-v1"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RecognitionPreprocessing;
    use crate::{RecognitionProfile, RecognitionStatus, Thresholds, VoiceIdentity, VoiceRegistry};

    #[test]
    fn variants_roundtrip_and_keep_distinct_profile_keys() -> anyhow::Result<()> {
        for variant in [
            RecognitionPreprocessing::Uniform,
            RecognitionPreprocessing::ValidV1,
        ] {
            assert_eq!(
                variant
                    .to_string()
                    .parse::<RecognitionPreprocessing>()
                    .map_err(anyhow::Error::msg)?,
                variant
            );
            let json = serde_json::to_string(&variant)?;
            assert_eq!(
                serde_json::from_str::<RecognitionPreprocessing>(&json)?,
                variant
            );
        }
        assert_eq!(
            RecognitionPreprocessing::default(),
            RecognitionPreprocessing::Uniform
        );
        assert!("valid-v2".parse::<RecognitionPreprocessing>().is_err());
        assert_ne!(
            RecognitionPreprocessing::Uniform.engine_id(),
            RecognitionPreprocessing::ValidV1.engine_id()
        );
        Ok(())
    }

    #[test]
    fn queries_cannot_match_a_profile_from_the_other_preprocessing() {
        for (enrolled, queried) in [
            (
                RecognitionPreprocessing::Uniform,
                RecognitionPreprocessing::ValidV1,
            ),
            (
                RecognitionPreprocessing::ValidV1,
                RecognitionPreprocessing::Uniform,
            ),
        ] {
            let mut identity = VoiceIdentity::new("vox-fixture", "Fixture");
            identity.recognition_profiles.push(RecognitionProfile::new(
                enrolled.engine_id(),
                enrolled.model_version(),
                vec![vec![1.0, 0.0]],
                vec![],
            ));
            let registry = VoiceRegistry {
                identities: vec![identity],
                ..VoiceRegistry::default()
            };
            let result = registry.identify(&[1.0, 0.0], queried.engine_id(), Thresholds::default());
            assert_eq!(
                result,
                crate::Recognition::incompatible(queried.engine_id())
            );
            assert_eq!(result.status, RecognitionStatus::IncompatibleModel);
        }
    }
}
