//! Static build discovery. Intentionally independent of configuration, registry and model loading.

use anyhow::Result;
use serde::Serialize;

#[derive(Debug, Serialize)]
struct RecognitionCapabilities {
    compiled: bool,
    identify_json: bool,
    enroll_audio: bool,
    enroll_json: bool,
    preprocessing: &'static [&'static str],
}

#[derive(Debug, Serialize)]
struct Capabilities {
    protocol: &'static str,
    version: &'static str,
    recognition: RecognitionCapabilities,
    registry_inspection_json: bool,
    microphone_capture: bool,
    model_state: &'static str,
    registry_state: &'static str,
}

const fn describe() -> Capabilities {
    let compiled = cfg!(feature = "recognition");
    Capabilities {
        protocol: "voqx-capabilities/1",
        version: env!("CARGO_PKG_VERSION"),
        recognition: RecognitionCapabilities {
            compiled,
            identify_json: compiled,
            enroll_audio: compiled,
            enroll_json: false,
            preprocessing: if compiled {
                &["uniform", "valid-v1"]
            } else {
                &[]
            },
        },
        registry_inspection_json: true,
        microphone_capture: false,
        model_state: "not_checked",
        registry_state: "not_checked",
    }
}

/// Print the build contract without initializing application state.
///
/// # Errors
/// Returns an error if serialization fails.
pub fn run(yaml: bool) -> Result<()> {
    let capabilities = describe();
    if yaml {
        print!("{}", serde_yaml::to_string(&capabilities)?);
    } else {
        println!("{}", serde_json::to_string_pretty(&capabilities)?);
    }
    Ok(())
}
