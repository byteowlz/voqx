//! `voqx identify` — recognition evidence for an audio file.
//!
//! Output is evidence only (`voice_id`, score, status). Authorization decisions
//! belong to the consuming application.

use std::path::Path;

use anyhow::Result;
use voqx_core::Thresholds;

use crate::audio::{self, RECOGNITION_SAMPLE_RATE};

use super::open_store;

/// Run `voqx identify`.
///
/// # Errors
///
/// Returns an error if the audio cannot be read, the model cannot be loaded, or
/// the registry cannot be read.
pub fn run(
    path: &Path,
    root: Option<&Path>,
    mode: &str,
    thresholds: Thresholds,
    json: bool,
) -> Result<()> {
    let source = audio::read_wav(path)?;
    let mono16 = source.to_rate(RECOGNITION_SAMPLE_RATE)?;

    let store = open_store(root)?;
    let registry = store.load()?;

    let mut embedder = voqx_recognition::Embedder::new(mode, None)?;
    let evidence = embedder.identify(&registry, &mono16.samples, thresholds)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&evidence)?);
        return Ok(());
    }

    match (evidence.voice_id.as_deref(), evidence.score) {
        (Some(matched_id), Some(similarity)) => println!(
            "{matched_id} ({}) score {similarity:.3} [{:?}]",
            evidence.display_name.as_deref().unwrap_or("-"),
            evidence.status
        ),
        _ => println!("no match [{:?}]", evidence.status),
    }
    Ok(())
}
