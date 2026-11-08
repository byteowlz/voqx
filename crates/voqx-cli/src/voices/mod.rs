//! Voice Identity registry commands: identities, enrollment, references,
//! recognition, and importers for existing tool-local stores.

pub mod enroll;
#[cfg(feature = "recognition")]
pub mod identify;
pub mod identities;
pub mod import;

use std::path::Path;

use anyhow::{Context, Result, bail};
use voqx_core::registry::RegistryStore;
use voqx_core::registry_paths::RegistryPaths;
use voqx_core::{VoiceIdentity, VoiceRegistry};

/// Open the registry store, honouring an explicit root override.
///
/// # Errors
///
/// Returns an error if the default location cannot be resolved.
pub fn open_store(root: Option<&Path>) -> Result<RegistryStore> {
    root.map_or_else(RegistryStore::from_default, |path| {
        Ok(RegistryStore::new(RegistryPaths::from_root(
            path.to_path_buf(),
        )))
    })
}

/// Resolve an identity by exact `voice_id`, then by display name or alias.
///
/// # Errors
///
/// Returns an error if nothing matches.
pub fn resolve<'a>(registry: &'a VoiceRegistry, key: &str) -> Result<&'a VoiceIdentity> {
    if let Some(identity) = registry.find(key) {
        return Ok(identity);
    }
    let id = registry
        .id_for_name(key)
        .with_context(|| format!("no voice identity matching '{key}'"))?
        .to_string();
    registry
        .find(&id)
        .with_context(|| format!("no voice identity matching '{key}'"))
}

/// Resolve an identity key to its stable `voice_id`.
///
/// # Errors
///
/// Returns an error if nothing matches.
pub fn resolve_id(registry: &VoiceRegistry, key: &str) -> Result<String> {
    Ok(resolve(registry, key)?.id.clone())
}

/// Express `absolute` as a registry-relative path (forward-slashed).
///
/// # Errors
///
/// Returns an error if the path is outside the registry root or not UTF-8.
pub fn relative_to_root(paths: &RegistryPaths, absolute: &Path) -> Result<String> {
    let relative = absolute.strip_prefix(&paths.root).with_context(|| {
        format!(
            "{} is outside the registry root {}",
            absolute.display(),
            paths.root.display()
        )
    })?;
    let text = relative
        .to_str()
        .with_context(|| format!("non-UTF-8 path {}", relative.display()))?;
    Ok(text.replace('\\', "/"))
}

/// Read a transcript from an explicit string or a file, trimming whitespace.
///
/// # Errors
///
/// Returns an error if neither is provided, the file cannot be read, or the
/// resulting transcript is empty.
pub fn read_transcript(text: Option<&str>, file: Option<&Path>) -> Result<String> {
    let body = match (text, file) {
        (Some(value), _) => value.to_string(),
        (None, Some(path)) => std::fs::read_to_string(path)
            .with_context(|| format!("reading transcript {}", path.display()))?,
        (None, None) => bail!("provide --transcript or --transcript-file"),
    };
    let trimmed = body.trim().to_string();
    if trimmed.is_empty() {
        bail!("transcript is empty");
    }
    Ok(trimmed)
}
