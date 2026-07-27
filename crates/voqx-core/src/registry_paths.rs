//! Filesystem layout for the shared voice registry.
//!
//! The registry lives beside — not under — any single app's data directory, at
//! `$XDG_DATA_HOME/voice-registry`, so trnscrb, spqx, eaRS, and Foxline share
//! one identity store. `registry.json` is authoritative; heavy assets live under
//! `identities/<voice_id>/` referenced by registry-relative paths.

use std::path::PathBuf;

use anyhow::Result;

use crate::paths::data_home_root;

/// The registry directory name under the XDG data root.
pub const REGISTRY_DIR_NAME: &str = "voice-registry";

/// Durable filesystem layout for one voice registry root.
#[derive(Debug, Clone)]
pub struct RegistryPaths {
    /// Registry root directory (`.../voice-registry`).
    pub root: PathBuf,
    /// Authoritative registry file (`root/registry.json`).
    pub registry_file: PathBuf,
    /// Per-identity asset directory (`root/identities`).
    pub identities_dir: PathBuf,
}

impl RegistryPaths {
    /// Build paths rooted at the default shared XDG data location.
    ///
    /// # Errors
    ///
    /// Returns an error if no base directory can be determined.
    pub fn from_default_data_dir() -> Result<Self> {
        Ok(Self::from_root(default_registry_root()?))
    }

    /// Build paths rooted at an explicit registry directory.
    #[must_use]
    pub fn from_root(root: PathBuf) -> Self {
        let registry_file = root.join("registry.json");
        let identities_dir = root.join("identities");
        Self {
            root,
            registry_file,
            identities_dir,
        }
    }

    /// The asset directory for one identity (`root/identities/<voice_id>`).
    #[must_use]
    pub fn identity_dir(&self, voice_id: &str) -> PathBuf {
        self.identities_dir.join(voice_id)
    }
}

/// The default registry root (`$XDG_DATA_HOME/voice-registry`).
///
/// # Errors
///
/// Returns an error if no base directory can be determined.
pub fn default_registry_root() -> Result<PathBuf> {
    Ok(data_home_root()?.join(REGISTRY_DIR_NAME))
}

#[cfg(test)]
mod tests {
    use super::RegistryPaths;
    use std::path::PathBuf;

    #[test]
    fn layout_is_rooted() {
        let p = RegistryPaths::from_root(PathBuf::from("/r"));
        assert_eq!(p.registry_file, PathBuf::from("/r/registry.json"));
        assert_eq!(
            p.identity_dir("vox-tommy"),
            PathBuf::from("/r/identities/vox-tommy")
        );
    }
}
