//! The durable voice registry: a versioned collection of [`VoiceIdentity`]
//! records with atomic, lock-guarded persistence and cosine-based recognition.

use std::time::Duration;

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::atomic::{LockGuard, write_atomic};
use crate::identity::VoiceIdentity;
use crate::ids::allocate_voice_id;
use crate::math::cosine;
use crate::recognition::{Recognition, Thresholds};
use crate::registry_paths::RegistryPaths;

/// On-disk registry format version.
pub const REGISTRY_VERSION: u32 = 1;

/// How long to wait for the write lock before giving up.
const LOCK_TIMEOUT: Duration = Duration::from_secs(5);

/// The durable collection of all known voice identities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceRegistry {
    /// Format version for future migrations.
    pub version: u32,
    /// Known identities.
    #[serde(default)]
    pub identities: Vec<VoiceIdentity>,
}

impl Default for VoiceRegistry {
    fn default() -> Self {
        Self {
            version: REGISTRY_VERSION,
            identities: Vec::new(),
        }
    }
}

impl VoiceRegistry {
    /// Load the registry from `paths`, or an empty registry if none exists.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read/parsed, or was
    /// written by a newer, unsupported format version.
    pub fn load(paths: &RegistryPaths) -> Result<Self> {
        if !paths.registry_file.exists() {
            return Ok(Self::default());
        }
        let body = std::fs::read_to_string(&paths.registry_file)
            .with_context(|| format!("reading registry {}", paths.registry_file.display()))?;
        let registry: Self = serde_json::from_str(&body)
            .with_context(|| format!("parsing registry {}", paths.registry_file.display()))?;
        if registry.version > REGISTRY_VERSION {
            bail!(
                "voice registry {} is version {} but this build supports up to {}",
                paths.registry_file.display(),
                registry.version,
                REGISTRY_VERSION
            );
        }
        Ok(registry)
    }

    /// Persist the registry to `paths` atomically.
    ///
    /// # Errors
    ///
    /// Returns an error if serialization or the atomic write fails.
    pub fn save(&self, paths: &RegistryPaths) -> Result<()> {
        let body = serde_json::to_string_pretty(self).context("serializing registry")?;
        write_atomic(&paths.registry_file, body.as_bytes())
    }

    /// Find an identity by exact `voice_id`.
    #[must_use]
    pub fn find(&self, voice_id: &str) -> Option<&VoiceIdentity> {
        self.identities.iter().find(|i| i.id == voice_id)
    }

    /// Find a mutable identity by exact `voice_id`.
    #[must_use]
    pub fn find_mut(&mut self, voice_id: &str) -> Option<&mut VoiceIdentity> {
        self.identities.iter_mut().find(|i| i.id == voice_id)
    }

    /// The `voice_id` of an identity whose display name or alias equals `name`.
    #[must_use]
    pub fn id_for_name(&self, name: &str) -> Option<&str> {
        self.identities
            .iter()
            .find(|i| i.matches_name(name))
            .map(|i| i.id.as_str())
    }

    /// Create a new identity with a freshly allocated id; returns its id.
    pub fn create_identity(&mut self, display_name: &str) -> String {
        let id = allocate_voice_id(display_name, |candidate| self.find(candidate).is_some());
        self.identities
            .push(VoiceIdentity::new(id.clone(), display_name));
        id
    }

    /// Rename an identity, preserving the old name as an alias.
    ///
    /// # Errors
    ///
    /// Returns an error if `voice_id` is unknown.
    pub fn rename(&mut self, voice_id: &str, display_name: &str) -> Result<()> {
        let identity = self
            .find_mut(voice_id)
            .with_context(|| format!("unknown voice_id: {voice_id}"))?;
        if identity.display_name != display_name
            && !identity.aliases.iter().any(|a| a == &identity.display_name)
        {
            let old = identity.display_name.clone();
            identity.aliases.push(old);
        }
        identity.display_name = display_name.to_string();
        identity.touch();
        Ok(())
    }

    /// Remove an identity by `voice_id`. Returns whether one was removed.
    ///
    /// Note: this does not delete on-disk asset directories; callers that own
    /// the asset layout are responsible for that.
    pub fn remove(&mut self, voice_id: &str) -> bool {
        let before = self.identities.len();
        self.identities.retain(|i| i.id != voice_id);
        self.identities.len() != before
    }

    /// Best recognition evidence for `embedding` against profiles for `engine`.
    ///
    /// Returns [`crate::recognition::RecognitionStatus::IncompatibleModel`] when
    /// no identity has a recognition profile for `engine`. Otherwise returns the
    /// best-matching identity, its cosine score, and the classified status —
    /// evidence only, never authorization.
    #[must_use]
    pub fn identify(&self, embedding: &[f32], engine: &str, thresholds: Thresholds) -> Recognition {
        let mut best: Option<(&VoiceIdentity, f32)> = None;
        let mut engine_seen = false;
        for identity in &self.identities {
            if let Some(profile) = identity.recognition_profile(engine) {
                engine_seen = true;
                let score = cosine(&profile.centroid, embedding);
                if best.is_none_or(|(_, s)| score > s) {
                    best = Some((identity, score));
                }
            }
        }
        if !engine_seen {
            return Recognition::incompatible(engine);
        }
        match best {
            Some((identity, score)) => Recognition {
                voice_id: Some(identity.id.clone()),
                display_name: Some(identity.display_name.clone()),
                engine: engine.to_string(),
                score: Some(score),
                status: thresholds.classify(score),
            },
            None => Recognition::incompatible(engine),
        }
    }
}

/// A registry bound to a filesystem location, offering lock-guarded mutation.
#[derive(Debug, Clone)]
pub struct RegistryStore {
    paths: RegistryPaths,
}

impl RegistryStore {
    /// Bind a store to `paths`.
    #[must_use]
    pub const fn new(paths: RegistryPaths) -> Self {
        Self { paths }
    }

    /// Bind a store to the default XDG registry location.
    ///
    /// # Errors
    ///
    /// Returns an error if no base directory can be determined.
    pub fn from_default() -> Result<Self> {
        Ok(Self::new(RegistryPaths::from_default_data_dir()?))
    }

    /// The bound filesystem paths.
    #[must_use]
    pub const fn paths(&self) -> &RegistryPaths {
        &self.paths
    }

    /// Load the current registry without taking the lock (read-only).
    ///
    /// # Errors
    ///
    /// Returns an error if the registry cannot be read or parsed.
    pub fn load(&self) -> Result<VoiceRegistry> {
        VoiceRegistry::load(&self.paths)
    }

    /// Take the write lock, load, apply `mutate`, save atomically, and return
    /// the closure's value. The lock is released before returning.
    ///
    /// # Errors
    ///
    /// Returns an error if the lock cannot be acquired within the timeout, or
    /// loading/saving fails, or `mutate` returns an error (in which case no
    /// changes are written).
    pub fn edit<T>(&self, mutate: impl FnOnce(&mut VoiceRegistry) -> Result<T>) -> Result<T> {
        let lock_path = self.paths.root.join(".lock");
        let _guard = LockGuard::acquire(&lock_path, LOCK_TIMEOUT)?;
        let mut registry = VoiceRegistry::load(&self.paths)?;
        let value = mutate(&mut registry)?;
        registry.save(&self.paths)?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests {
    use super::{RegistryStore, VoiceRegistry};
    use crate::atomic::LockGuard;
    use crate::identity::RecognitionProfile;
    use crate::recognition::{RecognitionStatus, Thresholds};
    use crate::registry_paths::RegistryPaths;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn temp_paths() -> RegistryPaths {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_nanos());
        let root = std::env::temp_dir().join(format!("voqx-reg-{}-{nanos}", std::process::id()));
        RegistryPaths::from_root(root)
    }

    fn profile() -> RecognitionProfile {
        RecognitionProfile::new("speakrs-wespeaker", "v1", vec![vec![1.0, 0.0, 0.0]], vec![])
    }

    fn registry_with_tommy() -> (VoiceRegistry, String) {
        let mut registry = VoiceRegistry::default();
        let id = registry.create_identity("Tommy");
        if let Some(identity) = registry.find_mut(&id) {
            identity.recognition_profiles.push(profile());
        }
        (registry, id)
    }

    #[test]
    fn create_load_roundtrip_and_capabilities() {
        let paths = temp_paths();
        let store = RegistryStore::new(paths.clone());

        let id = store
            .edit(|reg| {
                let id = reg.create_identity("Tommy");
                if let Some(identity) = reg.find_mut(&id) {
                    identity.recognition_profiles.push(profile());
                    identity.touch();
                }
                Ok(id)
            })
            .expect("enrolling identity");
        assert_eq!(id, "vox-tommy");

        let loaded = store.load().expect("reloading registry");
        let identity = loaded.find(&id).expect("identity persisted");
        assert!(identity.capabilities().recognition);
        assert_eq!(loaded.id_for_name("Tommy"), Some("vox-tommy"));

        std::fs::remove_dir_all(&paths.root).ok();
    }

    #[test]
    fn identify_returns_evidence() {
        let (registry, _id) = registry_with_tommy();
        let thresholds = Thresholds::default();

        let hit = registry.identify(&[0.95, 0.05, 0.0], "speakrs-wespeaker", thresholds);
        assert_eq!(hit.voice_id.as_deref(), Some("vox-tommy"));
        assert_eq!(hit.status, RecognitionStatus::Confident);

        let other_engine = registry.identify(&[1.0, 0.0, 0.0], "other-engine", thresholds);
        assert_eq!(other_engine.status, RecognitionStatus::IncompatibleModel);
        assert!(other_engine.voice_id.is_none());
    }

    #[test]
    fn rename_keeps_old_name_as_alias() {
        let (mut registry, id) = registry_with_tommy();
        registry.rename(&id, "Tommy Falkowski").expect("renaming");
        let identity = registry.find(&id).expect("renamed identity");
        assert_eq!(identity.display_name, "Tommy Falkowski");
        assert!(identity.matches_name("Tommy"));
    }

    #[test]
    fn edit_blocks_while_lock_is_held() {
        let paths = temp_paths();
        let store = RegistryStore::new(paths.clone());
        std::fs::create_dir_all(&paths.root).expect("creating registry root");

        let held = LockGuard::acquire(&paths.root.join(".lock"), Duration::from_millis(50))
            .expect("holding lock");
        let outcome = std::thread::scope(|scope| {
            scope
                .spawn(|| store.edit(|reg| Ok(reg.create_identity("X"))))
                .join()
                .map_err(|_| ())
        });
        drop(held);
        let inner = outcome.expect("worker thread finished");
        assert!(inner.is_err(), "edit should fail while the lock is held");

        std::fs::remove_dir_all(&paths.root).ok();
    }
}
