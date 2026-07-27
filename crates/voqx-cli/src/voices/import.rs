//! `voqx import` — migrate existing tool-local voice stores into the shared
//! registry.
//!
//! Importers never merge by display name: a name collision is reported and
//! skipped so linking two stores to one identity stays an explicit act.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Subcommand;
use serde::Deserialize;
use voqx_core::identity::{RecognitionProfile, SynthesisProfile, SynthesisReference};
use voqx_core::registry_paths::RegistryPaths;
use voqx_core::{VoiceIdentity, VoiceRegistry};

use crate::audio;

use super::{open_store, relative_to_root};

/// Engine identifier used for imported spqx references.
const SPQX_ENGINE: &str = "spqx-qwen3";

/// Import subcommands.
#[derive(Debug, Subcommand)]
pub enum ImportCommand {
    /// Import speakers (recognition profiles) from a trnscrb registry
    Trnscrb {
        /// trnscrb `registry.json` (default: `$XDG_DATA_HOME/trnscrb/speakers/registry.json`)
        #[arg(long, value_name = "PATH")]
        from: Option<PathBuf>,
        /// Report what would change without writing
        #[arg(long)]
        dry_run: bool,
    },
    /// Import named cloning voices (synthesis references) from spqx
    Spqx {
        /// spqx voices directory (default: `$XDG_DATA_HOME/spqx/voices`)
        #[arg(long, value_name = "PATH")]
        from: Option<PathBuf>,
        /// Report what would change without writing
        #[arg(long)]
        dry_run: bool,
    },
}

/// Run an import subcommand.
///
/// # Errors
///
/// Returns an error if the source store cannot be read or the registry written.
pub fn run(command: ImportCommand, root: Option<&Path>, quiet: bool) -> Result<()> {
    match command {
        ImportCommand::Trnscrb { from, dry_run } => {
            let path = match from {
                Some(path) => path,
                None => data_home()?.join("trnscrb/speakers/registry.json"),
            };
            import_trnscrb(&path, root, dry_run, quiet)
        }
        ImportCommand::Spqx { from, dry_run } => {
            let dir = match from {
                Some(path) => path,
                None => data_home()?.join("spqx/voices"),
            };
            import_spqx(&dir, root, dry_run, quiet)
        }
    }
}

/// The trnscrb on-disk speaker registry (read-only mirror of its format).
#[derive(Debug, Deserialize)]
struct TrnscrbRegistry {
    #[serde(default)]
    speakers: Vec<TrnscrbSpeaker>,
}

#[derive(Debug, Deserialize)]
struct TrnscrbSpeaker {
    id: String,
    name: String,
    #[serde(default)]
    aliases: Vec<String>,
    #[serde(default)]
    embedding_model: String,
    #[serde(default)]
    centroid: Vec<f32>,
    #[serde(default)]
    embeddings: Vec<Vec<f32>>,
    #[serde(default)]
    samples: Vec<PathBuf>,
}

fn import_trnscrb(path: &Path, root: Option<&Path>, dry_run: bool, quiet: bool) -> Result<()> {
    let body = std::fs::read_to_string(path)
        .with_context(|| format!("reading trnscrb registry {}", path.display()))?;
    let source: TrnscrbRegistry = serde_json::from_str(&body)
        .with_context(|| format!("parsing trnscrb registry {}", path.display()))?;

    let store = open_store(root)?;
    let paths = store.paths().clone();
    let existing = store.load()?;

    let mut planned = Vec::new();
    for speaker in &source.speakers {
        if existing.find(&speaker.id).is_some() {
            if !quiet {
                eprintln!("skip {}: already in the registry", speaker.id);
            }
            continue;
        }
        if let Some(other) = existing.id_for_name(&speaker.name) {
            if !quiet {
                eprintln!(
                    "skip {}: display name '{}' already belongs to {other} (link explicitly)",
                    speaker.id, speaker.name
                );
            }
            continue;
        }
        planned.push(speaker);
    }

    if dry_run {
        for speaker in &planned {
            println!(
                "would import {} ({}) with {} embedding(s)",
                speaker.id,
                speaker.name,
                speaker.embeddings.len()
            );
        }
        return Ok(());
    }

    let imported = store.edit(|registry| {
        let mut count = 0usize;
        for speaker in &planned {
            registry.identities.push(build_identity(speaker, &paths)?);
            count += 1;
        }
        Ok(count)
    })?;
    println!("imported {imported} identity(ies) from {}", path.display());
    Ok(())
}

/// Build a voqx identity from a trnscrb speaker, preserving its stable id so
/// existing transcript references remain valid.
fn build_identity(speaker: &TrnscrbSpeaker, paths: &RegistryPaths) -> Result<VoiceIdentity> {
    let mut identity = VoiceIdentity::new(speaker.id.clone(), speaker.name.clone());
    identity.aliases.clone_from(&speaker.aliases);

    let engine = if speaker.embedding_model.is_empty() {
        "speakrs-wespeaker".to_string()
    } else {
        speaker.embedding_model.clone()
    };

    let samples = copy_samples(speaker, paths, &engine)?;
    let mut profile = RecognitionProfile::new(
        engine,
        "imported-from-trnscrb",
        speaker.embeddings.clone(),
        samples,
    );
    // Preserve the source centroid verbatim rather than recomputing, so imported
    // matching behaves exactly as it did in trnscrb.
    if !speaker.centroid.is_empty() {
        profile.centroid.clone_from(&speaker.centroid);
    }
    identity.recognition_profiles.push(profile);
    Ok(identity)
}

fn copy_samples(
    speaker: &TrnscrbSpeaker,
    paths: &RegistryPaths,
    engine: &str,
) -> Result<Vec<String>> {
    let mut stored = Vec::new();
    let dir = paths
        .identity_dir(&speaker.id)
        .join("recognition")
        .join(engine);
    for (index, source) in speaker.samples.iter().enumerate() {
        if !source.exists() {
            continue;
        }
        let Ok(decoded) = audio::read_wav(source) else {
            continue;
        };
        let target = dir.join(format!("sample-{}.wav", index + 1));
        audio::write_wav(&target, &decoded)?;
        stored.push(relative_to_root(paths, &target)?);
    }
    Ok(stored)
}

/// Per-voice metadata written by `spqx voices add`.
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
struct SpqxVoiceManifest {
    language: String,
    note: String,
    reference_seconds: f64,
}

fn import_spqx(dir: &Path, root: Option<&Path>, dry_run: bool, quiet: bool) -> Result<()> {
    let store = open_store(root)?;
    let paths = store.paths().clone();
    let existing = store.load()?;

    let mut planned = Vec::new();
    let entries = std::fs::read_dir(dir)
        .with_context(|| format!("reading spqx voices directory {}", dir.display()))?;
    for entry in entries.filter_map(std::result::Result::ok) {
        let path = entry.path();
        let (Some(name), true) = (
            path.file_name().and_then(|n| n.to_str()),
            path.join("reference.wav").exists() && path.join("reference.txt").exists(),
        ) else {
            continue;
        };
        if let Some(other) = existing.id_for_name(name) {
            if !quiet {
                eprintln!("skip {name}: display name already belongs to {other}");
            }
            continue;
        }
        planned.push((name.to_string(), path));
    }
    planned.sort_by(|a, b| a.0.cmp(&b.0));

    if dry_run {
        for (name, path) in &planned {
            println!("would import voice '{name}' from {}", path.display());
        }
        return Ok(());
    }

    let imported = store.edit(|registry| {
        let mut count = 0usize;
        for (name, path) in &planned {
            import_one_spqx_voice(registry, &paths, name, path)?;
            count += 1;
        }
        Ok(count)
    })?;
    println!("imported {imported} voice(s) from {}", dir.display());
    Ok(())
}

fn import_one_spqx_voice(
    registry: &mut VoiceRegistry,
    paths: &RegistryPaths,
    name: &str,
    source_dir: &Path,
) -> Result<()> {
    let manifest: SpqxVoiceManifest = std::fs::read_to_string(source_dir.join("voice.toml"))
        .ok()
        .and_then(|body| toml::from_str(&body).ok())
        .unwrap_or_default();
    let transcript = std::fs::read_to_string(source_dir.join("reference.txt"))
        .with_context(|| format!("reading transcript in {}", source_dir.display()))?
        .trim()
        .to_string();
    let decoded = audio::read_wav(&source_dir.join("reference.wav"))?;

    let id = registry.create_identity(name);
    let dir = paths.identity_dir(&id).join("synthesis").join(SPQX_ENGINE);
    let audio_path = dir.join("ref-1.wav");
    let text_path = dir.join("ref-1.txt");
    audio::write_wav(&audio_path, &decoded)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    std::fs::write(&text_path, format!("{transcript}\n"))
        .with_context(|| format!("writing {}", text_path.display()))?;

    let duration = if manifest.reference_seconds > 0.0 {
        manifest.reference_seconds
    } else {
        (decoded.seconds() * 10.0).round() / 10.0
    };
    let language = match manifest.language.as_str() {
        "" | "auto" => None,
        other => Some(other.to_string()),
    };

    let mut profile = SynthesisProfile::new(SPQX_ENGINE);
    profile.references.push(SynthesisReference {
        id: "ref-1".to_string(),
        audio: relative_to_root(paths, &audio_path)?,
        transcript: relative_to_root(paths, &text_path)?,
        reference_language: language,
        default: true,
        tags: Vec::new(),
        description: manifest.note,
        duration_seconds: duration,
        sample_rate: decoded.sample_rate,
    });

    let identity = registry
        .find_mut(&id)
        .context("identity vanished during import")?;
    identity.synthesis_profiles.push(profile);
    identity.touch();
    Ok(())
}

fn data_home() -> Result<PathBuf> {
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME").map(PathBuf::from)
        && dir.is_absolute()
    {
        return Ok(dir);
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .context("unable to determine HOME")?;
    Ok(home.join(".local/share"))
}
