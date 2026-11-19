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
use voqx_core::validate_vectors;
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

/// trnscrb v1 records only this engine name, with no checkpoint, frontend or
/// hash, so imported profiles carry an explicitly unverified model version.
const TRNSCRB_ENGINE: &str = "speakrs-wespeaker";
/// trnscrb v1 `speakrs-wespeaker` vectors are 256-dimensional.
const TRNSCRB_DIMENSIONS: usize = 256;
/// Provenance marker: the source names an engine but no exact model revision.
const TRNSCRB_MODEL_VERSION: &str = "unverified-trnscrb-v1";
/// Tag recording which store version the vectors came from.
const TRNSCRB_PROVENANCE_TAG: &str = "provenance:trnscrb-registry-v1";

/// Why exact-space selection cannot use an imported profile.
const EXACT_SPACE_NOTE: &str = "imported profiles use model_version \
'unverified-trnscrb-v1' because trnscrb v1 stores no encoder/version/hash; they \
stay out of exact-space matching until verified or re-embedded";

/// Per-speaker sample accounting; skipped audio never invalidates the vectors.
#[derive(Debug, Default, Clone, Copy)]
struct SampleStats {
    copied: usize,
    missing: usize,
    unreadable: usize,
}

impl SampleStats {
    const fn add(&mut self, other: Self) {
        self.copied += other.copied;
        self.missing += other.missing;
        self.unreadable += other.unreadable;
    }

    fn summary(self) -> String {
        format!(
            "{} copied, {} missing, {} unreadable",
            self.copied, self.missing, self.unreadable
        )
    }
}

/// One validated speaker selected for import.
#[derive(Debug)]
struct PlannedSpeaker<'a> {
    speaker: &'a TrnscrbSpeaker,
    engine: String,
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
    let mut refused = Vec::new();
    let mut seen_ids = std::collections::HashSet::new();
    let mut seen_names = std::collections::HashSet::new();
    for speaker in &source.speakers {
        if !seen_ids.insert(speaker.id.as_str()) {
            refused.push((
                speaker,
                format!("duplicate id {} in the source", speaker.id),
            ));
            continue;
        }
        if existing.find(&speaker.id).is_some() {
            refused.push((speaker, "already in the registry".to_string()));
            continue;
        }
        if let Some(other) = existing.id_for_name(&speaker.name) {
            refused.push((
                speaker,
                format!(
                    "display name '{}' already belongs to {other} (link explicitly)",
                    speaker.name
                ),
            ));
            continue;
        }
        if !seen_names.insert(speaker.name.as_str()) {
            refused.push((
                speaker,
                format!(
                    "display name '{}' appears twice in the source (link explicitly)",
                    speaker.name
                ),
            ));
            continue;
        }
        match validate_speaker(speaker) {
            Ok(engine) => planned.push(PlannedSpeaker { speaker, engine }),
            Err(reason) => refused.push((speaker, reason)),
        }
    }

    if dry_run {
        for planned in &planned {
            let stats = inspect_samples(planned.speaker);
            println!(
                "would import {} ({}): {} embedding(s), samples {}",
                planned.speaker.id,
                planned.speaker.name,
                planned.speaker.embeddings.len(),
                stats.summary()
            );
        }
        report_refusals(&refused, quiet);
        println!(
            "would import {} identity(ies), refuse {}, from {}",
            planned.len(),
            refused.len(),
            path.display()
        );
        report_note(quiet);
        return Ok(());
    }

    let mut imported: Vec<(String, String, String, usize, SampleStats)> = Vec::new();
    let mut totals = SampleStats::default();
    store.edit(|registry| {
        for planned in &planned {
            let (identity, stats) = build_identity(planned, &paths)?;
            imported.push((
                identity.id.clone(),
                identity.display_name.clone(),
                planned.engine.clone(),
                planned.speaker.embeddings.len(),
                stats,
            ));
            totals.add(stats);
            registry.identities.push(identity);
        }
        Ok(())
    })?;

    if !quiet {
        for (id, name, engine, embeddings, stats) in &imported {
            println!(
                "imported {id} ({name}): {embeddings} embedding(s) on {engine}, samples {}",
                stats.summary()
            );
        }
        report_refusals(&refused, quiet);
    }
    println!(
        "imported {} identity(ies), refused {}, from {}",
        imported.len(),
        refused.len(),
        path.display()
    );
    println!("samples: {}", totals.summary());
    report_note(quiet);
    Ok(())
}

/// Check the vectors and the sample-independent invariants for one speaker.
///
/// Returns the engine to record, or the refusal reason. Normalization cannot
/// establish which space the vectors belong to, so a known trnscrb engine also
/// pins the expected dimension.
fn validate_speaker(speaker: &TrnscrbSpeaker) -> Result<String, String> {
    let engine = if speaker.embedding_model.is_empty() {
        TRNSCRB_ENGINE.to_string()
    } else {
        speaker.embedding_model.clone()
    };
    if engine == TRNSCRB_ENGINE {
        for (index, embedding) in speaker.embeddings.iter().enumerate() {
            if embedding.len() != TRNSCRB_DIMENSIONS {
                return Err(format!(
                    "reference vector {index} has dimension {}, expected {TRNSCRB_DIMENSIONS} \
for {TRNSCRB_ENGINE}",
                    embedding.len()
                ));
            }
        }
        if !speaker.centroid.is_empty() && speaker.centroid.len() != TRNSCRB_DIMENSIONS {
            return Err(format!(
                "centroid has dimension {}, expected {TRNSCRB_DIMENSIONS} for {TRNSCRB_ENGINE}",
                speaker.centroid.len()
            ));
        }
    }
    validate_vectors(&speaker.embeddings, &speaker.centroid).map_err(|error| error.to_string())?;
    Ok(engine)
}

/// Read-only sample inspection for `--dry-run`.
fn inspect_samples(speaker: &TrnscrbSpeaker) -> SampleStats {
    let mut stats = SampleStats::default();
    for source in &speaker.samples {
        if !source.exists() {
            stats.missing += 1;
        } else if audio::read_wav(source).is_err() {
            stats.unreadable += 1;
        } else {
            stats.copied += 1;
        }
    }
    stats
}

fn report_refusals(refused: &[(&TrnscrbSpeaker, String)], quiet: bool) {
    if quiet {
        return;
    }
    for (speaker, reason) in refused {
        println!("refused {} ({}): {reason}", speaker.id, speaker.name);
    }
}

fn report_note(quiet: bool) {
    if !quiet {
        println!("note: {EXACT_SPACE_NOTE}");
    }
}

/// Build a voqx identity from a trnscrb speaker, preserving its stable id so
/// existing transcript references remain valid.
fn build_identity(
    planned: &PlannedSpeaker<'_>,
    paths: &RegistryPaths,
) -> Result<(VoiceIdentity, SampleStats)> {
    let speaker = planned.speaker;
    let mut identity = VoiceIdentity::new(speaker.id.clone(), speaker.name.clone());
    identity.aliases.clone_from(&speaker.aliases);

    let (samples, stats) = copy_samples(speaker, paths, &planned.engine)?;
    let mut profile = RecognitionProfile::new(
        planned.engine.clone(),
        TRNSCRB_MODEL_VERSION,
        speaker.embeddings.clone(),
        samples,
    );
    // Preserve the source centroid verbatim rather than recomputing, so imported
    // matching behaves exactly as it did in trnscrb. It was validated above.
    if !speaker.centroid.is_empty() {
        profile.centroid.clone_from(&speaker.centroid);
    }
    profile.tags.push(TRNSCRB_PROVENANCE_TAG.to_string());
    if !speaker.embedding_model.is_empty() {
        profile.tags.push(format!(
            "source-embedding-model:{}",
            speaker.embedding_model
        ));
    }
    identity.recognition_profiles.push(profile);
    Ok((identity, stats))
}

fn copy_samples(
    speaker: &TrnscrbSpeaker,
    paths: &RegistryPaths,
    engine: &str,
) -> Result<(Vec<String>, SampleStats)> {
    let mut stored = Vec::new();
    let mut stats = SampleStats::default();
    let dir = paths
        .identity_dir(&speaker.id)
        .join("recognition")
        .join(engine);
    for (index, source) in speaker.samples.iter().enumerate() {
        if !source.exists() {
            stats.missing += 1;
            continue;
        }
        let Ok(decoded) = audio::read_wav(source) else {
            stats.unreadable += 1;
            continue;
        };
        let target = dir.join(format!("sample-{}.wav", index + 1));
        audio::write_wav(&target, &decoded)?;
        stored.push(relative_to_root(paths, &target)?);
        stats.copied += 1;
    }
    Ok((stored, stats))
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
