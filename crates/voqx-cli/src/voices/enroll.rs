//! `voqx enroll` and `voqx references` — attach recognition and synthesis
//! derivatives to a Voice Identity from one reviewed source recording.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use voqx_core::identity::{RecognitionProfile, SourceRecording, SynthesisProfile, SynthesisReference};
use voqx_core::registry_paths::RegistryPaths;
use voqx_core::{VoiceIdentity, VoiceRegistry};

use crate::audio::{self, Audio, RECOGNITION_SAMPLE_RATE, SYNTHESIS_SAMPLE_RATE};

use super::{open_store, read_transcript, relative_to_root, resolve_id};

/// Default synthesis engine identifier.
pub const DEFAULT_SYNTHESIS_ENGINE: &str = "spqx-qwen3";

/// Arguments for `voqx enroll`.
#[derive(Debug, Args)]
pub struct EnrollArgs {
    /// Display name for a new identity (ignored when --id is given)
    pub name: String,
    /// Attach to an existing identity instead of creating one
    #[arg(long, value_name = "VOICE_ID")]
    pub id: Option<String>,
    /// Source recording (WAV)
    #[arg(long, value_name = "PATH")]
    pub audio: PathBuf,
    /// Exact transcript of the recording
    #[arg(long, value_name = "TEXT")]
    pub transcript: Option<String>,
    /// File containing the exact transcript
    #[arg(long, value_name = "PATH")]
    pub transcript_file: Option<PathBuf>,
    /// Language spoken in the recording (e.g. de-DE)
    #[arg(long, value_name = "TAG")]
    pub language: Option<String>,
    /// Derive a recognition profile ("who spoke")
    #[arg(long)]
    pub recognition: bool,
    /// Derive a synthesis reference for this engine ("speak as them")
    #[arg(long, value_name = "ENGINE", num_args = 0..=1, default_missing_value = DEFAULT_SYNTHESIS_ENGINE)]
    pub synthesis: Option<String>,
    /// Namespaced tag for the synthesis reference (repeatable), e.g. emotion:happy
    #[arg(long = "tag", value_name = "TAG")]
    pub tags: Vec<String>,
    /// Human description of the delivery
    #[arg(long)]
    pub description: Option<String>,
    /// Mark the synthesis reference as the profile default
    #[arg(long)]
    pub default_reference: bool,
    /// Execution mode for embedding (cpu, coreml, cuda)
    #[arg(long, default_value = "cpu")]
    pub mode: String,
}

/// Run `voqx enroll`.
///
/// # Errors
///
/// Returns an error if the audio cannot be read, a derivative cannot be
/// written, or the registry cannot be updated.
pub fn run(args: &EnrollArgs, root: Option<&Path>, quiet: bool) -> Result<()> {
    if args.recognition && !cfg!(feature = "recognition") {
        bail!(
            "this build has no recognition support; rebuild with --features recognition to use --recognition"
        );
    }
    let source = audio::read_wav(&args.audio)?;
    if !quiet {
        for warning in audio::hygiene_warnings(&source) {
            eprintln!("warning: {warning}");
        }
    }

    let transcript = if args.synthesis.is_some() {
        Some(read_transcript(
            args.transcript.as_deref(),
            args.transcript_file.as_deref(),
        )?)
    } else {
        read_transcript(args.transcript.as_deref(), args.transcript_file.as_deref()).ok()
    };

    let embedding = derive_embedding(args, &source)?;

    let store = open_store(root)?;
    let paths = store.paths().clone();
    let voice_id = store.edit(|registry| {
        let id = resolve_or_create(registry, args)?;
        let identity = registry
            .find_mut(&id)
            .context("identity vanished during enrollment")?;
        attach_source(identity, &paths, args, &source, transcript.as_deref())?;
        if let Some(vector) = embedding.clone() {
            attach_recognition(identity, &paths, &source, vector)?;
        }
        if let Some(engine) = args.synthesis.as_deref() {
            let text = transcript
                .as_deref()
                .context("synthesis requires an exact transcript")?;
            attach_synthesis(identity, &paths, args, &source, engine, text)?;
        }
        identity.touch();
        Ok(id)
    })?;

    println!("enrolled {voice_id}");
    Ok(())
}

#[cfg(feature = "recognition")]
fn derive_embedding(args: &EnrollArgs, source: &Audio) -> Result<Option<Vec<f32>>> {
    if !args.recognition {
        return Ok(None);
    }
    let mono16 = source.to_rate(RECOGNITION_SAMPLE_RATE)?;
    let mut embedder = voqx_recognition::Embedder::new(&args.mode, None)?;
    Ok(Some(embedder.embed(&mono16.samples)?))
}

#[cfg(not(feature = "recognition"))]
fn derive_embedding(_args: &EnrollArgs, _source: &Audio) -> Result<Option<Vec<f32>>> {
    Ok(None)
}

fn resolve_or_create(registry: &mut VoiceRegistry, args: &EnrollArgs) -> Result<String> {
    if let Some(key) = args.id.as_deref() {
        return resolve_id(registry, key);
    }
    if let Some(id) = registry.id_for_name(&args.name).map(ToString::to_string) {
        return Ok(id);
    }
    Ok(registry.create_identity(&args.name))
}

fn attach_source(
    identity: &mut VoiceIdentity,
    paths: &RegistryPaths,
    args: &EnrollArgs,
    source: &Audio,
    transcript: Option<&str>,
) -> Result<()> {
    let index = identity.sources.len() + 1;
    let id = format!("src-{index}");
    let dir = paths.identity_dir(&identity.id).join("sources");
    let audio_path = dir.join(format!("{id}.wav"));
    audio::write_wav(&audio_path, source)?;

    let transcript_rel = match transcript {
        Some(text) => {
            let path = dir.join(format!("{id}.txt"));
            write_text(&path, text)?;
            Some(relative_to_root(paths, &path)?)
        }
        None => None,
    };

    identity.sources.push(SourceRecording {
        id,
        audio: relative_to_root(paths, &audio_path)?,
        transcript: transcript_rel,
        language: args.language.clone(),
        note: args.description.clone().unwrap_or_default(),
        created_at_unix: now_unix(),
    });
    Ok(())
}

fn attach_recognition(
    identity: &mut VoiceIdentity,
    paths: &RegistryPaths,
    source: &Audio,
    embedding: Vec<f32>,
) -> Result<()> {
    let engine = recognition_engine();
    let mono16 = source.to_rate(RECOGNITION_SAMPLE_RATE)?;
    let existing = identity
        .recognition_profiles
        .iter()
        .position(|p| p.engine == engine);
    let index = existing.map_or(0, |i| identity.recognition_profiles[i].embeddings.len()) + 1;
    let sample_path = paths
        .identity_dir(&identity.id)
        .join("recognition")
        .join(&engine)
        .join(format!("sample-{index}.wav"));
    audio::write_wav(&sample_path, &mono16)?;
    let sample_rel = relative_to_root(paths, &sample_path)?;

    match existing {
        Some(i) => identity.recognition_profiles[i].add_samples(vec![embedding], vec![sample_rel]),
        None => identity.recognition_profiles.push(RecognitionProfile::new(
            engine,
            recognition_model_version(),
            vec![embedding],
            vec![sample_rel],
        )),
    }
    Ok(())
}

fn attach_synthesis(
    identity: &mut VoiceIdentity,
    paths: &RegistryPaths,
    args: &EnrollArgs,
    source: &Audio,
    engine: &str,
    transcript: &str,
) -> Result<()> {
    let resampled = source.to_rate(SYNTHESIS_SAMPLE_RATE)?;
    let position = identity
        .synthesis_profiles
        .iter()
        .position(|p| p.engine == engine)
        .unwrap_or_else(|| {
            identity.synthesis_profiles.push(SynthesisProfile::new(engine));
            identity.synthesis_profiles.len() - 1
        });

    let profile = &mut identity.synthesis_profiles[position];
    let reference_id = format!("ref-{}", profile.references.len() + 1);
    let dir = paths.identity_dir(&identity.id).join("synthesis").join(engine);
    let audio_path = dir.join(format!("{reference_id}.wav"));
    let text_path = dir.join(format!("{reference_id}.txt"));
    audio::write_wav(&audio_path, &resampled)?;
    write_text(&text_path, transcript)?;

    let is_default = args.default_reference || profile.references.is_empty();
    if is_default {
        for reference in &mut profile.references {
            reference.default = false;
        }
    }
    profile.references.push(SynthesisReference {
        id: reference_id,
        audio: relative_to_root(paths, &audio_path)?,
        transcript: relative_to_root(paths, &text_path)?,
        reference_language: args.language.clone(),
        default: is_default,
        tags: args.tags.clone(),
        description: args.description.clone().unwrap_or_default(),
        duration_seconds: (resampled.seconds() * 10.0).round() / 10.0,
        sample_rate: resampled.sample_rate,
    });
    profile.updated_at_unix = now_unix();
    Ok(())
}

/// `voqx references` subcommands.
#[derive(Debug, Subcommand)]
pub enum ReferencesCommand {
    /// List synthesis references for an identity
    List {
        /// `voice_id`, display name, or alias
        key: String,
    },
}

/// Run a `voqx references` subcommand.
///
/// # Errors
///
/// Returns an error if the registry cannot be read or the identity is unknown.
pub fn run_references(command: ReferencesCommand, root: Option<&Path>, json: bool) -> Result<()> {
    let store = open_store(root)?;
    let registry = store.load()?;
    match command {
        ReferencesCommand::List { key } => {
            let identity = super::resolve(&registry, &key)?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&identity.synthesis_profiles)?
                );
                return Ok(());
            }
            for profile in &identity.synthesis_profiles {
                println!("{} ({} references)", profile.engine, profile.references.len());
                for reference in &profile.references {
                    println!(
                        "  {}{:<14} {:<8} {:.1}s  {}",
                        if reference.default { "*" } else { " " },
                        reference.id,
                        reference.reference_language.as_deref().unwrap_or("-"),
                        reference.duration_seconds,
                        reference.tags.join(",")
                    );
                }
            }
            Ok(())
        }
    }
}

fn write_text(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    std::fs::write(path, format!("{text}\n"))
        .with_context(|| format!("writing {}", path.display()))
}

fn now_unix() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

#[cfg(feature = "recognition")]
fn recognition_engine() -> String {
    voqx_recognition::EMBEDDING_MODEL_ID.to_string()
}

#[cfg(not(feature = "recognition"))]
fn recognition_engine() -> String {
    "speakrs-wespeaker".to_string()
}

fn recognition_model_version() -> String {
    "speakrs-0.4".to_string()
}
