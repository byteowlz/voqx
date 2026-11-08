//! `voqx identities` — inspect and manage Voice Identities.

use std::path::Path;

use anyhow::{Context, Result, bail};
use clap::Subcommand;
use voqx_core::VoiceIdentity;

use super::{open_store, resolve, resolve_id};

/// Identity management subcommands.
#[derive(Debug, Subcommand)]
pub enum IdentitiesCommand {
    /// List all voice identities and their capabilities
    List,
    /// Show one identity by `voice_id`, display name, or alias
    Show {
        /// `voice_id`, display name, or alias
        key: String,
    },
    /// Rename an identity, keeping the old name as an alias
    Rename {
        /// `voice_id`, display name, or alias
        key: String,
        /// New display name
        name: String,
    },
    /// Remove an identity and (optionally) its assets
    Rm {
        /// `voice_id`, display name, or alias
        key: String,
        /// Also delete the identity's asset directory
        #[arg(long)]
        purge_assets: bool,
    },
}

/// Execute an identities subcommand.
///
/// # Errors
///
/// Returns an error if the registry cannot be read or written.
pub fn run(command: IdentitiesCommand, root: Option<&Path>, json: bool) -> Result<()> {
    let store = open_store(root)?;
    match command {
        IdentitiesCommand::List => list(&store, json),
        IdentitiesCommand::Show { key } => show(&store, &key, json),
        IdentitiesCommand::Rename { key, name } => rename(&store, &key, &name),
        IdentitiesCommand::Rm { key, purge_assets } => remove(&store, &key, purge_assets),
    }
}

fn list(store: &voqx_core::registry::RegistryStore, json: bool) -> Result<()> {
    let registry = store.load()?;
    if json {
        let rows: Vec<_> = registry
            .identities
            .iter()
            .map(|identity| {
                let caps = identity.capabilities();
                serde_json::json!({
                    "voice_id": identity.id,
                    "display_name": identity.display_name,
                    "aliases": identity.aliases,
                    "recognition": caps.recognition,
                    "synthesis": caps.synthesis,
                    "wake": caps.wake,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }

    if registry.identities.is_empty() {
        println!("No voice identities yet. Enroll one:");
        println!("  voqx enroll <name> --audio <file> --transcript-file <file> --recognition");
        return Ok(());
    }

    println!("Voice identities ({}):", registry.identities.len());
    for identity in &registry.identities {
        println!(
            "  {:<24} {:<20} [{}]",
            identity.id,
            identity.display_name,
            capability_summary(identity)
        );
    }
    Ok(())
}

fn capability_summary(identity: &VoiceIdentity) -> String {
    let caps = identity.capabilities();
    let mut parts = Vec::new();
    if caps.recognition {
        parts.push("recognition");
    }
    if caps.synthesis {
        parts.push("synthesis");
    }
    if caps.wake {
        parts.push("wake");
    }
    if parts.is_empty() {
        "no profiles".to_string()
    } else {
        parts.join(", ")
    }
}

fn show(store: &voqx_core::registry::RegistryStore, key: &str, json: bool) -> Result<()> {
    let registry = store.load()?;
    let identity = resolve(&registry, key)?;
    if json {
        println!("{}", serde_json::to_string_pretty(identity)?);
        return Ok(());
    }

    println!("voice_id:     {}", identity.id);
    println!("display_name: {}", identity.display_name);
    if !identity.aliases.is_empty() {
        println!("aliases:      {}", identity.aliases.join(", "));
    }
    if !identity.note.is_empty() {
        println!("note:         {}", identity.note);
    }
    println!("capabilities: {}", capability_summary(identity));

    for profile in &identity.recognition_profiles {
        println!(
            "  recognition  {} ({} samples, model {})",
            profile.engine,
            profile.embeddings.len(),
            if profile.model_version.is_empty() {
                "unknown"
            } else {
                &profile.model_version
            }
        );
    }
    for profile in &identity.synthesis_profiles {
        println!(
            "  synthesis    {} ({} reference(s))",
            profile.engine,
            profile.references.len()
        );
        for reference in &profile.references {
            let language = reference.reference_language.as_deref().unwrap_or("unknown");
            let marker = if reference.default { "*" } else { " " };
            println!(
                "    {marker} {:<16} {language:<8} {:.1}s {}",
                reference.id,
                reference.duration_seconds,
                reference.tags.join(",")
            );
        }
    }
    for profile in &identity.wake_profiles {
        println!(
            "  wake         {} '{}' ({})",
            profile.backend, profile.phrase, profile.id
        );
    }
    for source in &identity.sources {
        println!("  source       {} ({})", source.id, source.audio);
    }
    Ok(())
}

fn rename(store: &voqx_core::registry::RegistryStore, key: &str, name: &str) -> Result<()> {
    let renamed = store.edit(|registry| {
        let id = resolve_id(registry, key)?;
        registry.rename(&id, name)?;
        Ok(id)
    })?;
    println!("renamed {renamed} -> {name}");
    Ok(())
}

fn remove(store: &voqx_core::registry::RegistryStore, key: &str, purge_assets: bool) -> Result<()> {
    let paths = store.paths().clone();
    let removed = store.edit(|registry| {
        let id = resolve_id(registry, key)?;
        if !registry.remove(&id) {
            bail!("failed to remove {id}");
        }
        Ok(id)
    })?;

    if purge_assets {
        let dir = paths.identity_dir(&removed);
        if dir.exists() {
            std::fs::remove_dir_all(&dir)
                .with_context(|| format!("removing assets {}", dir.display()))?;
        }
    }
    println!("removed {removed}");
    Ok(())
}
