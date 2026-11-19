//! Registry discovery and explicit migration before application/config initialization.

use std::path::Path;

mod migrate;

use anyhow::Result;
use clap::{Subcommand, ValueEnum};
use voqx_core::RecognitionPreprocessing;
use voqx_core::registry_discovery::{MigrationPlan, RecognitionReadiness, RegistryDiscovery};
use voqx_core::registry_paths::{default_registry_root, proposed_registry_root};

use crate::output::print as print_response;

/// Registry discovery and explicitly confirmed migration commands.
#[derive(Debug, Subcommand)]
pub enum RegistryCommand {
    /// Inspect the effective and proposed roots without switching defaults
    Status,
    /// Inspect selected profile metadata; native runtime/model/calibration stay unchecked
    Readiness {
        /// Recognition contract to inspect without loading its encoder
        #[arg(long, default_value = "uniform", value_name = "uniform|valid-v1")]
        preprocessing: RecognitionPreprocessing,
    },
    /// Compare exact source/destination bytes without copying or rewriting IDs
    MigrationPlan {
        /// Proposed destination (default: shared identity/voices root)
        #[arg(long, value_name = "PATH")]
        destination: Option<std::path::PathBuf>,
    },
    /// Apply/resume/abort a reviewed plan; writes require --yes (or validate with --dry-run)
    Migrate(migrate::MigrateArgs),
    /// Print a JSON schema without reading any registry or config
    Schema {
        /// Response contract
        #[arg(value_enum)]
        kind: SchemaKind,
    },
}

/// Discovery response schemas.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum SchemaKind {
    /// voqx-registry-discovery/1
    Status,
    /// voqx-registry-migration-plan/1
    MigrationPlan,
    /// voqx-recognition-readiness/1
    Readiness,
    /// voqx-enrollment/1
    Enrollment,
    /// voqx-registry-migration/1
    Migration,
}

#[derive(Debug, serde::Serialize, schemars::JsonSchema)]
struct ReadinessReport {
    recognition_compiled: bool,
    #[serde(flatten)]
    registry: RecognitionReadiness,
}

/// Print a versioned JSON/YAML response without initializing application state.
///
/// # Errors
/// Returns an error if default roots cannot be resolved or serialization fails.
pub fn run(command: &RegistryCommand, common: &crate::CommonOpts) -> Result<()> {
    let explicit: Option<&Path> = common.registry.as_deref();
    let yaml = common.yaml;
    match command {
        RegistryCommand::Migrate(args) => migrate::run(args, common),
        RegistryCommand::Status => {
            let discovery = if let Some(root) = explicit {
                // An override must also work with no HOME or XDG variables.
                RegistryDiscovery::inspect(
                    root.to_path_buf(),
                    root.to_path_buf(),
                    Some(root.to_path_buf()),
                )
            } else {
                RegistryDiscovery::inspect(
                    default_registry_root()?,
                    proposed_registry_root()?,
                    None,
                )
            };
            print_response(&discovery, yaml)
        }
        RegistryCommand::Readiness { preprocessing } => {
            let root =
                explicit.map_or_else(default_registry_root, |root| Ok(root.to_path_buf()))?;
            print_response(
                &ReadinessReport {
                    recognition_compiled: cfg!(feature = "recognition"),
                    registry: RecognitionReadiness::inspect(root, *preprocessing),
                },
                yaml,
            )
        }
        RegistryCommand::MigrationPlan { destination } => {
            let source =
                explicit.map_or_else(default_registry_root, |root| Ok(root.to_path_buf()))?;
            let destination = destination
                .clone()
                .map_or_else(proposed_registry_root, Ok)?;
            print_response(&MigrationPlan::inspect(source, destination), yaml)
        }
        RegistryCommand::Schema { kind } => {
            let schema = match kind {
                SchemaKind::Status => schemars::schema_for!(RegistryDiscovery),
                SchemaKind::MigrationPlan => schemars::schema_for!(MigrationPlan),
                SchemaKind::Readiness => schemars::schema_for!(ReadinessReport),
                SchemaKind::Migration => {
                    schemars::schema_for!(voqx_core::registry_discovery::MigrationReceipt)
                }
                SchemaKind::Enrollment => {
                    schemars::schema_for!(crate::voices::enroll::EnrollmentReport)
                }
            };
            println!("{}", serde_json::to_string_pretty(&schema)?);
            Ok(())
        }
    }
}
