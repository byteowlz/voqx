//! voqx CLI: manage the shared Voice Identity registry.

mod audio;
mod capabilities;
#[cfg(test)]
mod preprocessing_tests;
mod voices;

use std::env;
use std::io::{self, IsTerminal};
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow};
use clap::{Args, CommandFactory, Parser, Subcommand, ValueEnum};
use clap_complete::Shell;
use env_logger::fmt::WriteStyle;
use log::{LevelFilter, debug, info};

use voqx_core::paths::write_default_config;
use voqx_core::{AppConfig, AppPaths, default_cache_dir, default_parallelism};

const APP_NAME: &str = env!("CARGO_PKG_NAME");

fn main() -> anyhow::Result<()> {
    try_main()
}

fn try_main() -> Result<()> {
    let cli = Cli::parse();
    // Discovery must work even with missing/invalid config and no models or registry.
    if matches!(cli.command, Command::Capabilities) {
        return capabilities::run(cli.common.yaml);
    }

    let ctx = RuntimeContext::new(cli.common.clone())?;
    ctx.init_logging()?;
    debug!("resolved paths: {:#?}", ctx.paths);

    let registry_root = cli.common.registry.clone();
    let json = cli.common.json;
    match cli.command {
        Command::Capabilities => capabilities::run(cli.common.yaml),
        Command::Run(cmd) => handle_run(&ctx, cmd),
        Command::Init(cmd) => handle_init(&ctx, cmd),
        Command::Config { command } => handle_config(&ctx, command),
        Command::Identities { command } => {
            voices::identities::run(command, registry_root.as_deref(), json)
        }
        Command::Enroll(args) => {
            voices::enroll::run(&args, registry_root.as_deref(), ctx.common.quiet)
        }
        Command::References { command } => {
            voices::enroll::run_references(command, registry_root.as_deref(), json)
        }
        Command::Import { command } => {
            voices::import::run(command, registry_root.as_deref(), ctx.common.quiet)
        }
        #[cfg(feature = "recognition")]
        Command::Identify {
            audio,
            mode,
            preprocessing,
        } => voices::identify::run(
            &audio,
            registry_root.as_deref(),
            &mode,
            preprocessing,
            voqx_core::Thresholds::default(),
            json,
        ),
        Command::Completions { shell } => {
            handle_completions(shell);
            Ok(())
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "A batteries-included Rust CLI template.",
    propagate_version = true
)]
struct Cli {
    #[command(flatten)]
    common: CommonOpts,
    #[command(subcommand)]
    command: Command,
}

/// Common CLI options shared across all subcommands.
#[derive(Debug, Clone, Args)]
pub struct CommonOpts {
    /// Override the config file path
    #[arg(long, value_name = "PATH", global = true)]
    pub config: Option<PathBuf>,
    /// Override the voice registry root (default: $XDG_DATA_HOME/voice-registry)
    #[arg(long, value_name = "PATH", global = true)]
    pub registry: Option<PathBuf>,
    /// Reduce output to only errors
    #[arg(short, long, action = clap::ArgAction::SetTrue, global = true)]
    pub quiet: bool,
    /// Increase logging verbosity (stackable)
    #[arg(short = 'v', long = "verbose", action = clap::ArgAction::Count, global = true)]
    pub verbose: u8,
    /// Enable debug logging (equivalent to -vv)
    #[arg(long, global = true)]
    pub debug: bool,
    /// Enable trace logging (overrides other levels)
    #[arg(long, global = true)]
    pub trace: bool,
    /// Output machine readable JSON
    #[arg(long, global = true, conflicts_with = "yaml")]
    pub json: bool,
    /// Output machine readable YAML
    #[arg(long, global = true)]
    pub yaml: bool,
    /// Disable ANSI colors in output
    #[arg(long = "no-color", global = true, conflicts_with = "color")]
    pub no_color: bool,
    /// Control color output (auto, always, never)
    #[arg(long, value_enum, default_value_t = ColorOption::Auto, global = true)]
    pub color: ColorOption,
    /// Do not change anything on disk
    #[arg(long = "dry-run", global = true)]
    pub dry_run: bool,
    /// Assume "yes" for interactive prompts
    #[arg(short = 'y', long = "yes", alias = "force", global = true)]
    pub assume_yes: bool,
    /// Never prompt for input; fail if confirmation would be required
    #[arg(long = "no-input", global = true)]
    pub no_input: bool,
    /// Maximum seconds to allow an operation to run
    #[arg(long = "timeout", value_name = "SECONDS", global = true)]
    pub timeout: Option<u64>,
    /// Override the degree of parallelism
    #[arg(long = "parallel", value_name = "N", global = true)]
    pub parallel: Option<usize>,
    /// Disable progress indicators
    #[arg(long = "no-progress", global = true)]
    pub no_progress: bool,
    /// Emit additional diagnostics for troubleshooting
    #[arg(long = "diagnostics", global = true)]
    pub diagnostics: bool,
}

/// Color output mode.
#[derive(Debug, Clone, Copy, ValueEnum)]
pub enum ColorOption {
    /// Detect terminal capabilities automatically.
    Auto,
    /// Always emit ANSI color codes.
    Always,
    /// Never emit ANSI color codes.
    Never,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Print versioned build capabilities without reading config, models or the registry
    Capabilities,
    /// Execute the CLI's primary behavior
    Run(RunCommand),
    /// Create config directories and default files
    Init(InitCommand),
    /// Inspect and manage configuration
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Inspect and manage voice identities
    Identities {
        #[command(subcommand)]
        command: voices::identities::IdentitiesCommand,
    },
    /// Enroll a recording as recognition and/or synthesis derivatives
    Enroll(voices::enroll::EnrollArgs),
    /// Inspect synthesis references
    References {
        #[command(subcommand)]
        command: voices::enroll::ReferencesCommand,
    },
    /// Import existing tool-local voice stores
    Import {
        #[command(subcommand)]
        command: voices::import::ImportCommand,
    },
    /// Identify a speaker in an audio file (recognition evidence only)
    #[cfg(feature = "recognition")]
    Identify {
        /// Audio file to identify
        audio: PathBuf,
        /// Execution mode (cpu, coreml, cuda)
        #[arg(long, default_value = "cpu")]
        mode: String,
        /// Recognition pooling; must match enrollment, scores remain uncalibrated
        #[arg(long, default_value = "uniform", value_name = "uniform|valid-v1")]
        preprocessing: voqx_core::RecognitionPreprocessing,
    },
    /// Generate shell completions
    Completions {
        #[arg(value_enum)]
        shell: Shell,
    },
}

#[derive(Debug, Clone, Args)]
struct RunCommand {
    /// Named task to execute
    #[arg(value_name = "TASK", default_value = "default")]
    task: String,
    /// Override the profile to run under
    #[arg(long, value_name = "PROFILE")]
    profile: Option<String>,
}

#[derive(Debug, Clone, Copy, Args)]
struct InitCommand {
    /// Recreate configuration even if it already exists
    #[arg(long = "force")]
    force: bool,
}

#[derive(Debug, Clone, Copy, Subcommand)]
enum ConfigCommand {
    /// Output the effective configuration
    Show,
    /// Print the resolved config file path
    Path,
    /// Print all resolved paths (config, data, state, cache)
    Paths,
    /// Print the JSON schema for the config file
    Schema,
    /// Regenerate the default configuration file
    Reset,
}

#[derive(Debug, Clone)]
struct RuntimeContext {
    common: CommonOpts,
    paths: AppPaths,
    config: AppConfig,
}

impl RuntimeContext {
    fn new(common: CommonOpts) -> Result<Self> {
        let paths = AppPaths::discover(common.config.as_deref())?;
        let config = AppConfig::load(&paths, common.dry_run)?;
        let paths = paths.apply_overrides(&config)?;
        let ctx = Self {
            common,
            paths,
            config,
        };
        ctx.ensure_directories()?;
        Ok(ctx)
    }

    fn init_logging(&self) -> Result<()> {
        if self.common.quiet {
            log::set_max_level(LevelFilter::Off);
            return Ok(());
        }

        let mut builder =
            env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"));

        builder.filter_level(self.effective_log_level());

        let force_color = matches!(self.common.color, ColorOption::Always)
            || env::var_os("FORCE_COLOR").is_some();
        let disable_color = self.common.no_color
            || matches!(self.common.color, ColorOption::Never)
            || env::var_os("NO_COLOR").is_some()
            || (!force_color && !io::stderr().is_terminal());

        if disable_color {
            builder.write_style(WriteStyle::Never);
        } else if force_color {
            builder.write_style(WriteStyle::Always);
        } else {
            builder.write_style(WriteStyle::Auto);
        }

        if self.common.diagnostics {
            builder.format_timestamp_millis();
            builder.format_module_path(true);
            builder.format_target(true);
        }

        builder.try_init().or_else(|err| {
            if self.common.verbose > 0 {
                eprintln!("logger already initialized: {err}");
            }
            Ok(())
        })
    }

    const fn effective_log_level(&self) -> LevelFilter {
        if self.common.trace {
            LevelFilter::Trace
        } else if self.common.debug {
            LevelFilter::Debug
        } else {
            match self.common.verbose {
                0 => LevelFilter::Info,
                1 => LevelFilter::Debug,
                _ => LevelFilter::Trace,
            }
        }
    }

    fn ensure_directories(&self) -> Result<()> {
        if self.common.dry_run {
            self.paths.log_dry_run();
            return Ok(());
        }
        self.paths.ensure_directories()
    }
}

fn handle_run(ctx: &RuntimeContext, cmd: RunCommand) -> Result<()> {
    let effective = ctx.config.clone().with_profile_override(cmd.profile);
    let output = if ctx.common.json {
        serde_json::to_string_pretty(&effective).context("serializing run output to JSON")?
    } else if ctx.common.yaml {
        serde_yaml::to_string(&effective).context("serializing run output to YAML")?
    } else {
        format!(
            "Running task '{}' with profile '{}' (parallelism: {})",
            cmd.task,
            effective.profile,
            effective
                .runtime
                .parallelism
                .unwrap_or_else(default_parallelism)
        )
    };

    println!("{output}");
    Ok(())
}

fn handle_init(ctx: &RuntimeContext, cmd: InitCommand) -> Result<()> {
    if ctx.paths.config_file.exists() && !(cmd.force || ctx.common.assume_yes) {
        return Err(anyhow!(
            "config already exists at {} (use --force to overwrite)",
            ctx.paths.config_file.display()
        ));
    }

    if ctx.common.dry_run {
        info!(
            "dry-run: would write default config to {}",
            ctx.paths.config_file.display()
        );
        return Ok(());
    }

    write_default_config(&ctx.paths.config_file)
}

fn handle_config(ctx: &RuntimeContext, command: ConfigCommand) -> Result<()> {
    match command {
        ConfigCommand::Show => {
            if ctx.common.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&ctx.config)
                        .context("serializing config to JSON")?
                );
            } else if ctx.common.yaml {
                println!(
                    "{}",
                    serde_yaml::to_string(&ctx.config).context("serializing config to YAML")?
                );
            } else {
                println!("{:#?}", ctx.config);
            }
            Ok(())
        }
        ConfigCommand::Path => {
            println!("{}", ctx.paths.config_file.display());
            Ok(())
        }
        ConfigCommand::Paths => {
            let cache_dir = default_cache_dir()?;
            if ctx.common.json {
                let paths = serde_json::json!({
                    "config": ctx.paths.config_file,
                    "data": ctx.paths.data_dir,
                    "state": ctx.paths.state_dir,
                    "cache": cache_dir,
                });
                println!(
                    "{}",
                    serde_json::to_string_pretty(&paths).context("serializing paths to JSON")?
                );
            } else if ctx.common.yaml {
                let paths = serde_json::json!({
                    "config": ctx.paths.config_file,
                    "data": ctx.paths.data_dir,
                    "state": ctx.paths.state_dir,
                    "cache": cache_dir,
                });
                println!(
                    "{}",
                    serde_yaml::to_string(&paths).context("serializing paths to YAML")?
                );
            } else {
                println!("config: {}", ctx.paths.config_file.display());
                println!("data:   {}", ctx.paths.data_dir.display());
                println!("state:  {}", ctx.paths.state_dir.display());
                println!("cache:  {}", cache_dir.display());
            }
            Ok(())
        }
        ConfigCommand::Schema => {
            println!("{}", include_str!("../../../examples/config.schema.json"));
            Ok(())
        }
        ConfigCommand::Reset => {
            if ctx.common.dry_run {
                info!(
                    "dry-run: would reset config at {}",
                    ctx.paths.config_file.display()
                );
                return Ok(());
            }
            write_default_config(&ctx.paths.config_file)
        }
    }
}

fn handle_completions(shell: Shell) {
    let mut cmd = Cli::command();
    clap_complete::generate(shell, &mut cmd, APP_NAME, &mut io::stdout());
}
