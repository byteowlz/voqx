//! Subprocess proof: discovery bypasses config initialization and changes no data.

use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Result, ensure};

#[path = "registry_discovery/enrollment.rs"]
mod enrollment;
#[path = "registry_discovery/migration.rs"]
mod migration;

#[derive(Debug)]
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Result<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "voqx-cli-discovery-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&root)?;
        std::fs::write(root.join("invalid.toml"), "not valid [toml")?;
        Ok(Self(root))
    }
    fn invoke(&self, args: &[&str]) -> Result<Output> {
        Ok(Command::new(env!("CARGO_BIN_EXE_voqx"))
            .args(args)
            .arg("--config")
            .arg(self.0.join("invalid.toml"))
            .env("HOME", &self.0)
            .env("XDG_CONFIG_HOME", self.0.join("config"))
            .env("XDG_STATE_HOME", self.0.join("state"))
            .env("XDG_DATA_HOME", self.0.join("data"))
            .env("XDG_CACHE_HOME", self.0.join("cache"))
            .env("HF_HOME", self.0.join("hf"))
            .env("HF_HUB_OFFLINE", "1")
            .output()?)
    }
    fn run(&self, args: &[&str]) -> Result<Output> {
        let output = self.invoke(args)?;
        ensure!(
            output.status.success(),
            "discovery failed: {}",
            output.status
        );
        ensure!(output.stderr.is_empty(), "discovery emitted diagnostics");
        Ok(output)
    }
    fn unchanged(&self) -> Result<()> {
        ensure!(
            std::fs::read_dir(&self.0)?.count() == 1,
            "discovery created user state"
        );
        ensure!(
            std::fs::read_to_string(self.0.join("invalid.toml"))? == "not valid [toml",
            "config changed"
        );
        Ok(())
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn missing_roots_and_invalid_config_are_structured_without_side_effects() -> Result<()> {
    let dir = Scratch::new()?;
    let output = dir.run(&["registry", "status", "--json"])?;
    let status: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    ensure!(status["protocol"] == "voqx-registry-discovery/1");
    ensure!(status["selection"] == "legacy_default");
    ensure!(status["effective"]["state"] == "missing");
    ensure!(status["proposed"]["state"] == "missing");
    ensure!(
        status["effective"]["root"] == dir.0.join("data/voice-registry").to_string_lossy().as_ref()
    );
    ensure!(
        status["proposed"]["root"]
            == dir
                .0
                .join("data/byteowlz/identity/voices")
                .to_string_lossy()
                .as_ref()
    );
    ensure!(status["ready_to_filter"] == false);
    let output = dir.run(&["registry", "migration-plan", "--json"])?;
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    ensure!(plan["state"] == "blocked");
    ensure!(plan["blockers"] == serde_json::json!([{"reason":"source_not_ready"}]));
    ensure!(plan["apply_supported"] == false);
    dir.unchanged()
}

#[test]
fn yaml_and_schemas_need_no_user_state() -> Result<()> {
    let dir = Scratch::new()?;
    let output = dir.run(&["registry", "status", "--yaml"])?;
    let status: serde_yaml::Value = serde_yaml::from_slice(&output.stdout)?;
    ensure!(status["protocol"] == "voqx-registry-discovery/1");
    for kind in [
        "status",
        "migration-plan",
        "readiness",
        "enrollment",
        "migration",
    ] {
        let output = dir.run(&["registry", "schema", kind])?;
        let schema: serde_json::Value = serde_json::from_slice(&output.stdout)?;
        ensure!(schema["$schema"].is_string());
        ensure!(schema["properties"]["protocol"].is_object());
    }
    dir.unchanged()
}

#[test]
fn explicit_override_does_not_require_home_or_read_default_roots() -> Result<()> {
    let dir = Scratch::new()?;
    let output = Command::new(env!("CARGO_BIN_EXE_voqx"))
        .args(["registry", "status", "--json", "--registry"])
        .arg(dir.0.join("explicit"))
        .env_remove("HOME")
        .env_remove("XDG_DATA_HOME")
        .env_remove("APPDATA")
        .output()?;
    ensure!(output.status.success());
    let status: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    ensure!(status["selection"] == "explicit");
    ensure!(status["proposed"].is_null());
    ensure!(status["effective"]["root"] == dir.0.join("explicit").to_string_lossy().as_ref());
    dir.unchanged()
}

#[test]
fn planner_reports_hashes_and_conflicts_but_never_copies() -> Result<()> {
    let dir = Scratch::new()?;
    let source = dir.0.join("source");
    let target = dir.0.join("target");
    std::fs::create_dir(&source)?;
    let bytes = b"{\"version\":1,\"identities\":[]}";
    std::fs::write(source.join("registry.json"), bytes)?;
    let args = [
        "registry",
        "migration-plan",
        "--json",
        "--registry",
        source
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("invalid fixture path"))?,
        "--destination",
        target
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("invalid fixture path"))?,
    ];
    let output = dir.run(&args)?;
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    ensure!(plan["state"] == "copy_required");
    ensure!(
        plan["entries"][0]["source"]["sha256"]
            .as_str()
            .is_some_and(|hash| hash.len() == 64)
    );
    ensure!(!target.exists());
    std::fs::create_dir(&target)?;
    std::fs::write(target.join("registry.json"), b"different")?;
    let output = dir.run(&args)?;
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    ensure!(plan["state"] == "blocked");
    ensure!(plan["entries"][0]["state"] == "conflict");
    ensure!(std::fs::read(source.join("registry.json"))? == bytes);
    ensure!(std::fs::read(target.join("registry.json"))? == b"different");
    ensure!(
        std::fs::read_dir(&dir.0)?.count() == 3,
        "unexpected initialized state"
    );
    Ok(())
}
