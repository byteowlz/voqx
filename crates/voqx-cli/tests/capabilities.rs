//! Capability discovery must not prepare models or initialize user state.

use anyhow::Result;
use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[derive(Debug)]
struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn capabilities_ignore_config_and_leave_model_registry_and_state_untouched() -> Result<()> {
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let dir = Scratch(
        std::env::temp_dir().join(format!("voqx-capabilities-{}-{nonce}", std::process::id())),
    );
    std::fs::create_dir(&dir.0)?;
    let invalid_config = dir.0.join("invalid.toml");
    std::fs::write(&invalid_config, "not a valid [toml file")?;
    let output = Command::new(env!("CARGO_BIN_EXE_voqx"))
        .args(["capabilities", "--json", "--config"])
        .arg(&invalid_config)
        .arg("--registry")
        .arg(dir.0.join("registry"))
        .env("HOME", &dir.0)
        .env("XDG_CONFIG_HOME", dir.0.join("config"))
        .env("XDG_STATE_HOME", dir.0.join("state"))
        .env("XDG_DATA_HOME", dir.0.join("data"))
        .env("XDG_CACHE_HOME", dir.0.join("cache"))
        .env("HF_HOME", dir.0.join("hf"))
        .env("HF_HUB_OFFLINE", "1")
        .output()?;
    anyhow::ensure!(
        output.status.success(),
        "capability probe failed: {}",
        output.status
    );
    let json: serde_json::Value = serde_json::from_slice(&output.stdout)?;
    assert_eq!(json["protocol"], "voqx-capabilities/1");
    assert_eq!(
        json["recognition"]["compiled"],
        cfg!(feature = "recognition")
    );
    assert_eq!(json["model_state"], "not_checked");
    assert_eq!(json["registry_state"], "not_checked");
    assert_eq!(json["microphone_capture"], false);
    assert_eq!(json["recognition"]["enroll_json"], false);
    if cfg!(feature = "recognition") {
        assert_eq!(
            json["recognition"]["preprocessing"],
            serde_json::json!(["uniform", "valid-v1"])
        );
    } else {
        assert_eq!(json["recognition"]["preprocessing"], serde_json::json!([]));
    }
    assert_eq!(
        std::fs::read_dir(&dir.0)?.count(),
        1,
        "probe created unexpected user state"
    );
    assert_eq!(
        std::fs::read_to_string(invalid_config)?,
        "not a valid [toml file"
    );
    Ok(())
}
