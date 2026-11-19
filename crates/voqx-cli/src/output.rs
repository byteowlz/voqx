//! Shared machine-readable CLI output.

use anyhow::Result;
use serde::Serialize;

pub fn print(value: &impl Serialize, yaml: bool) -> Result<()> {
    if yaml {
        print!("{}", serde_yaml::to_string(value)?);
    } else {
        println!("{}", serde_json::to_string_pretty(value)?);
    }
    Ok(())
}
