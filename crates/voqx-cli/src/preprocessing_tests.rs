//! CLI parsing contracts; no model load, files or registry mutations.

use anyhow::{Result, bail};
use clap::Parser;
use voqx_core::RecognitionPreprocessing;

use crate::{Cli, Command};

#[test]
fn enrollment_defaults_to_uniform_and_requires_recognition_for_explicit_pooling() -> Result<()> {
    let Command::Enroll(args) =
        Cli::try_parse_from(["voqx", "enroll", "Fixture", "--audio", "a.wav"])?.command
    else {
        bail!("expected enrollment");
    };
    assert_eq!(args.preprocessing, RecognitionPreprocessing::Uniform);
    assert!(!args.recognition);
    assert!(
        Cli::try_parse_from([
            "voqx",
            "enroll",
            "Fixture",
            "--audio",
            "a.wav",
            "--preprocessing",
            "valid-v1"
        ])
        .is_err()
    );
    let Command::Enroll(args) = Cli::try_parse_from([
        "voqx",
        "enroll",
        "Fixture",
        "--audio",
        "a.wav",
        "--recognition",
        "--preprocessing",
        "valid-v1",
    ])?
    .command
    else {
        bail!("expected enrollment");
    };
    assert_eq!(args.preprocessing, RecognitionPreprocessing::ValidV1);
    assert!(args.recognition);
    assert!(
        Cli::try_parse_from([
            "voqx",
            "enroll",
            "Fixture",
            "--audio",
            "a.wav",
            "--recognition",
            "--preprocessing",
            "typo"
        ])
        .is_err()
    );
    Ok(())
}

#[cfg(feature = "recognition")]
#[test]
fn identification_selects_the_same_contract_and_defaults_to_uniform() -> Result<()> {
    let Command::Identify { preprocessing, .. } =
        Cli::try_parse_from(["voqx", "identify", "a.wav"])?.command
    else {
        bail!("expected identification");
    };
    assert_eq!(preprocessing, RecognitionPreprocessing::Uniform);
    let Command::Identify { preprocessing, .. } =
        Cli::try_parse_from(["voqx", "identify", "a.wav", "--preprocessing", "valid-v1"])?.command
    else {
        bail!("expected identification");
    };
    assert_eq!(preprocessing, RecognitionPreprocessing::ValidV1);
    assert!(Cli::try_parse_from(["voqx", "identify", "a.wav", "--preprocessing", "typo"]).is_err());
    Ok(())
}
