//! Subprocess proof: `voqx import trnscrb` refuses unusable vectors per speaker,
//! imports usable ones, counts skipped audio explicitly instead of dropping it
//! silently, and marks the imported provenance as unverified.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, ensure};

/// One-hot unit vector, so every fixture satisfies the normalization check.
fn one_hot(dimension: usize, index: usize) -> Vec<f32> {
    let mut vector = vec![0.0_f32; dimension];
    if index < dimension {
        vector[index] = 1.0;
    }
    vector
}

/// L2-normalized mean, matching the registry's own centroid rule.
fn centroid(embeddings: &[Vec<f32>]) -> Vec<f32> {
    let dimension = embeddings.first().map_or(0, Vec::len);
    let mut mean = vec![0.0_f32; dimension];
    for embedding in embeddings {
        for (slot, value) in mean.iter_mut().zip(embedding) {
            *slot += value;
        }
    }
    let count = embeddings.len() as f32;
    for slot in &mut mean {
        *slot /= count;
    }
    let norm: f32 = mean.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for slot in &mut mean {
            *slot /= norm;
        }
    }
    mean
}

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Self> {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let sequence = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
        let root = std::env::temp_dir().join(format!(
            "voqx-cli-import-{}-{nonce}-{sequence}",
            std::process::id()
        ));
        std::fs::create_dir(&root)?;
        std::fs::write(root.join("config.toml"), "")?;
        Ok(Self(root))
    }

    fn registry_root(&self) -> PathBuf {
        self.0.join("data/voice-registry")
    }

    fn registry_file(&self) -> PathBuf {
        self.registry_root().join("registry.json")
    }

    fn write_registry(&self, speakers: &str) -> Result<PathBuf> {
        let path = self.0.join("trnscrb/registry.json");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(
            &path,
            format!("{{\"version\":1,\"speakers\":[{speakers}]}}"),
        )?;
        Ok(path)
    }

    fn write_wav(&self, name: &str) -> Result<PathBuf> {
        let path = self.0.join("audio").join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec)?;
        for index in 0..1600_u16 {
            writer.write_sample((index % 200) as i16 * 8)?;
        }
        writer.finalize()?;
        Ok(path)
    }

    fn invoke(&self, args: &[&str]) -> Result<Output> {
        Ok(Command::new(env!("CARGO_BIN_EXE_voqx"))
            .args(args)
            .arg("--config")
            .arg(self.0.join("config.toml"))
            .env("HOME", &self.0)
            .env("XDG_CONFIG_HOME", self.0.join("config"))
            .env("XDG_STATE_HOME", self.0.join("state"))
            .env("XDG_DATA_HOME", self.0.join("data"))
            .env("XDG_CACHE_HOME", self.0.join("cache"))
            .env("HF_HOME", self.0.join("hf"))
            .env("HF_HUB_OFFLINE", "1")
            .output()?)
    }

    fn run(&self, args: &[&str]) -> Result<String> {
        let output = self.invoke(args)?;
        ensure!(
            output.status.success(),
            "import failed: {} {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(String::from_utf8(output.stdout)?)
    }

    fn load(&self) -> Result<serde_json::Value> {
        let body = std::fs::read_to_string(self.registry_file())?;
        Ok(serde_json::from_str(&body)?)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn speaker_json(id: &str, name: &str, embeddings: &[Vec<f32>], samples: &[&Path]) -> String {
    let sample_paths: Vec<String> = samples
        .iter()
        .map(|path| path.display().to_string())
        .collect();
    serde_json::json!({
        "id": id,
        "name": name,
        "aliases": [format!("alias-{name}")],
        "embedding_model": "speakrs-wespeaker",
        "centroid": centroid(embeddings),
        "embeddings": embeddings,
        "samples": sample_paths,
    })
    .to_string()
}

#[test]
fn valid_speakers_import_vectors_centroid_provenance_and_samples() -> Result<()> {
    let dir = Scratch::new()?;
    let wav = dir.write_wav("one.wav")?;
    let embeddings = vec![one_hot(256, 0), one_hot(256, 1)];
    let source = dir.write_registry(&speaker_json(
        "SPEAKER_00",
        "Alice",
        &embeddings,
        &[wav.as_path()],
    ))?;

    let stdout = dir.run(&["import", "trnscrb", "--from", &source.display().to_string()])?;
    ensure!(stdout.contains("imported SPEAKER_00 (Alice)"));
    ensure!(stdout.contains("1 copied, 0 missing, 0 unreadable"));
    ensure!(stdout.contains("unverified-trnscrb-v1"));

    let registry = dir.load()?;
    let identities = registry["identities"]
        .as_array()
        .context("identities array")?;
    ensure!(identities.len() == 1);
    let identity = &identities[0];
    ensure!(identity["id"] == "SPEAKER_00");
    ensure!(identity["display_name"] == "Alice");
    ensure!(identity["aliases"][0] == "alias-Alice");

    let profile = &identity["recognition_profiles"][0];
    ensure!(profile["engine"] == "speakrs-wespeaker");
    ensure!(profile["model_version"] == "unverified-trnscrb-v1");
    ensure!(profile["tags"][0] == "provenance:trnscrb-registry-v1");
    ensure!(profile["tags"][1] == "source-embedding-model:speakrs-wespeaker");
    ensure!(
        profile["embeddings"]
            .as_array()
            .context("embeddings")?
            .len()
            == 2
    );
    ensure!(profile["samples"].as_array().context("samples")?.len() == 1);
    let stored = profile["samples"][0].as_str().context("sample path")?;
    ensure!(stored.starts_with("identities/SPEAKER_00/recognition/speakrs-wespeaker/"));
    ensure!(dir.registry_root().join(stored).exists());

    let expected = centroid(&embeddings);
    let stored_centroid = profile["centroid"].as_array().context("centroid")?;
    ensure!(stored_centroid.len() == 256);
    for (index, value) in stored_centroid.iter().enumerate() {
        let value = value.as_f64().context("centroid value")?;
        ensure!((value - f64::from(expected[index])).abs() < 1e-6);
    }
    Ok(())
}

#[test]
fn unusable_vectors_are_refused_per_speaker_without_blocking_the_rest() -> Result<()> {
    let dir = Scratch::new()?;
    let good = vec![one_hot(256, 0)];
    let mixed = vec![one_hot(256, 0), one_hot(192, 0)];
    let mut unnormalized = one_hot(256, 0);
    unnormalized[0] = 2.0;
    let zero = vec![vec![0.0_f32; 256]];
    let source = dir.write_registry(&format!(
        "{},{},{},{}",
        speaker_json("SPEAKER_OK", "Ok", &good, &[]),
        speaker_json("SPEAKER_MIXED", "Mixed", &mixed, &[]),
        speaker_json("SPEAKER_UNNORMALIZED", "Wide", &[unnormalized], &[]),
        speaker_json("SPEAKER_ZERO", "Zero", &zero, &[]),
    ))?;

    let stdout = dir.run(&["import", "trnscrb", "--from", &source.display().to_string()])?;
    ensure!(stdout.contains("refused SPEAKER_MIXED (Mixed): reference vector 1 has dimension 192"));
    ensure!(stdout.contains("refused SPEAKER_UNNORMALIZED (Wide): reference vector 0 has L2 norm"));
    ensure!(stdout.contains("refused SPEAKER_ZERO (Zero): reference vector 0 has zero norm"));
    ensure!(stdout.contains("imported 1 identity(ies), refused 3"));

    let registry = dir.load()?;
    let identities = registry["identities"]
        .as_array()
        .context("identities array")?;
    ensure!(identities.len() == 1);
    ensure!(identities[0]["id"] == "SPEAKER_OK");
    Ok(())
}

#[test]
fn missing_and_unreadable_audio_is_counted_and_never_drops_the_vectors() -> Result<()> {
    let dir = Scratch::new()?;
    let wav = dir.write_wav("two.wav")?;
    let corrupt = dir.0.join("audio/corrupt.wav");
    if let Some(parent) = corrupt.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&corrupt, b"not a wav file")?;
    let missing = dir.0.join("audio/missing.wav");
    let embeddings = vec![one_hot(256, 2), one_hot(256, 3)];
    let source = dir.write_registry(&speaker_json(
        "SPEAKER_01",
        "Bob",
        &embeddings,
        &[missing.as_path(), corrupt.as_path(), wav.as_path()],
    ))?;

    let stdout = dir.run(&["import", "trnscrb", "--from", &source.display().to_string()])?;
    ensure!(stdout.contains("1 copied, 1 missing, 1 unreadable"));
    ensure!(stdout.contains("samples: 1 copied, 1 missing, 1 unreadable"));

    let registry = dir.load()?;
    let profile = &registry["identities"][0]["recognition_profiles"][0];
    ensure!(
        profile["embeddings"]
            .as_array()
            .context("embeddings")?
            .len()
            == 2
    );
    ensure!(profile["samples"].as_array().context("samples")?.len() == 1);
    Ok(())
}

#[test]
fn repeated_import_and_name_collisions_are_refused_without_duplicates() -> Result<()> {
    let dir = Scratch::new()?;
    let embeddings = vec![one_hot(256, 4)];
    let source = dir.write_registry(&format!(
        "{},{}",
        speaker_json("SPEAKER_02", "Carol", &embeddings, &[]),
        speaker_json("SPEAKER_03", "Carol", &embeddings, &[]),
    ))?;
    let args = ["import", "trnscrb", "--from", &source.display().to_string()];

    let first = dir.run(&args)?;
    ensure!(first.contains("imported 1 identity(ies), refused 1"));
    ensure!(
        first.contains(
            "refused SPEAKER_03 (Carol): display name 'Carol' appears twice in the source"
        )
    );

    let second = dir.run(&args)?;
    ensure!(second.contains("imported 0 identity(ies), refused 2"));
    ensure!(second.contains("refused SPEAKER_02 (Carol): already in the registry"));
    ensure!(
        dir.load()?["identities"]
            .as_array()
            .context("identities")?
            .len()
            == 1
    );
    Ok(())
}

#[test]
fn dry_run_validates_vectors_and_audio_without_writing_state() -> Result<()> {
    let dir = Scratch::new()?;
    let missing = dir.0.join("audio/absent.wav");
    let embeddings = vec![one_hot(256, 5)];
    let source = dir.write_registry(&format!(
        "{},{}",
        speaker_json("SPEAKER_04", "Dave", &embeddings, &[missing.as_path()]),
        speaker_json("SPEAKER_05", "Eve", &[one_hot(192, 5)], &[]),
    ))?;

    let stdout = dir.run(&[
        "import",
        "trnscrb",
        "--dry-run",
        "--from",
        &source.display().to_string(),
    ])?;
    ensure!(
        stdout.contains(
            "would import SPEAKER_04 (Dave): 1 embedding(s), samples 0 copied, 1 missing"
        )
    );
    ensure!(stdout.contains("refused SPEAKER_05 (Eve)"));
    ensure!(stdout.contains("would import 1 identity(ies), refuse 1"));
    ensure!(!dir.registry_file().exists());
    ensure!(!dir.registry_root().join("identities").exists());
    Ok(())
}
