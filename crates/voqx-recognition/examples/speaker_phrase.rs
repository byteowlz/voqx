//! Explicit, local-only phrase tuning. Reads supplied WAVs; never persists enrollment.
use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::{Context, Result};
use serde::Deserialize;
use voqx_recognition::{
    fastdiar::{FastdiarWindowEncoder, phrase_embedding_space},
    speaker::{
        ReferenceSnapshot, ReferenceVoice, SpeakerCalibration, SpeakerMatcher, SpeakerObservation,
        SpeechActivity, WindowEncoder,
    },
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    minimum_ms: usize,
    maximum_ms: usize,
    voices: Vec<Voice>,
    thresholds: Option<Thresholds>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Voice {
    voice_id: String,
    references: Vec<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Thresholds {
    confident: f64,
    tentative: f64,
    minimum_margin: f64,
    numerical_guard: f64,
}

impl Job {
    fn bounds(&self) -> Result<(usize, usize)> {
        let minimum = self
            .minimum_ms
            .checked_mul(16)
            .context("minimum_ms overflow")?;
        let maximum = self
            .maximum_ms
            .checked_mul(16)
            .context("maximum_ms overflow")?;
        phrase_embedding_space(minimum, maximum)?;
        anyhow::ensure!(
            self.voices.len() <= 64,
            "at most 64 explicit canonical voice IDs"
        );
        for voice in &self.voices {
            anyhow::ensure!(
                (1..=32).contains(&voice.references.len()),
                "1-32 references per voice"
            );
        }
        Ok((minimum, maximum))
    }
}

fn read_job(path: &Path) -> Result<Job> {
    anyhow::ensure!(
        fs::metadata(path)?.len() <= 65_536,
        "job JSON exceeds 64 KiB"
    );
    serde_json::from_slice(&fs::read(path)?).context("invalid phrase tuning job")
}

fn read_wav(path: &Path, maximum_samples: usize) -> Result<Vec<f32>> {
    anyhow::ensure!(
        fs::metadata(path)?.len() <= 1_048_576,
        "WAV exceeds 1 MiB bound"
    );
    let mut reader = hound::WavReader::open(path).context("open explicitly supplied WAV")?;
    let spec = reader.spec();
    anyhow::ensure!(
        spec.channels == 1 && spec.sample_rate == 16_000,
        "WAV must be mono 16 kHz; no implicit channel mix or resampling"
    );
    anyhow::ensure!(
        reader.duration() as usize <= maximum_samples,
        "WAV exceeds selected maximum; never truncating"
    );
    match (spec.sample_format, spec.bits_per_sample) {
        (hound::SampleFormat::Float, 32) => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into),
        (hound::SampleFormat::Int, 16) => reader
            .samples::<i16>()
            .map(|sample| sample.map(|value| f32::from(value) / 32768.0))
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into),
        _ => anyhow::bail!("WAV must contain 16-bit PCM or 32-bit float"),
    }
}

fn prepare(model: &Path, job: &Job) -> Result<(FastdiarWindowEncoder, SpeakerMatcher)> {
    let (minimum, maximum) = job.bounds()?;
    let mut encoder = FastdiarWindowEncoder::load_phrase(model, minimum, maximum)?;
    let mut voices = Vec::with_capacity(job.voices.len());
    for voice in &job.voices {
        let embeddings = voice
            .references
            .iter()
            .map(|path| {
                let pcm = read_wav(path, maximum)?;
                Ok(encoder.encode(&pcm)?.values().to_vec())
            })
            .collect::<Result<Vec<_>>>()?;
        voices.push(ReferenceVoice {
            voice_id: voice.voice_id.clone(),
            embeddings,
        });
    }
    let mut matcher = SpeakerMatcher::new(ReferenceSnapshot {
        space: encoder.space().clone(),
        voices,
    })?;
    if let Some(thresholds) = &job.thresholds {
        let calibration = SpeakerCalibration {
            id: "local-phrase-tuning-not-qualified".into(),
            reference_revision: matcher.reference_revision().into(),
            minimum_samples: minimum,
            maximum_samples: maximum,
            confident: thresholds.confident,
            tentative: thresholds.tentative,
            minimum_margin: thresholds.minimum_margin,
            numerical_guard: thresholds.numerical_guard,
        };
        matcher = matcher.with_calibration(calibration)?;
    }
    Ok((encoder, matcher))
}

fn run(model: &Path, job_path: &Path, audio_path: &Path) -> Result<()> {
    let job = read_job(job_path)?;
    let (mut encoder, matcher) = prepare(model, &job)?;
    let pcm = read_wav(audio_path, encoder.space().maximum_samples)?;
    let observation = SpeakerObservation {
        generation: 1,
        start_sample: 0,
        end_sample: pcm.len() as u64,
        // Explicit caller-delimited speech probe, not a claim of automatic VAD/overlap detection.
        activity: SpeechActivity::Speech,
    };
    let started = Instant::now();
    let embedding = if matcher.needs_embedding(observation)? {
        Some(encoder.encode(&pcm)?)
    } else {
        None
    };
    let evidence = matcher.evaluate(observation, embedding.as_ref())?;
    let output = serde_json::json!({
        "qualification": "local-tuning-not-qualified",
        "input_duration_ms": pcm.len() as f64 / 16.0,
        "compute_ms": started.elapsed().as_secs_f64() * 1000.0,
        "evidence": evidence,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    match args.as_slice() {
        [model, job, audio] => run(Path::new(model), Path::new(job), Path::new(audio)),
        _ => anyhow::bail!(
            "usage: speaker_phrase MODEL.onnx JOB.json SPEECH.wav (explicit local files; see docs/speaker-library.md)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_bounds_and_unknown_fields_are_validated() -> Result<()> {
        let job: Job =
            serde_json::from_str(r#"{"minimum_ms":300,"maximum_ms":10000,"voices":[]}"#)?;
        anyhow::ensure!(job.bounds()? == (4800, 160_000));
        for bad in [
            r#"{"minimum_ms":79,"maximum_ms":10000,"voices":[]}"#,
            r#"{"minimum_ms":600,"maximum_ms":300,"voices":[]}"#,
            r#"{"minimum_ms":300,"maximum_ms":10001,"voices":[]}"#,
        ] {
            anyhow::ensure!(serde_json::from_str::<Job>(bad)?.bounds().is_err());
        }
        anyhow::ensure!(
            serde_json::from_str::<Job>(
                r#"{"minimum_ms":300,"maximum_ms":10000,"voices":[],"typo":1}"#
            )
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn wav_reader_never_truncates_or_silently_resamples() -> Result<()> {
        let directory = tempfile::tempdir()?;
        for rate in [16_000, 24_000] {
            let path = directory.path().join(format!("{rate}.wav"));
            let mut writer = hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: rate,
                    bits_per_sample: 16,
                    sample_format: hound::SampleFormat::Int,
                },
            )?;
            for _ in 0..4800 {
                writer.write_sample(123_i16)?;
            }
            writer.finalize()?;
            anyhow::ensure!(read_wav(&path, 4799).is_err());
            if rate == 16_000 {
                anyhow::ensure!(read_wav(&path, 4800)?.len() == 4800);
            } else {
                anyhow::ensure!(read_wav(&path, 4800).is_err());
            }
        }
        Ok(())
    }
}
