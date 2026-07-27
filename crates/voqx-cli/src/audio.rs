//! Audio helpers for enrollment: decoding, mono/rate conversion, and writing
//! the engine-specific derivatives a profile needs.

use std::path::Path;

use anyhow::{Context, Result, bail};
use rubato::{FftFixedIn, Resampler};

/// Sample rate recognition embeddings expect.
pub const RECOGNITION_SAMPLE_RATE: u32 = 16_000;

/// Sample rate the spqx synthesis references are stored at.
pub const SYNTHESIS_SAMPLE_RATE: u32 = 24_000;

/// Decoded mono audio and its sample rate.
#[derive(Debug, Clone)]
pub struct Audio {
    /// Mono `f32` samples in `[-1, 1]`.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

impl Audio {
    /// Duration in seconds.
    #[must_use]
    pub fn seconds(&self) -> f64 {
        if self.sample_rate == 0 {
            return 0.0;
        }
        self.samples.len() as f64 / f64::from(self.sample_rate)
    }

    /// Return this audio resampled to `target`, or a clone when already correct.
    ///
    /// # Errors
    ///
    /// Returns an error if the resampler cannot be constructed or run.
    pub fn to_rate(&self, target: u32) -> Result<Self> {
        if self.sample_rate == target {
            return Ok(self.clone());
        }
        Ok(Self {
            samples: resample(&self.samples, self.sample_rate, target)?,
            sample_rate: target,
        })
    }
}

/// Read a WAV file and downmix to mono `f32`.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or decoded.
pub fn read_wav(path: &Path) -> Result<Audio> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening WAV {}", path.display()))?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let scale = match spec.bits_per_sample {
                8 => 128.0,
                16 => 32_768.0,
                24 => 8_388_608.0,
                _ => 2_147_483_648.0,
            };
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<std::result::Result<Vec<_>, _>>()
                .context("reading integer WAV samples")?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .context("reading float WAV samples")?,
    };

    let channels = spec.channels.max(1) as usize;
    let samples = if channels == 1 {
        interleaved
    } else {
        interleaved
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
            .collect()
    };

    if samples.is_empty() {
        bail!("{} contains no audio samples", path.display());
    }

    Ok(Audio {
        samples,
        sample_rate: spec.sample_rate,
    })
}

/// Write mono `f32` samples as a 16-bit PCM WAV.
///
/// # Errors
///
/// Returns an error if the parent cannot be created or writing fails.
pub fn write_wav(path: &Path, audio: &Audio) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: audio.sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("creating WAV {}", path.display()))?;
    for sample in &audio.samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let value = (clamped * 32_767.0).round() as i16;
        writer
            .write_sample(value)
            .with_context(|| format!("writing WAV {}", path.display()))?;
    }
    writer
        .finalize()
        .with_context(|| format!("finalizing WAV {}", path.display()))
}

/// Resample mono samples between rates.
fn resample(samples: &[f32], source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    if source_sr == target_sr {
        return Ok(samples.to_vec());
    }
    let chunk = 1024usize;
    let mut resampler =
        FftFixedIn::<f32>::new(source_sr as usize, target_sr as usize, chunk, 2, 1)
            .map_err(|err| anyhow::anyhow!("creating resampler: {err}"))?;
    let mut output = Vec::new();
    let mut input = vec![vec![0.0f32; chunk]];
    let mut buffer = resampler.output_buffer_allocate(true);

    let mut pos = 0usize;
    while pos < samples.len() {
        let end = (pos + chunk).min(samples.len());
        let len = end - pos;
        input[0][..len].copy_from_slice(&samples[pos..end]);
        if len < chunk {
            input[0][len..].fill(0.0);
        }
        let (_, out_len) = resampler
            .process_into_buffer(&input, &mut buffer, None)
            .map_err(|err| anyhow::anyhow!("resampling: {err}"))?;
        output.extend_from_slice(&buffer[0][..out_len]);
        pos += chunk;
    }
    Ok(output)
}

/// Cheap reference-quality checks. Warnings, never hard errors.
#[must_use]
pub fn hygiene_warnings(audio: &Audio) -> Vec<String> {
    let mut warnings = Vec::new();
    let seconds = audio.seconds();
    if seconds < 6.0 {
        warnings.push(format!(
            "reference is {seconds:.1}s; 6-12s clones more reliably"
        ));
    } else if seconds > 12.0 {
        warnings.push(format!(
            "reference is {seconds:.1}s; >12s can degrade first-word reliability"
        ));
    }
    let peak = audio.samples.iter().fold(0f32, |a, s| a.max(s.abs()));
    if peak >= 0.999 {
        warnings.push("audio appears clipped (peak at full scale)".to_string());
    }
    if peak < 0.05 {
        warnings.push(format!("audio is very quiet (peak {peak:.3})"));
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::{Audio, RECOGNITION_SAMPLE_RATE, hygiene_warnings, read_wav, write_wav};

    #[test]
    fn roundtrip_and_resample() {
        let dir = std::env::temp_dir().join(format!("voqx-audio-{}", std::process::id()));
        let path = dir.join("a.wav");
        let audio = Audio {
            samples: (0..24_000).map(|i| ((i as f32) / 50.0).sin() * 0.5).collect(),
            sample_rate: 24_000,
        };
        write_wav(&path, &audio).expect("writing wav");
        let read = read_wav(&path).expect("reading wav");
        assert_eq!(read.sample_rate, 24_000);
        assert!((read.seconds() - 1.0).abs() < 0.01);

        let down = read.to_rate(RECOGNITION_SAMPLE_RATE).expect("resampling");
        assert_eq!(down.sample_rate, RECOGNITION_SAMPLE_RATE);
        assert!(down.samples.len() > 15_000 && down.samples.len() < 17_000);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn hygiene_flags_short_audio() {
        let audio = Audio {
            samples: vec![0.5; 16_000],
            sample_rate: 16_000,
        };
        assert!(hygiene_warnings(&audio).iter().any(|w| w.contains("6-12s")));
    }
}
