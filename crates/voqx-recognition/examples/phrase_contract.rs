//! Generated-signal software parity fixture. Never speaker accuracy or enrollment.
use std::{path::Path, time::Instant};

use anyhow::Result;
use serde_json::json;
use voqx_recognition::{
    fastdiar::FastdiarWindowEncoder,
    speaker::{
        ReferenceSnapshot, ReferenceVoice, SpeakerCalibration, SpeakerMatcher, SpeakerObservation,
        SpeechActivity, WindowEncoder,
    },
};

fn signal(samples: usize, frequency: f32) -> Vec<f32> {
    (0..samples)
        .map(|i| {
            let t = i as f32 / 16_000.0;
            0.05_f32.mul_add(
                (t * 350.0 * std::f32::consts::TAU).cos(),
                0.2 * (t * frequency * std::f32::consts::TAU).sin(),
            )
        })
        .collect()
}

fn main() -> Result<()> {
    let model = std::env::args_os().nth(1).ok_or_else(|| {
        anyhow::anyhow!("usage: phrase_contract MODEL.onnx (trusted explicit local graph)")
    })?;
    let started = Instant::now();
    let mut encoder = FastdiarWindowEncoder::load_phrase(Path::new(&model), 4800, 32000)?;
    let reference = encoder.encode(&signal(19_200, 200.0))?;
    let snapshot = ReferenceSnapshot {
        space: encoder.space().clone(),
        voices: vec![ReferenceVoice {
            voice_id: "vox-synthetic-fixture".into(),
            embeddings: vec![reference.values().to_vec()],
        }],
    };
    let matcher = SpeakerMatcher::new(snapshot.clone())?;
    let calibration = SpeakerCalibration {
        id: "synthetic-software-parity-not-qualified".into(),
        reference_revision: matcher.reference_revision().into(),
        minimum_samples: 4800,
        maximum_samples: 32000,
        confident: 0.99,
        tentative: 0.8,
        minimum_margin: 0.05,
        numerical_guard: 0.001,
    };
    let tuned = SpeakerMatcher::new(snapshot.clone())?.with_calibration(calibration.clone())?;
    let mut queries = Vec::new();
    for samples in [4800, 9600, 12_801, 19_200] {
        let pcm = signal(samples, 200.0);
        let begin = Instant::now();
        let embedding = encoder.encode(&pcm)?;
        let observation = SpeakerObservation {
            generation: 1,
            start_sample: 1600,
            end_sample: 1600 + samples as u64,
            activity: SpeechActivity::Speech,
        };
        queries.push(json!({
            "pcm": pcm, "values": embedding.values(), "observation": observation,
            "uncalibrated": matcher.evaluate(observation, Some(&embedding))?,
            "tuned": tuned.evaluate(observation, Some(&embedding))?,
            "native_compute_ms": begin.elapsed().as_secs_f64() * 1000.0,
        }));
    }
    println!(
        "{}",
        serde_json::to_string(&json!({
            "format": "voqx-phrase-contract/1", "snapshot": snapshot,
            "calibration": calibration, "queries": queries,
            "native_total_ms": started.elapsed().as_secs_f64() * 1000.0,
        }))?
    );
    Ok(())
}
