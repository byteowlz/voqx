//! Emit deterministic synthetic native evidence for the actual WASM contract check.
//! Development fixture only: no model, audio, enrolled person or registry is accessed.

use serde::Serialize;
use serde_json::{Value, json};
use voqx_recognition::speaker::{
    EmbeddingSpace, ReferenceSnapshot, ReferenceVoice, SpeakerCalibration, SpeakerEmbedding,
    SpeakerEvidence, SpeakerMatcher, SpeakerObservation, SpeakerResult, SpeakerTracker,
    SpeechActivity,
};

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "operation", rename_all = "snake_case")]
enum Action {
    Observe {
        values: Vec<f32>,
        generation: u64,
        start_sample: u64,
        end_sample: u64,
        activity: SpeechActivity,
    },
    Reset {
        generation: u64,
    },
}

fn observe(start_sample: u64, values: &[f32], activity: SpeechActivity) -> Action {
    Action::Observe {
        values: values.to_vec(),
        generation: 1,
        start_sample,
        end_sample: start_sample + 4,
        activity,
    }
}

fn native_observe(
    matcher: &SpeakerMatcher,
    tracker: &mut SpeakerTracker,
    observation: SpeakerObservation,
    values: &[f32],
) -> SpeakerResult<SpeakerEvidence> {
    tracker.validate_observation(observation)?;
    let mut current = || {
        let embedding = if matcher.needs_embedding(observation)? {
            Some(SpeakerEmbedding::new(
                matcher.space().clone(),
                values.to_vec(),
            )?)
        } else {
            None
        };
        tracker.observe(matcher.evaluate(observation, embedding.as_ref())?)
    };
    let result = current();
    if result.is_err() {
        tracker.invalidate();
    }
    result
}

fn one_shot(matcher: &SpeakerMatcher, observation: SpeakerObservation, values: &[f32]) -> Value {
    let result = matcher.needs_embedding(observation).and_then(|needed| {
        let embedding = if needed {
            Some(SpeakerEmbedding::new(
                matcher.space().clone(),
                values.to_vec(),
            )?)
        } else {
            None
        };
        matcher.evaluate(observation, embedding.as_ref())
    });
    match result {
        Ok(evidence) => json!({"ok": evidence}),
        Err(error) => json!({"error": error.to_string()}),
    }
}

/// Matcher-only (no tracker) evidence for every observation, as WASM `evaluate` returns.
fn one_shot_all(matcher: &SpeakerMatcher, actions: &[Action]) -> Vec<Value> {
    actions
        .iter()
        .filter_map(|action| match action {
            Action::Observe {
                values,
                generation,
                start_sample,
                end_sample,
                activity,
            } => Some(one_shot(
                matcher,
                SpeakerObservation {
                    generation: *generation,
                    start_sample: *start_sample,
                    end_sample: *end_sample,
                    activity: *activity,
                },
                values,
            )),
            Action::Reset { .. } => None,
        })
        .collect()
}

fn case(
    snapshot: &ReferenceSnapshot,
    policy: Option<&SpeakerCalibration>,
    actions: &[Action],
) -> anyhow::Result<Value> {
    let mut matcher = SpeakerMatcher::new(snapshot.clone())?;
    if let Some(policy) = policy {
        matcher = matcher.with_calibration(policy.clone())?;
    }
    let mut tracker = SpeakerTracker::new(&matcher, 1, 4)?;
    let mut expected = Vec::new();
    let one_shot = one_shot_all(&matcher, actions);
    for action in actions {
        let result = match action {
            Action::Reset { generation } => {
                tracker.reset(*generation).map(|()| json!({"reset": true}))
            }
            Action::Observe {
                values,
                generation,
                start_sample,
                end_sample,
                activity,
            } => {
                let observation = SpeakerObservation {
                    generation: *generation,
                    start_sample: *start_sample,
                    end_sample: *end_sample,
                    activity: *activity,
                };
                native_observe(&matcher, &mut tracker, observation, values)
                    .map(|evidence| json!(evidence))
            }
        };
        expected.push(match result {
            Ok(value) => json!({"ok": value}),
            Err(error) => json!({"error": error.to_string()}),
        });
    }
    Ok(
        json!({ "snapshot": snapshot, "calibration": policy, "generation": 1, "stable_samples": 4, "actions": actions, "expected": expected, "one_shot": one_shot, "revision": matcher.reference_revision() }),
    )
}

fn main() -> anyhow::Result<()> {
    let space = EmbeddingSpace {
        engine: "fixture".into(),
        model_version: "cold-v1".into(),
        dimensions: 2,
        sample_rate: 16_000,
        minimum_samples: 4,
        maximum_samples: 16,
    };
    let snapshot = ReferenceSnapshot {
        space,
        voices: vec![
            ReferenceVoice {
                voice_id: "vox-a".into(),
                embeddings: vec![vec![1.0, 0.0], vec![0.9, 0.1]],
            },
            ReferenceVoice {
                voice_id: "vox-b".into(),
                embeddings: vec![vec![0.0, 1.0]],
            },
        ],
    };
    let matcher = SpeakerMatcher::new(snapshot.clone())?;
    let policy = SpeakerCalibration {
        id: "synthetic-test-only".into(),
        reference_revision: matcher.reference_revision().into(),
        minimum_samples: 4,
        maximum_samples: 8,
        confident: 0.9,
        tentative: 0.5,
        minimum_margin: 0.05,
        numerical_guard: 0.001,
    };
    let actions = vec![
        observe(0, &[1.0, 0.0], SpeechActivity::Speech),
        observe(1, &[1.0, 0.0], SpeechActivity::Speech),
        observe(4, &[1.0, 0.0], SpeechActivity::Speech),
        observe(8, &[0.0, 1.0], SpeechActivity::Speech),
        observe(12, &[0.0, 1.0], SpeechActivity::Speech),
        observe(16, &[-1.0, -1.0], SpeechActivity::Speech),
        observe(20, &[], SpeechActivity::NoSpeech),
        observe(24, &[], SpeechActivity::Overlap),
        observe(28, &[], SpeechActivity::Unknown),
        observe(32, &[1.0, 0.0], SpeechActivity::Speech),
        observe(32, &[1.0, 0.0], SpeechActivity::Speech),
        Action::Reset { generation: 2 },
        observe(36, &[1.0, 0.0], SpeechActivity::Speech),
        Action::Observe {
            values: vec![1.0, 0.0],
            generation: 2,
            start_sample: 0,
            end_sample: 4,
            activity: SpeechActivity::Speech,
        },
        observe(36, &[1.0, 0.0], SpeechActivity::Speech),
        Action::Observe {
            values: vec![1.0, 0.0],
            generation: 2,
            start_sample: 4,
            end_sample: 8,
            activity: SpeechActivity::Speech,
        },
    ];
    let mut ambiguous = snapshot.clone();
    ambiguous.voices[1].embeddings = vec![vec![1.0, 0.0]];
    let ambiguous_matcher = SpeakerMatcher::new(ambiguous.clone())?;
    let mut ambiguous_policy = policy.clone();
    ambiguous_policy.reference_revision = ambiguous_matcher.reference_revision().into();
    let cases = vec![
        case(&snapshot, None, &actions)?,
        case(&snapshot, Some(&policy), &actions)?,
        case(&ambiguous, Some(&ambiguous_policy), &actions)?,
        case(
            &ReferenceSnapshot {
                space: snapshot.space.clone(),
                voices: vec![],
            },
            None,
            &[observe(0, &[], SpeechActivity::Speech)],
        )?,
        case(
            &snapshot,
            None,
            &[
                Action::Observe {
                    values: vec![],
                    generation: 1,
                    start_sample: 0,
                    end_sample: 1,
                    activity: SpeechActivity::Speech,
                },
                Action::Observe {
                    values: vec![1.0, 0.0],
                    generation: 1,
                    start_sample: 1,
                    end_sample: 13,
                    activity: SpeechActivity::Speech,
                },
                observe(14, &[1.0], SpeechActivity::Speech),
            ],
        )?,
    ];
    println!(
        "{}",
        serde_json::to_string(&json!({"format": "voqx-speaker-contract/1", "cases": cases}))?
    );
    Ok(())
}
