//! Public-interface tests with synthetic vectors, not speaker accuracy evidence.

#[path = "../src/test_support.rs"]
mod support;
use support::check_eq;

use std::{cell::Cell, rc::Rc};

use voqx_recognition::speaker::{
    EmbeddingSpace, ReferenceSnapshot, ReferenceVoice, SpeakerAttributor, SpeakerCalibration,
    SpeakerEmbedding, SpeakerError, SpeakerMatcher, SpeakerObservation, SpeakerResult,
    SpeakerStatus, SpeakerTracker, SpeechActivity, WindowEncoder,
};

fn space() -> EmbeddingSpace {
    EmbeddingSpace {
        engine: "fixture".into(),
        model_version: "cold-v1".into(),
        dimensions: 2,
        sample_rate: 16_000,
        minimum_samples: 4,
        maximum_samples: 16,
    }
}

fn snapshot() -> ReferenceSnapshot {
    ReferenceSnapshot {
        space: space(),
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
    }
}

fn calibration(matcher: &SpeakerMatcher) -> SpeakerCalibration {
    SpeakerCalibration {
        id: "synthetic-test-only".into(),
        reference_revision: matcher.reference_revision().into(),
        minimum_samples: 4,
        maximum_samples: 8,
        confident: 0.9,
        tentative: 0.5,
        minimum_margin: 0.05,
        numerical_guard: 0.001,
    }
}

fn calibrated() -> SpeakerResult<SpeakerMatcher> {
    let matcher = SpeakerMatcher::new(snapshot())?;
    let policy = calibration(&matcher);
    matcher.with_calibration(policy)
}

const fn observation(start: u64) -> SpeakerObservation {
    SpeakerObservation {
        generation: 1,
        start_sample: start,
        end_sample: start + 4,
        activity: SpeechActivity::Speech,
    }
}

fn vector(values: &[f32]) -> SpeakerResult<SpeakerEmbedding> {
    SpeakerEmbedding::new(space(), values.to_vec())
}

#[test]
fn uncalibrated_is_not_a_nearest_person_label() -> anyhow::Result<()> {
    let matcher = SpeakerMatcher::new(snapshot())?;
    let evidence = matcher.evaluate(observation(0), Some(&vector(&[1.0, 0.0])?))?;
    check_eq(
        &(evidence.status, evidence.voice_id),
        &(SpeakerStatus::Uncalibrated, None),
    )?;
    check_eq(&(evidence.candidates.len()), &(2))?;
    check_eq(&(evidence.candidates[0].voice_id), &("vox-a"))?;
    Ok(())
}

#[test]
fn scoring_preserves_both_candidate_metrics_and_unknown() -> anyhow::Result<()> {
    let matcher = calibrated()?;
    let evidence = matcher.evaluate(observation(0), Some(&vector(&[1.0, 0.0])?))?;
    check_eq(
        &(evidence.status, evidence.voice_id.as_deref()),
        &(SpeakerStatus::Confident, Some("vox-a")),
    )?;
    anyhow::ensure!(
        evidence.candidates[0].nearest
            > evidence.candidates[0]
                .centroid
                .ok_or(SpeakerError::InvalidInput("fixture centroid"))?
    );
    let unknown = matcher.evaluate(observation(0), Some(&vector(&[-1.0, -1.0])?))?;
    check_eq(
        &(unknown.status, unknown.voice_id),
        &(SpeakerStatus::Unknown, None),
    )?;
    Ok(())
}

#[test]
fn close_competitors_abstain_even_when_both_scores_are_high() -> anyhow::Result<()> {
    let mut references = snapshot();
    references.voices[1].embeddings = vec![vec![1.0, 0.0]];
    let matcher = SpeakerMatcher::new(references)?;
    let policy = calibration(&matcher);
    let evidence = matcher
        .with_calibration(policy)?
        .evaluate(observation(0), Some(&vector(&[1.0, 0.0])?))?;
    check_eq(
        &(evidence.status, evidence.voice_id),
        &(SpeakerStatus::Ambiguous, None),
    )?;
    Ok(())
}

#[test]
fn guard_and_duration_band_are_explicit() -> anyhow::Result<()> {
    let matcher = SpeakerMatcher::new(snapshot())?;
    let mut policy = calibration(&matcher);
    policy.confident = 1.0;
    policy.numerical_guard = 0.0;
    let guarded = matcher.with_calibration(policy)?;
    check_eq(
        &(guarded
            .evaluate(observation(0), Some(&vector(&[1.0, 0.0])?))?
            .status),
        &(SpeakerStatus::Confident),
    )?;
    let matcher = calibrated()?;
    let mut long = observation(0);
    long.end_sample = 12;
    check_eq(
        &(matcher.evaluate(long, Some(&vector(&[1.0, 0.0])?))?.status),
        &(SpeakerStatus::Uncalibrated),
    )?;
    Ok(())
}

#[test]
fn numerical_guard_really_changes_near_threshold_decisions() -> anyhow::Result<()> {
    let mut references = snapshot();
    references.voices[0].embeddings.truncate(1);
    let matcher = SpeakerMatcher::new(references.clone())?;
    let mut policy = calibration(&matcher);
    policy.confident = 0.75;
    policy.numerical_guard = 0.0;
    let query = vector(&[0.8, 0.6])?;
    check_eq(
        &matcher
            .with_calibration(policy.clone())?
            .evaluate(observation(0), Some(&query))?
            .status,
        &SpeakerStatus::Confident,
    )?;
    policy.numerical_guard = 0.125;
    let guarded = SpeakerMatcher::new(references)?
        .with_calibration(policy)?
        .evaluate(observation(0), Some(&query))?;
    check_eq(
        &(guarded.status, guarded.voice_id),
        &(SpeakerStatus::Tentative, None),
    )?;
    Ok(())
}

#[test]
fn undefined_centroid_is_absent_not_a_fake_zero_cosine() -> anyhow::Result<()> {
    let mut references = snapshot();
    references.voices[0].embeddings = vec![vec![1.0, 0.0], vec![-1.0, 0.0]];
    let evidence =
        SpeakerMatcher::new(references)?.evaluate(observation(0), Some(&vector(&[1.0, 0.0])?))?;
    check_eq(&evidence.candidates[0].centroid, &None)?;
    Ok(())
}

#[test]
fn activity_and_short_input_never_require_embeddings() -> anyhow::Result<()> {
    let matcher = calibrated()?;
    for (activity, expected) in [
        (SpeechActivity::NoSpeech, SpeakerStatus::NoSpeech),
        (SpeechActivity::Overlap, SpeakerStatus::Overlap),
        (SpeechActivity::Unknown, SpeakerStatus::Unknown),
    ] {
        let mut input = observation(0);
        input.activity = activity;
        anyhow::ensure!(!matcher.needs_embedding(input)?);
        let evidence = matcher.evaluate(input, None)?;
        check_eq(
            &(evidence.status, evidence.voice_id, evidence.candidates),
            &(expected, None, vec![]),
        )?;
    }
    let mut short = observation(0);
    short.end_sample = 3;
    check_eq(
        &(matcher.evaluate(short, None)?.status),
        &(SpeakerStatus::InsufficientAudio),
    )?;
    Ok(())
}

#[test]
fn bank_revision_binds_reference_bytes_order_count_and_model() -> anyhow::Result<()> {
    let original = SpeakerMatcher::new(snapshot())?;
    let policy = calibration(&original);
    for change in 0..4 {
        let mut changed = snapshot();
        match change {
            0 => changed.voices[0].embeddings[0][1] = 0.01,
            1 => changed.voices.reverse(),
            2 => changed.voices[0]
                .embeddings
                .pop()
                .map(|_| ())
                .ok_or(SpeakerError::InvalidInput("fixture"))?,
            _ => changed.space.model_version.push_str("-changed"),
        }
        anyhow::ensure!(
            SpeakerMatcher::new(changed)?
                .with_calibration(policy.clone())
                .is_err()
        );
    }
    check_eq(
        &(original.reference_revision()),
        &(SpeakerMatcher::new(snapshot())?.reference_revision()),
    )?;
    Ok(())
}

#[test]
fn malformed_confident_evidence_cannot_advance_tracking() -> anyhow::Result<()> {
    let matcher = calibrated()?;
    let mut tracker = SpeakerTracker::new(&matcher, 1, 4)?;
    let valid = matcher.evaluate(observation(0), Some(&vector(&[1.0, 0.0])?))?;
    let mut malformed = valid.clone();
    malformed.voice_id = None;
    malformed.candidates.clear();
    anyhow::ensure!(matches!(
        tracker.observe(malformed),
        Err(SpeakerError::InvalidInput(_))
    ));
    check_eq(&tracker.observe(valid)?.status, &SpeakerStatus::Tentative)?;
    Ok(())
}

#[test]
fn reusing_a_policy_name_cannot_hide_changed_thresholds() -> anyhow::Result<()> {
    let matcher = calibrated()?;
    let mut tracker = SpeakerTracker::new(&matcher, 1, 4)?;
    let changed = SpeakerMatcher::new(snapshot())?;
    let mut policy = calibration(&changed);
    policy.confident = 0.8;
    let changed = changed.with_calibration(policy)?;
    check_eq(&matcher.calibration_id(), &changed.calibration_id())?;
    anyhow::ensure!(matcher.calibration_revision() != changed.calibration_revision());
    let evidence = changed.evaluate(observation(0), Some(&vector(&[1.0, 0.0])?))?;
    anyhow::ensure!(matches!(
        tracker.observe(evidence),
        Err(SpeakerError::StaleObservation)
    ));
    Ok(())
}

#[test]
fn invalid_calibration_is_rejected_not_clamped() -> anyhow::Result<()> {
    for change in 0..7 {
        let matcher = SpeakerMatcher::new(snapshot())?;
        let mut policy = calibration(&matcher);
        match change {
            0 => policy.confident = f64::NAN,
            1 => policy.numerical_guard = -0.1,
            2 => policy.tentative = 0.99,
            3 => policy.minimum_margin = f64::INFINITY,
            4 => policy.minimum_samples = 1,
            5 => policy.maximum_samples = 17,
            _ => policy.id.clear(),
        }
        anyhow::ensure!(matcher.with_calibration(policy).is_err());
    }
    Ok(())
}

#[test]
fn invalid_vectors_and_profile_dimensions_never_fall_back() -> anyhow::Result<()> {
    for invalid in [
        vec![],
        vec![1.0],
        vec![0.0, 0.0],
        vec![f32::NAN, 1.0],
        vec![f32::INFINITY, 0.0],
    ] {
        anyhow::ensure!(SpeakerEmbedding::new(space(), invalid.clone()).is_err());
        let mut references = snapshot();
        references.voices[0].embeddings = vec![invalid];
        anyhow::ensure!(SpeakerMatcher::new(references).is_err());
    }
    let mut other_space = space();
    other_space.model_version = "different-reset-cadence".into();
    let embedding = SpeakerEmbedding::new(other_space, vec![1.0, 0.0])?;
    let evidence = calibrated()?.evaluate(observation(0), Some(&embedding))?;
    check_eq(
        &(evidence.status, evidence.voice_id),
        &(SpeakerStatus::IncompatibleModel, None),
    )?;
    Ok(())
}

#[test]
fn references_are_bounded_and_ids_cannot_collide() {
    for change in 0..4 {
        let mut references = snapshot();
        match change {
            0 => references.voices[1].voice_id = references.voices[0].voice_id.clone(),
            1 => references.voices[0].embeddings = vec![vec![1.0, 0.0]; 33],
            2 => references.voices[0].voice_id.clear(),
            _ => references.voices = vec![references.voices[0].clone(); 65],
        }
        assert!(SpeakerMatcher::new(references).is_err());
    }
}

#[test]
fn empty_selection_stays_unknown_and_does_not_request_inference() -> anyhow::Result<()> {
    let matcher = SpeakerMatcher::new(ReferenceSnapshot {
        space: space(),
        voices: vec![],
    })?;
    anyhow::ensure!(!matcher.needs_embedding(observation(0))?);
    check_eq(
        &(matcher.evaluate(observation(0), None)?.status),
        &(SpeakerStatus::Unknown),
    )?;
    Ok(())
}

#[test]
fn stabilization_counts_endpoint_advance_not_repeated_window_length() -> anyhow::Result<()> {
    let matcher = calibrated()?;
    let mut tracker = SpeakerTracker::new(&matcher, 1, 4)?;
    for (start, expected) in [
        (0, SpeakerStatus::Tentative),
        (1, SpeakerStatus::Tentative),
        (2, SpeakerStatus::Tentative),
        (4, SpeakerStatus::Confident),
    ] {
        let evidence = matcher.evaluate(observation(start), Some(&vector(&[1.0, 0.0])?))?;
        check_eq(&(tracker.observe(evidence)?.status), &(expected))?;
    }
    let repeated = matcher.evaluate(observation(4), Some(&vector(&[1.0, 0.0])?))?;
    anyhow::ensure!(matches!(
        tracker.observe(repeated),
        Err(SpeakerError::OutOfOrder)
    ));
    Ok(())
}

#[test]
fn speaker_switch_never_inherits_the_previous_persons_label() -> anyhow::Result<()> {
    let matcher = calibrated()?;
    let mut tracker = SpeakerTracker::new(&matcher, 1, 4)?;
    for (start, values, expected, id) in [
        (0, [1.0, 0.0], SpeakerStatus::Tentative, None),
        (4, [1.0, 0.0], SpeakerStatus::Confident, Some("vox-a")),
        (8, [0.0, 1.0], SpeakerStatus::Tentative, None),
        (12, [0.0, 1.0], SpeakerStatus::Confident, Some("vox-b")),
    ] {
        let evidence =
            tracker.observe(matcher.evaluate(observation(start), Some(&vector(&values)?))?)?;
        check_eq(
            &(evidence.status, evidence.voice_id.as_deref()),
            &(expected, id),
        )?;
    }
    Ok(())
}

#[test]
fn uncertainty_and_coverage_gaps_break_continuity() -> anyhow::Result<()> {
    for activity in [
        SpeechActivity::NoSpeech,
        SpeechActivity::Overlap,
        SpeechActivity::Unknown,
    ] {
        let matcher = calibrated()?;
        let mut tracker = SpeakerTracker::new(&matcher, 1, 4)?;
        for start in [0, 4] {
            tracker.observe(matcher.evaluate(observation(start), Some(&vector(&[1.0, 0.0])?))?)?;
        }
        let mut uncertain = observation(8);
        uncertain.activity = activity;
        tracker.observe(matcher.evaluate(uncertain, None)?)?;
        check_eq(
            &(tracker
                .observe(matcher.evaluate(observation(12), Some(&vector(&[1.0, 0.0])?))?)?
                .status),
            &(SpeakerStatus::Tentative),
        )?;
        // A gap must restart the dwell rather than promote the pending match.
        check_eq(
            &(tracker
                .observe(matcher.evaluate(observation(20), Some(&vector(&[1.0, 0.0])?))?)?
                .status),
            &(SpeakerStatus::Tentative),
        )?;
    }
    Ok(())
}

#[test]
fn generation_bank_and_policy_fences_reject_delayed_work() -> anyhow::Result<()> {
    let matcher = calibrated()?;
    let mut tracker = SpeakerTracker::new(&matcher, 1, 4)?;
    let old = matcher.evaluate(observation(0), Some(&vector(&[1.0, 0.0])?))?;
    tracker.reset(2)?;
    anyhow::ensure!(matches!(
        tracker.observe(old.clone()),
        Err(SpeakerError::StaleObservation)
    ));
    anyhow::ensure!(tracker.reset(1).is_err());
    for change in 0..3 {
        let mut delayed = old.clone();
        delayed.observation.generation = 2;
        match change {
            0 => delayed.reference_revision.push('x'),
            1 => delayed.calibration_id = None,
            _ => delayed.space.model_version.push('x'),
        }
        anyhow::ensure!(matches!(
            tracker.observe(delayed),
            Err(SpeakerError::StaleObservation)
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct CountingEncoder {
    calls: Rc<Cell<usize>>,
    failure: Rc<Cell<bool>>,
    space: EmbeddingSpace,
}

impl WindowEncoder for CountingEncoder {
    fn space(&self) -> &EmbeddingSpace {
        &self.space
    }
    fn encode(&mut self, samples: &[f32]) -> SpeakerResult<SpeakerEmbedding> {
        self.calls.set(self.calls.get() + 1);
        if self.failure.get() {
            return Err(SpeakerError::Backend("synthetic injected failure".into()));
        }
        vector(if samples[0] > 0.0 {
            &[1.0, 0.0]
        } else {
            &[0.0, 1.0]
        })
    }
}

#[test]
fn attributor_shares_one_inference_across_people_and_skips_abstentions() -> anyhow::Result<()> {
    let calls = Rc::new(Cell::new(0));
    let encoder = CountingEncoder {
        calls: calls.clone(),
        failure: Rc::new(Cell::new(false)),
        space: space(),
    };
    let mut pipeline = SpeakerAttributor::new(encoder, calibrated()?, 1, 4)?;
    pipeline.process(observation(0), &[0.1; 4])?;
    check_eq(&(calls.get()), &(1))?;
    for (start, activity) in [
        (4, SpeechActivity::NoSpeech),
        (8, SpeechActivity::Overlap),
        (12, SpeechActivity::Unknown),
    ] {
        let mut input = observation(start);
        input.activity = activity;
        pipeline.process(input, &[0.1; 4])?;
    }
    let silent = pipeline.process(observation(16), &[0.0; 4])?;
    check_eq(&(silent.status), &(SpeakerStatus::InsufficientAudio))?;
    check_eq(&(calls.get()), &(1))?;
    pipeline.reset(2)?;
    anyhow::ensure!(matches!(
        pipeline.process(observation(20), &[0.1; 4]),
        Err(SpeakerError::StaleObservation)
    ));
    check_eq(&(calls.get()), &(1))?;
    Ok(())
}

#[test]
fn failed_inference_clears_stability_and_never_becomes_negative_identity_evidence()
-> anyhow::Result<()> {
    let failure = Rc::new(Cell::new(false));
    let encoder = CountingEncoder {
        calls: Rc::new(Cell::new(0)),
        failure: failure.clone(),
        space: space(),
    };
    let mut pipeline = SpeakerAttributor::new(encoder, calibrated()?, 1, 4)?;
    pipeline.process(observation(0), &[0.1; 4])?;
    check_eq(
        &(pipeline.process(observation(4), &[0.1; 4])?.status),
        &(SpeakerStatus::Confident),
    )?;
    failure.set(true);
    anyhow::ensure!(matches!(
        pipeline.process(observation(8), &[0.1; 4]),
        Err(SpeakerError::Backend(_))
    ));
    failure.set(false);
    check_eq(
        &(pipeline.process(observation(12), &[0.1; 4])?.status),
        &(SpeakerStatus::Tentative),
    )?;
    Ok(())
}

#[test]
fn invalid_pcm_scope_is_rejected_before_inference() -> anyhow::Result<()> {
    let calls = Rc::new(Cell::new(0));
    let encoder = CountingEncoder {
        calls: calls.clone(),
        failure: Rc::new(Cell::new(false)),
        space: space(),
    };
    let mut pipeline = SpeakerAttributor::new(encoder, calibrated()?, 1, 4)?;
    anyhow::ensure!(pipeline.process(observation(0), &[0.1; 3]).is_err());
    anyhow::ensure!(pipeline.process(observation(0), &[f32::NAN; 4]).is_err());
    let mut oversized = observation(0);
    oversized.end_sample = 17;
    anyhow::ensure!(pipeline.process(oversized, &[0.1; 17]).is_err());
    check_eq(&(calls.get()), &(0))?;
    Ok(())
}

#[test]
fn evidence_and_debug_output_do_not_include_voiceprints() -> anyhow::Result<()> {
    let embedding = vector(&[0.123_456, 0.987_654])?;
    anyhow::ensure!(!format!("{embedding:?}").contains("0.123"));
    let snapshot = snapshot();
    anyhow::ensure!(!format!("{snapshot:?}").contains("embeddings"));
    let evidence = calibrated()?.evaluate(observation(0), Some(&embedding))?;
    let json =
        serde_json::to_string(&evidence).map_err(|_| SpeakerError::InvalidInput("fixture JSON"))?;
    anyhow::ensure!(!json.contains("embeddings"));
    check_eq(
        &(serde_json::from_str::<voqx_recognition::speaker::SpeakerEvidence>(&json)
            .map_err(|_| SpeakerError::InvalidInput("fixture JSON"))?),
        &(evidence),
    )?;
    Ok(())
}
