# Embedded speaker library

`voqx-recognition` provides in-process CPU FASTDIAR, compatible reference matching
and timestamped known-speaker evidence. No CLI installation, Python, microphone
owner, registry daemon or HTTP inference is required. Existing browser-test and
WeSpeaker defaults are unchanged; application filtering is not enabled.

## Dependencies

Use pinned Git/path dependencies until public publication is approved:

```toml
voqx-recognition = { git = "<repository>", rev = "<commit>", default-features = false, features = ["fastdiar", "registry"] }
```

| Features | Contents |
|---|---|
| default / `wespeaker` | Existing WeSpeaker interface and behavior |
| none | Portable matching/tracking; no native ML or filesystem registry |
| `fastdiar` | One-thread CPU ONNX adapter; no WeSpeaker/OpenBLAS |
| `registry` | Read-only canonical snapshots and reviewed profile-building adapters |
| `wasm` | JavaScript bindings to the same Rust matcher/tracker |

The application bundles its executable and native runtime libraries. The host
supplies the pinned model asset; the library never downloads it. Model
redistribution/licensing approval is separate from code reuse. Model bytes are not
total runtime memory. Browser hosts bundle generated WASM and compatible encoder/
ORT Web assets; users install no extra binaries.

## Native use

```rust,ignore
use voqx_recognition::{
    fastdiar::FastdiarWindowEncoder,
    speaker::{SpeakerAttributor, SpeakerMatcher, SpeakerObservation, SpeechActivity, WindowEncoder},
    speaker_registry::reference_snapshot,
};

let encoder = FastdiarWindowEncoder::load(model_path)?;
let snapshot = reference_snapshot(&registry, encoder.space(), &["vox-reviewed-person"])?;
let matcher = SpeakerMatcher::new(snapshot)?; // No default thresholds.
// Optionally: matcher.with_calibration(explicit_qualified_policy)?
let mut speakers = SpeakerAttributor::new(encoder, matcher, capture_generation, 16_000)?;
let evidence = speakers.process(
    SpeakerObservation {
        generation: capture_generation,
        start_sample,
        end_sample: start_sample + mono_16k_pcm.len() as u64,
        activity: SpeechActivity::Speech, // Supplied by the host, not guessed here.
    },
    &mono_16k_pcm,
)?;
```

Without calibration, candidates remain diagnostic and `voice_id` stays empty.
`SpeakerCalibration` requires a policy ID, exact `matcher.reference_revision()`,
duration band, thresholds, runner-up margin and numerical guard. An ID is a caller
declaration, not empirical qualification. Evidence includes `voqx-speaker-evidence/1`
and a fingerprint of the actual policy contents, not just that ID. Changing model, preprocessing, references,
reference order or bank size invalidates calibration. Never substitute legacy
0.60/0.45 thresholds.

FASTDIAR windows contain 24,320–160,000 real mono 16 kHz samples. They reset encoder
state, preserve real silence, pad the final 80-ms frame and pool in FP64 before f32
storage. No prefix truncation or onset trimming occurs. Existing 256-dimensional
WeSpeaker profiles and experimental FP64 banks are not this 192-dimensional profile.

### Tunable short phrases (opt-in)

The 1.52 s floor above is an operational choice, not a model requirement or a
qualified accuracy boundary. For a caller-chosen band, use:

```rust,ignore
use voqx_recognition::fastdiar::FastdiarWindowEncoder;
// 300 ms to 10 s in the 16 kHz source clock; no implicit threshold defaults.
let encoder = FastdiarWindowEncoder::load_phrase(model_path, 4_800, 160_000)?;
```

`phrase_embedding_space(minimum_samples, maximum_samples)` returns the same exact
descriptor without loading inference. The structural floor is one 80 ms frame;
it is **not** a recommendation that 80 ms identifies a speaker reliably. Shorter
phrases contain less voice evidence. Choose minimum duration, confident/tentative
cosines, runner-up margin and numerical guard together on independent speakers and
sessions, then freeze the policy before evaluating the holdout.

The phrase engine slot is separate from the legacy window slot. Its model version
also records both bounds, so changing either invalidates canonical reference
profile compatibility and calibration. Rebuild references from consented sources
for the selected band; never relabel old vectors to make them appear compatible.
A changed operating point changes the evidence's calibration-content fingerprint.
Missing thresholds still yield `Uncalibrated`, not a nearest-name guess. These are
caller-controlled tuning parameters, not built-in production policy.

For local development, write a job JSON (paths resolve against the command's
working directory; WAVs must be mono 16 kHz, PCM16 or float32):

```json
{
  "minimum_ms": 300,
  "maximum_ms": 10000,
  "voices": [
    {"voice_id": "vox-consenting-person", "references": ["take-1.wav", "take-2.wav", "take-3.wav"]}
  ]
}
```

```sh
just speaker-phrase "$PINNED_GRAPH" job.json short-phrase.wav
just test-speaker-phrase-model "$PINNED_GRAPH"
```

The command emits JSON with diagnostic candidates, exact space, policy/reference
fingerprints, source duration and compute time. It reads only explicit local files;
there is no microphone, upload, enrollment write or saved audio/vector bank.
`voices` is a temporary reference selection using caller-supplied canonical IDs,
not a new registry. Unknown/overlap detection is not supplied by this speech probe.
To try a decision rule, add an explicit `thresholds` object containing `confident`,
`tentative`, `minimum_margin` and `numerical_guard`. All four are required; scores
are cosines, not probabilities. The result is marked `local-tuning-not-qualified`.
Short observations return `InsufficientAudio`; oversized input is rejected rather
than truncated or padded to reach the configured minimum. Only the final model
frame may receive structural padding, excluded from the source interval.

This is a native library/development interface, not an enabled eaRS setting. The
browser sandbox's older `:sandbox-phrase` space is also distinct; existing sandbox
references or thresholds are not silently migrated.

`recognition_profile(&reviewed_embeddings)` builds a recognition-only profile in
memory. The host owns human review, consent, identity selection and atomic
persistence. No audio retention, automatic enrollment or cloning is implied.
`reference_snapshot` selects exact canonical IDs; empty stays empty. Missing,
duplicate, wrong-version or malformed selected profiles fail rather than disappear.

`FastdiarEncoder` separately exposes continuous raw frames, fragmented input,
counters and explicit reset. These frames are not a calibrated continuous
recognition profile.

## Host responsibilities

- Share capture/resampling and provide activity evidence. Unknown activity and
  reported overlap abstain; this encoder does not automatically detect music,
  silence or overlap.
- Run inference off capture/UI threads. Coalesce same-span wake candidates: one
  observation is encoded once, then compared against all selected voices.
- Bound cadence/queues. Do not re-encode a 1.52-second window every 80 ms. A roughly
  one-second cadence overlaps these windows; two-second cadence leaves gaps and
  cannot accumulate stable labels. Observation/dwell delay is not compute time.
- Advance generation on Stop/Clear/new capture and call `reset(new_generation)`.
  Also reject delayed completions at the asynchronous host boundary. Immediate
  cancellation of in-flight synchronous ORT is not claimed.
- Bind evidence to its source interval only. Window matching does not prove speaker
  purity, authorize future PCM or solve switches/overlap inside that window.
- Rebuild the matcher/tracker and advance generation on changed/revoked profiles or
  policy changes. Snapshots are immutable; the library does not watch registry files.
- Surface errors as unavailable/invalid evidence, never permission to bypass a
  strict policy. Current-input failures clear stabilization; stale work cannot
  replace current state.

Only confident stabilized evidence contains a resolved `voice_id`. Tentative,
Unknown, Ambiguous, Uncalibrated, InsufficientAudio, IncompatibleModel, NoSpeech and
Overlap do not. Scores are not probabilities or authentication. Centroid cosine is
absent when references cancel. Snapshots hold at most 64 voices, 32 references per
voice and 1,048,576 components.

## Browser interface and verification

Developer tooling, not end-user installation:

```sh
just test-speaker
just check-speaker
just test-speaker-model "$PINNED_GRAPH" "$SYNTHETIC_FIXTURES"
just speaker-wasm-build "$WASM_PKG"
just speaker-contract-fixture > "$NATIVE_FIXTURE"
# Set CHROMIUM and PLAYWRIGHT to explicit installed development tools.
WASM_PKG="$WASM_PKG" NATIVE_FIXTURE="$NATIVE_FIXTURE" \
EVIDENCE_DIR="$NEW_EVIDENCE_DIR" just speaker-wasm-check
```

The model test requires explicit local `encoder.onnx`, `input.f32` and
`expected.f32` generated-signal files from the export probe. No weights are fetched
and no private audio is used.

Generated `SpeakerAttributor` accepts snapshot/calibration JSON and a BigInt capture
generation/dwell at construction. `observe(Float32Array, observationJSON)` uses the
native `SpeakerObservation` schema with exact-safe integer sample clocks and a
compatible browser embedding. It returns JSON; `reset` fences generations;
`referenceRevision` exposes
the exact bank fingerprint. Reference serialization is sensitive and must be an
explicit authorized operation. The binding performs no fetch, capture or upload.

The Chromium check executes real WASM matching/tracking and compares full native
evidence, fingerprints and stale errors. It is not encoder inference or speaker
accuracy. Independent real voices, physical devices, long continuous tracking,
clip playback and eaRS admission integration remain separate qualification/work.
