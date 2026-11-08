# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Clip speaker test page

- Per-phrase microphone mode: Silero segments live speech (onset after 64 ms, end after a
  256 ms pause, 4 s maximum) and each phrase gets one decision from the new one-shot WASM
  `evaluate`, shown with its measured delay split into pause detection and CPU time. Built
  for the wake-word case, which the 1.5 s continuous windows can never name.
- "Record wake word ×3" and "Record sentence (3 s)" take references from the live
  microphone, so references and tests share the same channel.
- Adds `/clips` to the isolated speaker sandbox: name up to eight people, take 3-second
  references from a playing clip or the microphone, then see who is speaking during
  playback, in a whole-clip timeline or live from the microphone. Exports JSON/WebVTT.
- Uses the FASTDIAR CPU/WASM worker for embeddings and the voqx Rust/WASM matcher/tracker
  for decisions. Embeddings are cached per 0.5 s window, so references and exploratory
  thresholds re-run matching instantly.
- Speech detection uses Silero VAD in a second CPU worker, with the same pinned model and
  512+64-sample input contract as eaRS `client_vad`. Windows that are not mostly speech never
  reach the speaker encoder, so music, noise and silence cannot produce a name. References
  must also be mostly speech.
- Synthetic demo (three fictional voices, 70.8 s): 57/91 fully in-turn windows named, none
  wrong. 11 s of appended synthetic chords, noise bursts and silence: 19/19 windows rejected
  as no speech (the earlier loudness gate would have analysed 17). With one voice not
  enrolled, 20 of its 29 windows were named as another enrolled voice: open-set false
  matches on real speech remain the main limitation.
  Still no persistence, uploads, saved profiles or qualified operating point.

### Embeddable speaker library

- Adds optional one-thread native CPU FASTDIAR with pinned-graph verification, explicit
  streaming state and independently reset, versioned recognition windows. No subprocess,
  microphone, model download or new daemon is required.
- Adds portable bounded multi-person scoring and timestamped stabilization, read-only
  canonical profile adapters and actual Rust/WASM bindings. Missing calibration abstains;
  bank/model/policy/generation changes cannot silently inherit a speaker label.
- Preserves legacy WeSpeaker behavior/profiles and supports FASTDIAR-only builds without
  WeSpeaker/OpenBLAS. Tests actual native encoder parity and native/WASM evidence. No ASR
  filtering, enrollment persistence or real-speaker qualification is enabled.
  See `docs/speaker-library.md` and ADR 0003.

### Live speaker-test sandbox

- Adds an isolated browser CPU test: capture temporary own-voice references and inspect
  uncalibrated similarity from microphone or local files, without uploads or persistence.
- Leaves unknown similarity bars empty rather than painting a numeric midpoint.
- Reuses the incumbent recorder look/assets read-only. Human controls and semantic browser
  operations share one temporary bank; a native file CLI provides the same scoring path.
- Tests actual WASM, synthetic microphone capture, stop/clear/late-permission fencing and
  read-only serving. Physical-device and real-speaker qualification remain open; no saved
  voice profiles, actions, production filtering or defaults change.

### Registered FASTDIAR binding calibration replay

- Replays the frozen speech cohort through Torch, native ORT and real browser WASM with
  source/hash/coverage checks and transient same-host IPC; no saved audio/vector corpus.
- Independently calibrated banks and test decisions agree across all 36 configurations.
  Unchanged threshold transfer still flips calibration decisions; a predeclared numerical
  margin is reported with its recall trade-offs, not installed as an admission policy.
- Keeps all live profiles/defaults/filtering unchanged; real non-target and session-level
  qualification remain open. Clears completed probe timeout timers to avoid retaining IPC
  promise chains until their timeout.

### FASTDIAR ONNX / WASM portability probe

- Adds a self-contained, explicit-state 80-ms ONNX export and a CPU-only browser-worker
  probe with hash checks, buffering/reset/error controls and native/browser parity.
- Measures single-thread inference, WASM linear-memory capacity and browser RSS separately;
  model file size is not a RAM claim. Synthetic numerical fixtures do not qualify speaker
  accuracy, mobile use or production filtering. No live backend or profile changes.

### FASTDIAR-small CPU diagnostic

- Adds a pinned-checkpoint/source experiment using the causal FASTDIAR encoder on CPU, with
  offline inference, safe weight loading, sample-hash checks and small/jumbo chunk parity.
- The pilot improves 1.5-second speaker discrimination on the existing synthetic-impostor
  cohort; results, timing, memory and inherited limitations are in the short-window report.
  This is not a new live recognition profile, browser backend, diarizer or enabled filter.

### Reference-bank diagnostic

- Adds a read-only CPU scoring sweep over registered cached speaker embeddings. It compares
  1–7 enrollment references, chooses banks/thresholds on calibration only, verifies input hashes,
  and preserves abstentions. No live profiles, model defaults or thresholds are changed.
- The first sweep shows that adding references alone does not qualify a short-window speaker
  filter; see `experiments/short-window/RESULTS.md`. Browser inference and real household
  false-accept rates are not measured by this replay.

### Capability discovery

- `voqx capabilities --json` reports a versioned build contract before any
  configuration, registry or model initialization. Consumers can distinguish
  recognition-enabled builds and supported preprocessing without scraping help
  or assuming that file presence/package version means compatibility. Models,
  profiles and calibration remain explicitly unchecked. See `docs/capabilities.md`.

### Opt-in recognition pooling

- Enrollment and identification accept `--preprocessing uniform|valid-v1`.
  Uniform remains the default. Valid-v1 excludes waveform padding from pooling
  under a separate `speakrs-wespeaker-valid-v1` profile key; it never matches or
  appends to old uniform profiles. See ADR 0002.
- Libraries select the contract with `Embedder::with_preprocessing` and obtain
  the matching profile key from `engine_id()`. No automatic registry migration.
- Valid-v1 embedding/enrollment requires one second or the model minimum,
  whichever is larger. Silence/nonfinite audio and invalid model vectors are
  rejected rather than persisted or treated as negative-speaker evidence.
- Fixed-window encoder cost and uncalibrated thresholds are unchanged. This is
  not an enabled speaker filter or a latency improvement.

### Added

- **Voice Identity model** (`docs/adr/0001-voice-identity-model.md`): a stable
  `voice_id` with any number of engine-scoped recognition, synthesis, and wake
  profiles. Identity is never keyed by display name.
- `voqx-core`: durable registry at `$XDG_DATA_HOME/voice-registry` with atomic,
  lock-guarded writes, cosine matching, and recognition *evidence*
  (Confident/Tentative/Unknown/InsufficientAudio/IncompatibleModel). No ML deps.
- `voqx-recognition`: optional WeSpeaker embeddings via `speakrs`.
- `voqx` CLI: `identities`, `enroll`, `references`, `identify`, and `import`.
- Importers for existing stores: `voqx import trnscrb` (preserves `spk-*` ids and
  centroids verbatim so matching behaves exactly as it did) and
  `voqx import spqx` (named voices become synthesis references). Neither merges by
  display name; collisions are reported and skipped so linking stays explicit.

### Fixed (recognition scoring)

- Recognition ranks identities by their nearest valid enrolled sample instead of
  collapsing recording conditions into one averaged centroid. Results expose
  `score_sample` and `score_centroid` for the same winning identity; `score` is the
  nearest-sample score. Legacy profiles without embeddings retain centroid scoring.
- Empty, zero, mismatched-dimensional, and nonfinite vectors cannot become matches.
- Audio identification reports input/observed duration, padding, truncation, and
  its operational minimum. Below one second (or the loaded model's larger minimum),
  and for exact digital silence, results are `insufficient_audio` without inference.
  This floor is not a measured accuracy guarantee or speech detector.
- The pinned speakrs 0.4.2 backend observes only the leading 10 seconds, padding
  shorter windows. Unobserved tails are now explicit; no full-input consistency,
  overlap detection, liveness, or authentication is implied.
- Scores and legacy default thresholds remain **uncalibrated** for duration,
  channel, enrollment count, and nearest-sample scoring (`calibration_band: null`).
  Enrolling only one condition cannot recover an absent condition. Offline analytic
  vector and generated-PCM tests prove scoring/orchestration, not acoustic accuracy;
  held-out licensed/consented speech validation and cohort calibration remain open.

### Notes

- `ort-sys` prebuilt downloads fail here under its default `tls-native`; the
  `tls-rustls` backend is selected explicitly in `voqx-recognition`.


### Added (registry UI)

- `voqx-core::analysis`: leave-one-out sample scoring and a centroid similarity
  matrix, so contaminated enrollments can be found. Pure cosine math over stored
  embeddings — no audio and no model required. Thresholds are inputs, not policy.
- `RecognitionProfile::remove_sample`: prune a sample and recompute the centroid.
- `voqx-api`: `/api/registry`, `/api/identities/{id}`, `/api/analysis`,
  `/api/audio` (traversal-guarded), plus rename and sample-drop mutations that go
  through the same lock-guarded `RegistryStore::edit` as the CLI.
- `ui/`: React + Vite cleanup workbench themed with `@byteowlz/design-system`.
  Identities ranked by risk, similarity heatmap, per-sample scores with audio
  playback, and one-click pruning. `just ui-api` + `just ui`.
