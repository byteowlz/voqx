# Short-window ablation: 2026-10-30

Exploratory CPU measurements, not a production speaker filter. Tracked in
`trx-2mzg`; private evidence card: `voqx-short-window-ablation/v1` in the central
dataset registry. No installation, registry mutation, classifier training or
threshold deployment occurred.

## Controls and scope

- Pinned speakrs 0.4.2 graph and external weights; direct ORT CPU 1.31.1.
- Uniform-mask output reproduced three original wrapper outputs with maximum
  absolute component error `5.96e-8`.
- 113 source records from the screened recognition comparison; 2,034 combinations
  of three durations, prefix/onset alignment and three masks. Two invalid model
  outputs are explicit abstentions. Initial failed run retained separately.
- 42 additional full-enrollment vectors reuse the exact same short query vectors.
- Test set: 21 nominal owner queries and 32 synthetic non-target queries from eight
  designed voices. Channel variants and utterances are dependent; this is not 32
  independent people, nor evidence about children's voices. Speaker labels still
  need listening confirmation. Calibration/test grouping is inherited from the
  source card and is not a new independent cohort.

## Main result: full references, short queries

Mixed enrollment (one longer clip and six short takes), file-prefix queries,
nearest-sample scoring. Both references and queries use the indicated mask.

| Query duration | Uniform-mask AUC | Valid-audio-mask AUC | Activity-proxy-mask AUC |
|---|---:|---:|---:|
| 1.0 s | 0.138 | 0.945 | 0.716 |
| 1.5 s | 0.262 | 0.958 | 0.872 |
| 2.6 s | 0.865 | 0.912 | 0.854 |

This supports excluding padded frames from pooling as a candidate fix for
cross-duration mismatch. It does not prove channel invariance or a universal
improvement: with both references and queries cropped to one second, uniform
already achieved 0.967 versus 0.896 for valid masking. The single longer-reference
scenario remained poor in several conditions. Do not mix old unmasked profiles
with masked queries.

**Ranking improvement is not an acceptable operating point.** At the threshold
above the largest calibration-negative score, valid-mask 1.5 s queries accepted
20/21 target queries **and 10/32 synthetic non-target queries**. At 1.0 s those
counts were 19/21 and 9/32. Threshold transfer remains unqualified; none installed.
AUC is conditional on valid vectors; full counts and abstentions are in reports.

## Onset, activity, phrase matching

Some one-second file-prefix crops contained only about 120 ms of RMS-active audio.
A sparse activity mask produced nonfinite statistics-pooling outputs. The probe
now records those as abstentions, never zero-score non-owner evidence. RMS activity
is not a speech detector, and onset alignment is not a causal live VAD.

Same-language, exact-prompt template comparison covered 11 paired held-out target
queries. With valid masks and onset alignment, median same-phrase advantages were
+0.030 at 1.0 s, +0.006 at 1.5 s, and -0.006 at 2.6 s. Small and inconsistent;
this measures target consistency only, not rejection of non-target speakers saying
the wake phrase. No claim that phrase-dependent verification is disproven.

## Latency correction

On the producing Apple Silicon host, one resident CPU session with two threads:
model initialization 17 ms, warm fixed-ten-second graph inference median 293 ms,
p95 323 ms. This contradicts the earlier claim that nearly all call latency was
model loading. Masking changes pooling, not the fixed amount of encoder work.
Microphone capture, ASR, IPC and text insertion were not benchmarked here.

## Next gates

1. If exposed by voqx, make masking explicit and engine/profile-versioned; retain
   the old path for parity and abstain on invalid embeddings.
2. Qualify a speech-aligned query path and listener-confirmed target/non-target
   labels before selecting any pre-insertion filter threshold.
3. A genuine short-window or variable-length encoder is still worth benchmarking
   for latency. Denoising/dereverberation and alternate models have **not** been
   evaluated in this ablation.

Raw traces and private mappings remain local; aggregate results and reproducible
code only are in this repository.

## Reference-bank sweep: 2026-11-06

`bank_probe.py` replays the frozen registered embeddings, without new audio, vectors,
model inference, enrollment, training or threshold installation. It verifies the private
manifest and the three input hashes before scoring. Existing restricted dataset
`voqx-short-window-ablation/v1` remains immutable; outputs are aggregate experiment
reports outside that dataset, not a newly generated corpus.

The sweep has 168 comparisons: 1/1.5/2.6-second queries, prefix/onset alignment,
uniform/valid masks, nearest/centroid scoring and 1–7 references. The enrollment pool
contains the existing longer reference and six short takes, not new independent rooms
or the recent multi-microphone collection. Each size exhausts enrollment-only subsets;
bank selection maximises calibration target acceptance then calibration AUC, with
stable ID-order ties. Thresholds are above calibration negatives, never fitted on test.
Repeated searches over a small calibration set remain exploratory and may overfit.

Nearest-reference, valid mask, 1.5-second prefix queries:

| References | Target accepted | Synthetic non-target accepted |
|---|---:|---:|
| 1, selected on calibration | 20/21 | 10/32 |
| 2 | 20/21 | 10/32 |
| 3 | 20/21 | 9/32 |
| 4 | 20/21 | 9/32 |
| 5 | 20/21 | 10/32 |
| 6 | 20/21 | 10/32 |
| 7 | 20/21 | 10/32 |

More references alone do not provide a qualified filtering point. At an unchanged
0.60 score threshold, the single-reference and seven-reference selections accept
21/32 and 31/32 synthetic non-targets respectively, illustrating why a larger maximum
cannot reuse an uncalibrated threshold. Thresholds above calibration negatives reduce
those counts but still do not transfer safely to these test clips.

This is CPU/NumPy scoring of cached embeddings, not a fresh encoder timing or native/
browser inference-parity claim. Synthetic non-targets do not establish household FAR;
source-hash-disjoint owner clips still share a capture session. All previous label,
session, duration and synthetic-identity caveats remain. A different encoder or better
calibration/condition coverage must be tested rather than declaring bank size the fix.
Seven additional analytic/manifest-integrity tests pass, for 17 probe tests total.

## FASTDIAR-small CPU encoder: 2026-11-06

`fastdiar_probe.py` tests the causal encoder from Microsoft FASTDIAR, not its VAD/
clustering diarizer. Source is pinned to `f48bff53b9a6a0cce4539e94b6def95f609eaea4`;
released checkpoint SHA-256 is
`90be5da5e86645c39eceaedfd9f05cc7d35734d9105691da3e69b3e4ec47fb2a`.
The 2,382,724-parameter checkpoint is 10,616,074 file bytes, not a RAM measurement.
Code is MIT; product redistribution of the checkpoint is not approved by this spike.

The probe uses an existing Torch 2.14.0 / NumPy 2.5.4 interpreter without installing
or changing packages. Model parameters and buffers are checked as CPU-only, with two
Torch threads. The local checkpoint uses `weights_only=True`; network connection
attempts are refused during inference. Input source/config manifests and audio hashes
are checked against the same registered comparison/ablation data used above.
No new audio or embedding corpus is written; vectors stay in RAM. No voqx profiles,
thresholds, recognition defaults or production filtering change.

650 actual encoder calls cover 113 sources: 14 full-reference crops and 212 queries
at each requested duration. Frame mean and last-frame pooling are diagnostic identity
baselines. Both use fresh FASTDIAR references; 192-dimensional FASTDIAR vectors are
never compared with the 256-dimensional WeSpeaker profiles.

At 1.5 seconds, prefix queries, nearest-reference scoring:

| Encoder / pooling | References | Target accepted | Synthetic non-target accepted |
|---|---:|---:|---:|
| WeSpeaker valid-mask control | 7 | 20/21 | 10/32 |
| FASTDIAR frame mean | 7 | 20/21 | 3/32 |
| FASTDIAR last frame | 7 | 14/21 | 0/32 |
| FASTDIAR last frame | 1, calibration-selected | 18/21 | 0/32 |

Every bank and threshold uses calibration only. These settings are reported separately,
not selected for deployment from their test results. Zero observed accepts among 32
dependent synthetic clips is not a household FAR guarantee. Owner session correlation,
source-label limits and unqualified synthetic identity consistency still apply.

Native CPU timing on the producing Apple M2 Ultra, including streaming Python calls:

| Requested query window | Calls | Compute p50 | Compute p95 |
|---|---:|---:|---:|
| 1.0 s | 212 | 44.2 ms | 49.5 ms |
| 1.5 s | 212 | 59.1 ms | 65.5 ms |
| 2.6 s | 212 | 102.8 ms | 115.0 ms |

Model load was 45.5 ms after Torch import. Peak whole-process RSS was 302,940,160 bytes
(includes Python/Torch and the cohort; not browser or model-only RAM). These are compute
measurements, not microphone-to-action latency, OS CPU usage or low-end-client claims.

The encoder emits one vector per 80 ms; this probe batches 320 ms at a time. The
upstream 960 ms diarization policy delay is not imposed by this encoder-only test.
Final partial frames are explicitly zero-padded by less than 80 ms, not silently
truncated. Onset alignment uses the existing offline RMS proxy, not causal VAD.
Small and jumbo input-fragmentation controls agreed within maximum component error
`2.21e-6` and mean `1.80e-7`. Two full runs reproduced all scoring reports exactly.

20 probe tests pass. ONNX export with explicit streaming state, actual browser WASM
CPU parity/timing/memory and consented real non-target calibration remain open. This
is a promising alternate encoder, not an enabled speaker filter or complete diarizer.
See `trnscrb-59qf` and voqx `trx-2mzg`/`trx-j205`.

## Explicit-state ONNX / browser WASM CPU probe: 2026-11-06

`../fastdiar-export/` now exports the same pinned small checkpoint into a self-contained
ONNX opset-17, IR-8 research artifact. SHA-256:
`e7e3878d00afb742094eb22a8edc3a7f756d8ae1a10ec16200d95e56cf6ec3b1`.
The file has 10,665,298 bytes; the 50 explicit cache/counter tensors occupy 531,772 bytes.
There is no hidden stream state in the graph: one mono 16-kHz, 1,280-sample input yields
one 192-dimensional embedding plus next-state tensors. Counters are dynamic int64
scalars (BigInt in JavaScript). Fixed-size trace warnings do not imply variable-size
input support: the caller buffers arbitrary fragments to exactly 80 ms.

Reproduction entrypoints are `just export SOURCE CHECKPOINT NEW_OUT` and
`just browser ARTIFACTS ORT_DIST NEW_EVIDENCE` in that directory. Run `uv sync` first;
the isolated project pins CPU Torch 2.14.0, ONNX 1.19.1 and ORT 1.23.2 without changing
the earlier interpreter. The browser command requires explicit `PLAYWRIGHT` and
`CHROMIUM` paths and an existing ONNX Runtime Web 1.23.2 distribution. It starts only a
loopback allowlisted numerical fixture, never a product UI or microphone service.
All model binaries, numerical fixtures and receipts remain outside git.

Native ORT CPU and actual Chromium 156 WASM CPU, one inference thread, Apple M2 Ultra:

| Binding | Max embedding-component error vs upstream Torch | Mean error | Warm 80-ms step p50 / p95 |
|---|---:|---:|---:|
| Native ORT 1.23.2 | 5.34e-6 | 4.79e-7 | 4.11 / 4.31 ms |
| Browser ORT Web 1.23.2, WASM only | 5.54e-6 | 4.92e-7 | 6.86 / 7.24 ms |

The reference uses 80-ms steps, not the earlier 320-ms quality-probe batching.
240 frames (19.2 seconds) cover changing tones, digital silence, noise and DC offset;
these are numerical test fixtures, NOT new speaker-accuracy evidence. Both 997-sample
fragments and 19,001-sample fragments reproduce all frames and sample coverage after
reset. Partial frames remain buffered, never implicitly padded/dropped. Nonfinite
PCM is rejected and reset is fenced while inference is in flight.

For 20 fresh-state 24,320-sample (1.52-second) browser queries, compute p50 was 131.9 ms,
p95 135.1 ms. Model fetch/hash/session creation took 208.4 ms over loopback after worker
module startup; first inference took 27.4 ms. These are standalone compute/load times,
not network download, microphone-to-action or combined wake/STT/TTS scheduling claims.
WASM is explicitly the only provider; GPU is disabled. No private recordings enter the
browser and no external inference requests are allowed.

Memory was measured, not inferred from file size:

- A read-only observer of the unchanged Emscripten factory return records the actual
  `HEAPU8.byteLength`: 41,943,040 bytes (40 MiB), unchanged after both streams and two
  20-query bursts. This is linear-memory capacity, NOT total physical RAM or a soak.
- Worker CDP heap snapshots after explicit GC show roughly 2.3 MB JS heap and 1.84 MB
  ordinary backing storage. CDP backing storage **omits this WASM linear memory**.
- Summed owned-renderer RSS was roughly 0.43 GB before loading and 0.84 GB after the
  workload. It includes browser/JIT/debugging overhead and shared pages; it is not
  model-only RAM or unique physical memory. Small model/cache sizes do not establish
  a small whole-browser footprint. Low-memory/mobile qualification remains open.
- Session release frees state/session resources but does not shrink linear-memory
  capacity; worker termination ends the runtime lifetime.

An uninstrumented control and the memory-observed runs have identical numerical
results. The first browser attempt failed only because native stderr `[W]` output was
misclassified as a JS error. The exact ORT WASM CPU-vendor warning is now retained as a
warning; all other error-channel messages still fail. Numerical tolerances were not
relaxed. A user-agent-specific memory estimate timed out; no result is claimed from it.

3 new Python controls, 8 Bun tests and 13 real-browser assertions pass; the earlier
20 short-window tests still pass. `cargo check --locked` passes (offline initially
lacked cached public crates; normal locked dependency fetch resolved that). Repo-wide
`check-all` remains blocked at the unchanged formatting baseline, `trx-qhaj`.

This proves a local CPU portability path, not a production native Rust backend or an
enabled speaker filter. Next: replay the registered cohort through the actual bindings
and pooling/crop grid, recalibrate references/thresholds with numerical safety margins,
and obtain independent, consented real non-target/session evidence. Do not transplant
Torch thresholds that sit one float step above calibration negatives: tiny binding
errors can change boundary decisions. Phone/Safari/Firefox, long-running capture,
whole-app memory/scheduling and checkpoint redistribution remain unqualified.

## Registered speech replay across actual bindings: 2026-11-06

`../fastdiar-export/binding_cohort.py` and the optional `just cohort` browser mode now
replay 650 frozen windows per binding: 7 reference sources and 106 calibration/test
queries, with the same duration/alignment grid. This is 1,950 window inferences, NOT
1,950 independent recordings. The data remain the same registered real-owner /
eight-synthetic-voice cohort, with the same session dependence and exposed test set.
No fresh household or session-level false-admission evidence is added.

The plan is written before inference. Source/config/model hashes are checked, source
WAVs remain immutable, and only `plan.json` and aggregate `report.json` are retained.
No PCM, frame vectors or reference/query vectors are written. Private PCM moves through
same-host Python/Bun/Playwright/worker IPC, never an HTTP audio route or external
service. Automation debug logging is disabled. Inference uses only CPU; live registry,
enrollment, profiles, thresholds, filtering and defaults are untouched.

All bindings use the exported 80-ms grid with explicit final zero padding (<80 ms).
NN output is FP32; frame mean/last normalization and cosine scoring use FP64 controls.
Browser JavaScript pooling agrees with the host control within `1.39e-16`. Per binding,
2,674,560 embedding components were compared with Torch: native max/mean error
`1.21e-5 / 1.80e-7`; browser `1.21e-5 / 1.82e-7`. The original numerical tolerances
(max `1e-4`, mean `1e-5`) were not relaxed.

All 36 prespecified configurations select the same reference IDs and produce the same
test acceptance/abstention counts under **independent calibration** on each binding.
For the primary 1.5-second prefix, mean pool, seven-reference nearest setting:

| Binding | Target accepted | Synthetic non-target accepted |
|---|---:|---:|
| Torch, 80-ms steps | 20/21 | 3/32 |
| Native ORT CPU | 20/21 | 3/32 |
| Browser WASM CPU | 20/21 | 3/32 |

Porting a threshold unchanged is different. Using the Torch-selected bank and its
one-float-step-above-calibration-negative threshold produces 24 native and 22 browser
calibration decision disagreements summed across the 36 overlapping configurations;
these are repeated condition/query cells, NOT independent people or sessions. There
are no test decision disagreements. Maximum cosine-score drift is `7.82e-6`. In the
primary setting, the raw Torch threshold admits one calibration negative in WASM.

A fixed `+0.001` cosine margin was declared before this replay, not fitted on test.
It removes all measured cross-binding calibration/test decision disagreements relative
to **guarded** Torch. The primary setting remains 20/21 and 3/32. This margin is not
free: four other settings change test counts, including onset/mean/seven-nearest at
1.5 seconds dropping target acceptance from 15/21 to 13/21. It is a numerical
robustness diagnostic, not a qualified FAR gate or universally lossless adjustment.

Producing M2 Ultra, actual 1.5-second crops (explicitly padded to a full frame):

| Binding | Calls | Compute p50 / p95 |
|---|---:|---:|
| Native ORT CPU, one thread | 212 | 77.9 / 79.8 ms |
| Browser WASM CPU, one thread | 212 | 133.6 / 136.2 ms |

Torch at 80-ms cadence takes 154.7/172.0 ms; the earlier ~59 ms Torch result used
320-ms batching and is not a like-for-like timing comparison. Acquisition and IPC
transport time are excluded from these encoder-compute numbers. Browser linear-memory
capacity remains 40 MiB after the entire cohort. Total browser footprint is not 40 MiB:
owned-renderer RSS sums reach roughly 1.0 GB with browser/JIT/automation/shared-page
overhead. Mobile, whole-app contention and long-running physical capture remain open.

Two complete replays reproduce every score/decision report exactly. 7 Python tests,
11 Bun tests, the earlier 20 probe tests, 13 browser controls, exact PCM/hash/frame
coverage and all 650 per-binding parity checks pass. A regression test demonstrates
that a tiny binding delta defeats a one-float-step threshold but not the fixed-margin
control. Quality review removed added dispatcher complexity and duplicate test logic;
`cargo check` passes. The unrelated formatting baseline still blocks `check-all`.

The client bindings are no longer the main uncertainty on this cohort. Qualification
still requires independent real non-target voices and fresh source-group/session
holdouts, a frozen operating-point policy with owner-retention/FAR requirements,
binding-appropriate profiles and preprocessing, product integration, platform/memory
checks and exact checkpoint redistribution review. No filter is enabled.

## Live temporary-reference sandbox: 2026-11-06

The user requested live testing and explicitly approved an isolated page in the clean
voqx checkout, reusing the incumbent waqy look without touching the unrelated waqy /
Design Studio edits. `../fastdiar-export/live.html` uses the existing token/CSS and
capture/resampler assets read-only; it is not a redesign, persistent identity enrollment
or a production recognition backend. Studio catalog recording is deferred, not claimed.

`just live ARTIFACTS ORT_DIST CAPTURE_ROOT TOKEN_CSS RESAMPLER [PORT]` starts a foreground
loopback static service. Explicit certificate/key and a specific interface are required
for remote TLS; wildcard listeners are refused. The service accepts only GET/HEAD and
serves an asset allowlist plus health/model metadata. It has no audio upload, inference,
profile, action or persistence endpoint. Local development certificates may require a
browser trust exception; no trust store or shared proxy configuration is changed.

In the page, **Capture my reference** explicitly acquires one microphone and collects
3.04 seconds; up to seven normalized frame-mean vectors live in worker RAM. Queries use
the latest 1.52 seconds, at most every two seconds with no inference backlog. Scores are
raw best-reference and centroid cosine similarities, always **Uncalibrated**, never
accept/reject, identity confirmation or permission. Low energy is not labelled speech;
short/very-low-level inputs and missing references return separate states. Replay of
the reference speaker is not prevented.

Stop releases the microphone and cancels an unfinished capture; an already submitted
CPU operation drains. Clear fences stale operations, stops the microphone and removes
the bank. Refresh loses the entire session. Pending permission grants cannot be adopted
after Stop or overlapped by a new start. Files are decoded/resampled locally, with the
same worker bank/scoring path; no audio, vectors, cookies or local storage are saved.

Human controls and agent calls share `window.speakerTest`: `snapshot`, `start`, `stop`,
`captureReference`, `check`, `clear`, `referencePcm` and `queryPcm`. State reports readiness,
operation flags, reference durations and the latest evidence, not raw voiceprint vectors.
`just speaker-check ARTIFACTS REFERENCE.wav QUERY.wav` is the read-only native ORT file
alternative; inputs must be mono PCM16 at 16 kHz, and repeated `--reference` arguments
are available in `speaker_file.py`. Failures are structured JSON/nonzero; no registry
mutation occurs. Browser file decoding and native input validation are distinct paths.

Verification: 5 additional pure state/audio/bank tests, all earlier focused tests, and
22 real-WASM Chromium assertions using a synthetic microphone through the actual
AudioWorklet/resampler. These include exact 3.04-second capture, fresh live scores,
file parity, low/short audio, clear/stop/cancel, delayed permission/file-decode fencing, zero browser
storage and no upload/external requests. Native file CLI succeeds on the converted
synthetic fixture and rejects a 48-kHz source with JSON/exit 2. Synthetic desktop/phone
captures show the incumbent look without horizontal overflow. No physical microphone,
phone/Safari/Firefox or real-speaker quality pass is claimed.

The static design detector cannot resolve the runtime-served CSS; its heading-size
warning is not a runtime measurement. Its URL mode also failed contrast-screenshot
decoding. These checks are incomplete, not green. Real browser captures and computed
layout/semantic checks are the available evidence. `cargo check` and scoped code-quality
checks pass; the unchanged formatting baseline still blocks `check-all`.

The dev service is registered in owlznest. It runs only owned foreground Bun processes
under an isolated tmux supervisor, using user-owned development TLS; protected-certificate
sudo attempts timed out and no root service was started. Shared Caddy listeners and
unrelated checkouts remain untouched. Live speaker filtering is still off.

## Clip speaker test page: 2026-11-08

`../fastdiar-export/clips.html` (served at `/clips` with `--speaker-wasm`) adds named
multi-speaker references and clip/microphone attribution to the isolated sandbox. Encoder:
cold 1.52 s FASTDIAR windows every 0.5 s on WASM CPU. Matching/stabilisation: the
`voqx-recognition` WASM build of commit `1d478c5`. Exploratory defaults: match 0.50,
maybe 0.40, runner-up lead 0.05, stable 0.5 s, level gate -45 dBFS with at least half of
80 ms frames above it (an earlier 60 %/-42 dB rule discarded a third of in-turn windows on
natural pauses).

Synthetic demo `make_demo_clip.py` (fictional VoiceDesign voices de00/de01/de02, 12 turns,
70.8 s, one 3.04 s reference per voice from its first turn; reference windows are included
in scoring). Fully in-turn windows (whole 1.52 s inside one turn):

| Condition | Windows | Correct name | Wrong name | Unsure | Below level gate |
|---|---:|---:|---:|---:|---:|
| All three enrolled | 91 | 52 | 0 | 25 | 14 |
| Voice C not enrolled (its windows only) | 29 | n/a | 19 | — | — |

Whole-clip analysis: 139 windows in about 21 s wall time (about 130 ms CPU per window).
Threshold changes re-ran matching over cached embeddings in about 2 ms. The open-set row is
the important limitation: an unenrolled voice similar to an enrolled one is often named at
these defaults. Two of the three fictional voices share a "female, medium pitch" design, so
this is a hard case, but not a qualified false-accept rate. 22 Chromium checks pass on both
loopback and TLS; the earlier own-voice page's 22 checks still pass. Not real people,
physical devices, overlap detection or production filtering.

### Silero VAD gating: 2026-11-08

The clip page now replaces the loudness gate with Silero VAD in a separate CPU worker: the
model eaRS pins (`snakers4/silero-vad@066409f3`, SHA-256 `1a153a22…88e3`, MIT), 512-sample
frames with 64 samples of context and per-stream recurrent state, run over the whole clip
ahead of playback or streaming over the microphone. A window is compared with references only
when at least 50 % of its whole 32 ms frames have probability >= 0.5; at most 10 % abstains as
no speech, anything between as unknown. Both thresholds are user-adjustable and relabel from
stored probabilities. Same demo and references as above:

| Condition | Windows | Correct name | Wrong name | Named or "maybe" |
|---|---:|---:|---:|---:|
| All three enrolled, in-turn windows | 91 | 57 | 0 | — |
| Voice C not enrolled (its windows only) | 29 | n/a | 20 | — |
| Appended 5 s chords + 4 s noise bursts + 2 s silence | 19 | n/a | n/a | 0 (all no speech) |

The earlier rule (half the 80 ms frames above -45 dBFS) would have sent 17 of those 19
non-speech windows to the encoder. VAD does not address the open-set row: an unenrolled
person who is speaking is still often named as a similar enrolled voice. Synthetic chords
and noise are not real music or household audio; 24 Chromium checks pass on loopback and TLS.

### Per-phrase decisions and latency: 2026-11-08

Continuous mode needs half of a 1.52 s window to be speech and two agreeing windows, so names
appear about 1.5-2.5 s after speech starts and a single ~0.7 s wake word is never named.
Per-phrase mode instead segments the microphone with Silero (start after two speech frames,
end after eight non-speech frames = 256 ms, 4 s cap), pads the phrase by 0.15 s each side,
embeds the whole span (0.6 s floor) and calls the one-shot matcher (`evaluate`, voqx
`5c5ad17`; no tracker dwell). The phrase snapshot uses a distinct sandbox space
(`...:sandbox-phrase`, 0.6-10 s band), not the native 1.52 s window profile.

Fake Chromium microphone playing the synthetic demo, 4 decided phrases of 1.1-2.1 s per run:
median delay after the last speech sample reached the page 0.45 s (local) and 0.53 s (TLS),
range 0.37-0.54 s; pause detection is a fixed 0.26 s of that, encoder CPU 0.12-0.22 s.
Accuracy is not measured by this run: the fake microphone replays the same audio the
references came from. Real wake-word phrases, different microphones and unenrolled voices
remain to be tested by the owner. In eaRS/waqy the wake detector supplies the span directly,
so the 0.26 s pause wait would not apply to post-wake verification.
