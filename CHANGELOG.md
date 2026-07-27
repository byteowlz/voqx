# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

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
