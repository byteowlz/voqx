# voqx

The shared **Voice Identity** registry for byteowlz voice tools.

A Voice Identity is a stable, engine-agnostic entity with one display name, any number
of aliases, and any number of engine-specific profiles:

- **recognition** profiles (WeSpeaker embeddings) — *who spoke*
- **synthesis** profiles (TTS references/performances) — *speak as them*
- **wake** profiles (personalized wake/command phrases) — *they said the wake word*

Identity is by stable `voice_id`, never by display name. Profiles are engine-scoped:
vectors from different models, preprocessing contracts or dimensions are never mixed.

## Crates

- `voqx-core` — pure-Rust registry: identities, profiles, cosine matching, XDG paths,
  atomic + lock-guarded persistence. **No ML dependencies**, so lightweight consumers
  (spqx) can link it just to resolve a `voice_id`.
- `voqx-recognition` — optional WeSpeaker embeddings via `speakrs`/ONNX Runtime. Only
  needed to *produce* embeddings or identify audio.
- `voqx-cli` — the `voqx` binary: identities, enrollment, references, identification,
  and importers. There is no required daemon.

## Usage

```bash
# Migrate existing tool-local stores (never merges by display name)
voqx import trnscrb --dry-run     # speakers -> recognition profiles (ids preserved)
voqx import spqx --dry-run        # named voices -> synthesis references

# Inspect
voqx identities list
voqx identities show spk-tommy
voqx references list snake

# Attach a synthesis reference to an existing identity (explicit link)
voqx enroll snake --id spk-snake \
  --audio reference.wav --transcript-file reference.txt \
  --synthesis spqx-qwen3 --language en-US --tag tone:gravelly

# Recognition evidence (never authorization)
voqx identify utterance.wav --json
```

Build without native ML dependencies: `cargo build -p voqx-cli --no-default-features`.

## Registry UI

A small web UI for enrollment-quality analysis and cleanup — find contaminated
enrollments, listen to the sample, prune it, and watch the centroid recover.

```bash
just ui-api /tmp/vr    # terminal 1: API on :3111 (omit the path for the real registry)
just ui                # terminal 2: UI on :5175
```

It scores every enrolled sample **leave-one-out** — against its identity's centroid
rebuilt *without* that sample. A mislabeled snippet then shows a low score and matches
a different identity, which the plain score hides. Removing a sample recomputes the
centroid through the same lock-guarded path the CLI uses; the audio file is kept.

Themed with [`@byteowlz/design-system`](../design-system) (base24 slots + role layer).

## Storage

```
$XDG_DATA_HOME/voice-registry/           (default ~/.local/share/voice-registry)
├── registry.json                        authoritative: identities + profiles + vectors
└── identities/<voice_id>/                heavy assets, referenced by relative path
```

`registry.json` and enrolled audio are biometric-sensitive local data: never committed,
never logged.
