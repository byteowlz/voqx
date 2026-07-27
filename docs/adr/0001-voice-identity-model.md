# ADR 0001: The Voice Identity model

Status: accepted
Date: 2026-07-24

## Context

Several byteowlz tools independently record and reason about a person's voice:

- **trnscrb** enrolls speakers with WeSpeaker embeddings and recognizes returning
  voices in diarized transcripts.
- **spqx** enrolls named cloning voices from a reference recording + transcript and
  synthesizes speech in that voice.
- A planned **wake** detector will enroll personalized wake/command phrases.
- **eaRS** and **Foxline** want to know *who is speaking* to personalize behavior and
  gate voice commands.

Left alone, each tool grows its own "speaker" or "voice" store keyed by a display
name. The same real person then exists as three unrelated records, and there is no
stable identifier to join recognition ("who spoke") to synthesis ("speak as them") to
wake ("their wake phrase"). Names are not identities: they collide, get renamed, and
differ per engine.

The vector representations are also **not interchangeable**. A WeSpeaker recognition
embedding and a Qwen3-TTS reference/x-vector live in different spaces and serve
different purposes. Unifying them into one blob would be wrong.

## Decision

Introduce one durable, cross-tool concept:

> **A Voice Identity is a stable, engine-agnostic entity with one display name, any
> number of aliases, and any number of engine-specific profiles. A profile is either a
> recognition profile, a synthesis profile, or a wake profile. An identity may have any
> subset of these capabilities.**

Identity is by stable **`voice_id`**, never by display name. Every consumer references
`voice_id`; display names and aliases are lookup conveniences only.

```
Voice Identity  (voice_id, display_name, aliases)
├── sources[]              reusable source recordings (engine-agnostic audio + transcript)
├── recognition_profiles[] WeSpeaker/etc embeddings + centroid  → "who spoke"
├── synthesis_profiles[]   engine references (performances)      → "speak as them"
└── wake_profiles[]        phrase artifacts                      → "they said the wake word"
```

### Profiles are engine-scoped and plural

- A **recognition profile** belongs to one embedding engine (e.g. `speakrs-wespeaker`)
  and records the model identifier so it can be invalidated/migrated when the model
  changes. It holds per-sample embeddings and an L2-normalized centroid. Matching is
  cosine similarity; classification uses configurable thresholds and returns
  **evidence, not authorization**.
- A **synthesis profile** belongs to one TTS engine (e.g. `spqx-qwen3`) and contains
  one or more **references** ("performances"). Each reference has its own audio +
  exact transcript, an optional `reference_language`, an optional `default` flag,
  free-form `tags` (namespaced `dimension:value`, e.g. `emotion:happy`,
  `register:casual`), and a human description. The registry stores *all* references and
  exposes them; **it never encodes a closed emotion taxonomy and never picks one** —
  the caller (or a model) selects.
- A **wake profile** references a phrase and a backend artifact (e.g. `rustpotter`).
  The wake tool owns the model artifacts; the registry only records that the profile
  exists and which `voice_id` it belongs to.

### Capabilities are independent

An identity may be recognition-only (a meeting participant we can label but do not
clone), synthesis-only (a fictional/preset voice we never expect to hear on a mic), or
both, or neither-yet. Matching display names across capabilities does **not** imply the
same identity; linking is always an explicit operation.

### The registry is a library + data format, not a daemon

`voqx-core` is a pure-Rust library over a durable on-disk format. It has **no ML
dependencies** so lightweight consumers (spqx) can link it just to resolve a
`voice_id`. Producing embeddings is a separate, optional concern (`voqx-recognition`,
feature-gated speakrs). There is no required background process; a `voqx` CLI is an
optional management/enrollment surface.

### Canonical location

```
$XDG_DATA_HOME/voice-registry/            (default ~/.local/share/voice-registry)
├── registry.json                         authoritative: all identities + profiles + vectors
└── identities/<voice_id>/                 heavy assets referenced by relative path
    ├── sources/
    ├── recognition/<engine>/
    ├── synthesis/<engine>/
    └── wake/<backend>/
```

`registry.json` is the single authoritative, atomically-written, lock-guarded file
(embeddings inline as JSON, like trnscrb today). Audio and model assets live under
`identities/<voice_id>/` and are referenced by **registry-relative** paths so the whole
directory is portable.

### Ownership

- **voqx** — owns the Voice Identity, guided recording/enrollment, reusable sources,
  consent, and the registry format/migrations. Optional `voqx-recognition` produces
  WeSpeaker embeddings.
- **trnscrb** — a consumer + enrollment surface for recognition evidence.
- **spqx** — a consumer of synthesis references; links `voqx-core` only (no ML deps).
- **wake** — owns wake model artifacts; each wake profile references a `voice_id`.
- **eaRS / Foxline** — consumers that receive `{ voice_id, score, status }` and
  synthesis selections; they never parse another tool's private store.

## Consequences

- We migrate trnscrb's speaker registry into voice-registry first (preserving ids),
  then import spqx named voices as synthesis-only identities. No auto-merge by name.
- eaRS `trx-2stp.4` (speaker gating) and `trx-2stp.6` (wake) consume the voqx
  `voice_id` contract instead of coupling to trnscrb internals or standing up a third
  identity store.
- Selection of a synthesis reference (by id, by `reference_language` + tags, by a
  model interpreting a natural-language style request, or by default) lives in the
  caller. The registry stays a dumb, deterministic substrate — better models select
  better references from the same durable audio/transcripts/tags without schema change.
- Recognition thresholds and cosine scores are **evidence**. Command/authorization
  policy (fail-closed on unknown speaker, confirmation for destructive actions) lives
  downstream in eaRS/Foxline. Voice similarity is not authentication (replay, synthesis,
  short-utterance limits).
