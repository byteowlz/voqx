# voqx glossary

Authoritative terms for the shared voice registry. Keep this and
`docs/adr/0001-voice-identity-model.md` in sync as the model evolves.

**Voice Identity**:
The durable, engine-agnostic entity representing one voice. Has a stable `voice_id`, a
display name, aliases, reusable source recordings, and any number of engine-specific
profiles. Identity is by `voice_id`, never by display name.
_Avoid_: speaker record, voice name (those are per-tool views of an identity).

**voice_id**:
The stable internal identifier for a Voice Identity (`vox-<slug>` with a numeric suffix
on collision). Every consumer references this; renaming the display name never changes
it.

**Profile**:
An engine-scoped capability attached to a Voice Identity. Exactly one of: recognition,
synthesis, or wake. An identity may hold any subset.

**Recognition Profile**:
WeSpeaker-style per-sample embeddings plus an L2-normalized centroid, tagged with the
embedding model identifier. Answers "who spoke". Matching is cosine similarity;
classification returns evidence (Confident / Tentative / Unknown / InsufficientAudio /
IncompatibleModel), never authorization.

**Synthesis Profile**:
A TTS-engine-scoped set of references used to synthesize speech as this identity.

**Reference (Performance)**:
One recording within a synthesis profile: audio + exact transcript, optional
`reference_language`, optional `default` flag, free-form namespaced `tags`
(`emotion:happy`, `register:casual`, ...), and a human description. The registry stores
all references and never selects one.

**Wake Profile**:
A phrase + backend artifact reference (e.g. `rustpotter`) attached to an identity. The
wake tool owns the model files; the registry only records the profile.

**Source Recording**:
An engine-agnostic reusable recording (audio + optional transcript + language) kept on
the identity so recognition/synthesis/wake derivatives can be regenerated later.

**Voice Registry**:
The durable collection of Voice Identities under `$XDG_DATA_HOME/voice-registry`.
`registry.json` is authoritative (atomic, lock-guarded); heavy assets live under
`identities/<voice_id>/` referenced by registry-relative paths.

**Recognition evidence**:
The result of matching an embedding against the registry: a `voice_id`, a cosine score,
and a status. It is input to downstream command/authorization policy, not the policy
itself.
