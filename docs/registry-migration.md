# Reviewed registry migration

The ML-free `registry_discovery::execute_migration` operation and CLI share the
same plan validation, staging, recovery and receipt logic. See ADR 0004 for the
writer-lock, no-overwrite and non-destructive abort trade-offs.

## Explicit local workflow

```sh
umask 077
voqx --registry /path/to/source registry migration-plan \
  --destination /path/to/destination --json > /private/reviewed-plan.json
# Inspect the entire plan, including both roots and all path/hash comparisons.
REV=$(jq -r .revision /private/reviewed-plan.json)
voqx registry migrate --plan /private/reviewed-plan.json --expected-plan "$REV" --dry-run
voqx registry migrate --plan /private/reviewed-plan.json --expected-plan "$REV" --yes
```

`--expected-plan` is the plan's normalized `revision`, not a hash of its JSON
formatting. It is SHA-256 over compact serde JSON of the plan with its `revision`
field null. Recompute/compare it before execution; changing the plan file is not
an update to the user's reviewed revision. This detects corruption/staleness,
not malicious same-user edits or allocation/consent provenance. Keep the reviewed
plan private and retain it for recovery. JSON input is bounded to 32 MiB; unknown
plan fields, including nested entry/probe fields, are rejected.

Missing `--yes` refuses before execution. `--dry-run` needs no confirmation and
creates no locks, directories, journal, config or models. It is point-in-time
read-only validation, not authorization or a promise a later apply will succeed.
It validates the copy plan regardless of `--action`; it does not simulate an
abort/resume state transition.
Roots come exclusively from the reviewed plan; `--registry`, `--timeout` and
`--parallel` are refused on migrate rather than silently ignored. JSON is the
default; `--yaml` selects YAML. No config/log initialization is needed.

## Copy and publication

Both roots are revalidated before writes and under their existing cooperative
writer locks, acquired in deterministic order. The source must still match the
entire reviewed inventory, including unreferenced assets, empty directories,
exact JSON and known source/recognition/synthesis/wake references. Existing
baseline destination entries must remain exact. Conflicts, special files, links,
unknown entries and missing/external references fail closed.

`.voqx-migration` is reserved, owner-private recovery/staging state inside the
destination. Each file is copied to a private stage, synced and read back against
its reviewed length/SHA-256 before a same-volume hard link publishes it without
replacement. Source payloads are never hard-linked into the destination. Unsupported
publication/filesystem operations fail without a replacing-rename or partial-copy
fallback. Registry JSON is published last, after asset and source revalidation;
the final inventory must match before a complete receipt is returned.

A complete matching destination is an idempotent `already_complete` outcome after
locked verification, not permission to overwrite later consumer changes. File
contents/IDs/profile bindings are preserved, not archival metadata such as ACLs,
ownership, modes, timestamps or extended attributes. New Unix payloads/staging
files use mode 0600 and newly created directories 0700; existing permissions are
not changed. Windows inherits the selected destination ACL and remains separately
runtime-qualified. Unix directory entries and file contents are synced where
supported; cooperative interruption/persisted-state tests do not establish every platform/filesystem's
power-loss durability. No current configuration/default root is changed.

## Recovery and non-destructive rollback

```sh
voqx registry migrate --plan /private/reviewed-plan.json --expected-plan "$REV" \
  --action resume --yes
voqx registry migrate --plan /private/reviewed-plan.json --expected-plan "$REV" \
  --action abort --yes
```

Partial work requires explicit `resume`, a valid `voqx-registry-migration-state/1`
journal bound to the same review revision, unchanged source/baseline entries and
only exact reviewed additions. A declared unfinished private staging file may be
discarded and recopied. Unknown recovery data is retained and refused, never
silently deleted. A normal cancellation/error releases locks; the library offers
cancellation callbacks at durable staging/publication checkpoints, not arbitrary
mid-I/O deadlines.

Abort before registry publication is rollback of intent: mark aborted and retain
partial copies/staging for inspection. It does not delete payloads or claim their
bytes were undone. Explicit resume can continue a revalidated aborted operation.
Abort after registry publication refuses; this command is not an unpublish or
active-store repair operation.

Hard process death can leave `.lock` files because current consumers use
create-new locks. There is no automatic stale-lock timeout/PID inference or unlock
operation. An operator must confirm all holders stopped before clearing locks;
then resume the same reviewed plan. Malformed/foreign journals, initial journal
creation interrupted before a valid state exists, and source changes require
operator inspection rather than speculative repair. Completed recovery metadata
is retained; before using that destination as a source of a later distinct
migration, explicitly archive its completed `.voqx-migration` state outside the
registry after verification. Do not copy or delete another operation's journal
implicitly.

## Receipts and scope

`voqx-registry-migration/1` reports `validated`, `completed`, `already_complete`,
`aborted`, `interrupted`, `refused` or `failed`, plus a bounded error, plan revision,
verified registry revision/file totals on complete outcomes, possible destination
effects and `recovery_required`. Source-payload-write and default-change flags are
always false. Recovery required means inspect, not a guarantee of automatic resume.
A non-success outcome exits unsuccessfully; lost output or a crash is uncertain
to the caller and must not trigger blind retries. Output schema:

```sh
voqx registry schema migration
```

Plans, journals, filenames and receipts can expose sensitive legacy identifiers
or linkable hashes; keep real artifacts private. This is neither opaque-ID
conversion nor encrypted custody, authentication, model qualification, deployment
or permission to migrate production data. Tests use synthetic data only:

```sh
just test-registry-migration
```

## Importing legacy tool-local stores

`voqx import` copies existing stores into the canonical registry. Import is
one-way: voqx can read a trnscrb registry, but no command writes trnscrb's format
back out, and the destination tool keeps owning its own assignments.

```sh
voqx import trnscrb --from ~/.local/share/trnscrb/speakers/registry.json --dry-run
voqx import spqx --from ~/.local/share/spqx/voices --dry-run
```

Import never merges by display name. A duplicate id, a name already owned by
another identity, or the same display name appearing twice in one source file is
refused and reported, so linking two stores stays an explicit act. `--dry-run`
runs the same checks, including reading the sample audio, and writes nothing.

### Vector validation

Before an identity is created, every reference vector is checked for a shared
dimension, finite values, nonzero length and an L2 norm within
`NORMALIZATION_TOLERANCE` (`1e-3`) of `1.0`. A non-empty centroid must match the
vector dimension and satisfy the same numerical rules. A centroid-only profile is
valid: the legacy store may have kept the summary vector and lost the audio.

These checks establish numerical usability only. They cannot establish which
checkpoint, frontend or pooling produced the vectors, and equal dimensions do not
mean compatible embedding spaces. `RecognitionProfile::add_samples` applies the
same rules and leaves the profile unchanged when it refuses an append.

### Provenance and exact-space matching

trnscrb v1 records one engine name (`speakrs-wespeaker`) with no checkpoint,
frontend or hash. The importer therefore keeps that name in `engine` (which keeps
legacy engine-name matching working) but records
`model_version = "unverified-trnscrb-v1"` and tags the profile with
`provenance:trnscrb-registry-v1` plus `source-embedding-model:<name>`.

Exact-space selection (`speaker_registry::reference_snapshot`) requires both the
engine and the model version to match the encoder's `EmbeddingSpace`, so imported
profiles stay out of it until they are verified or re-embedded. The importer does
not invent a version, does not normalize an unknown space into a known one, and
does not re-embed anything: re-enrollment from the retained sample audio is a
separate, explicitly scoped operation.

### Skipped audio

Missing and undecodable sample files are counted per speaker (`copied`, `missing`,
`unreadable`) and in the run total instead of being dropped silently. A skipped
sample never invalidates an otherwise usable vector-only profile; the stored
`embeddings` stay intact and only `samples` is shorter.

Synthetic coverage:

```sh
cargo test -p voqx-cli --test voices_import
```

