# voqx-core

Pure-Rust library for the shared Voice Identity registry. See the
[repository README](../../README.md).

No ML dependencies: this crate owns the identity model, the on-disk format, cosine
matching, XDG path resolution, and atomic + lock-guarded persistence. Producing
embeddings from audio is a separate, optional concern.
