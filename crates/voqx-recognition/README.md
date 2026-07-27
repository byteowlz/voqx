# voqx-recognition

WeSpeaker embedding backend for the voqx Voice Identity registry.

This crate is the **optional, ML-dependent** half of recognition: it turns mono
16 kHz audio into speaker embeddings (via `speakrs` / ONNX Runtime) and matches
them against enrolled identities. Consumers that only need to *resolve* a
`voice_id` should depend on `voqx-core` alone and avoid this crate's native
dependencies.

Embeddings produced here are tagged `speakrs-wespeaker` so profiles can be
invalidated or migrated if the model changes.
