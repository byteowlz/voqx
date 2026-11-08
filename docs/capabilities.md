# Executable capability discovery

```sh
voqx capabilities --json
```

This command runs before configuration or registry initialization. It does not
read configuration, create user directories, open a registry, load/download a
model, start a service or use the microphone. JSON is the default; `--yaml` is
also supported. Build version alone is not a compatibility test.

The versioned response includes:

```json
{
  "protocol": "voqx-capabilities/1",
  "version": "0.1.0",
  "recognition": {
    "compiled": true,
    "identify_json": true,
    "enroll_audio": true,
    "enroll_json": false,
    "preprocessing": ["uniform", "valid-v1"]
  },
  "registry_inspection_json": true,
  "microphone_capture": false,
  "model_state": "not_checked",
  "registry_state": "not_checked"
}
```

A registry-only build reports recognition unavailable and no preprocessing
variants. These are build capabilities, not attestation of a loaded checkpoint,
model availability, an enrolled voice, calibration, or permission to record.
Unknown protocol revisions should be treated as unsupported. An old binary may
exit nonzero because it does not recognize this command; consumers must not
misreport that as an empty registry.

`enroll_json` and `microphone_capture` are currently false: structured enrollment
receipts and reference recording/setup remain follow-up work. The existing
browser manager is for registry inspection/cleanup, not a complete recorder.
The setup contract is tracked in `trx-qcb7`; eaRS consumes it in `trx-pbfx`.
