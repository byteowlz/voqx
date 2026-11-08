# Short-window speaker embedding probe

CPU-only diagnostic for the pinned speakrs 0.4.2 WeSpeaker export. No changes to
recognition defaults, installed binaries, registry, enrollment or thresholds.
All audio, vectors, labels and reports belong outside git.

## Questions

1. Does weighting the zero-padded region affect short-clip discrimination?
2. Are file-prefix crops measuring speech, or mainly leading silence?
3. Does matching prompted phrase and language improve target-template consistency?

The graph takes `[1, 1, 160000]` PCM and `[1, 589]` weights. The default speakrs
`embed()` call pads to ten seconds and passes ones everywhere. This probe keeps
that exact waveform/graph and varies only the mask: `uniform`, `valid` (exclude
padding), or `activity` (20 ms RMS proxy with 50 ms hangover). The graph resizes
weights with nearest/asymmetric/floor interpolation. Masks do not remove padded
context from convolution or feature normalization; this is not a variable-length
export. Sparse activity masks can produce nonfinite embeddings: those trials
abstain and remain in the denominators, rather than becoming negative evidence.

## Run

```sh
uv sync --locked
just test
just run /private/source /private/config.json /cached/model.onnx /private/new-run
uv run --offline python sanity.py --model /cached/model.onnx \
  --references /private/original-vectors.jsonl --out /private/sanity.json
uv run --offline python phrase_probe.py --config /private/config.json \
  --vectors /private/new-run/vectors.jsonl --phrases /private/phrase-map.json \
  --out /private/phrase-report.json
```

The model and its `.onnx.data` must already be present. Inference uses CPU only,
two threads by default, one resident session. A run directory must not exist;
failed runs remain evidence. Dependency installation may use the package index,
but inference never downloads models or sends audio anywhere.

Config uses the registered comparison dataset's `records` and `scenarios` shape:
records have `id`, relative `path`, `sha256`, `source_sha256`, `speaker`, `role`;
scenarios have `name`, `enrollment: {owner: [ids...]}`, and `queries: [ids...]`.
Source paths/hashes and enrollment/query source disjointness are checked.

Output includes per-record vectors, offsets, activity duration **proxies**,
abstention reasons, CPU inference timings and model/config/script hashes.
For every mask and crop, both enrollment and query use the same preprocessing.
AUC is conditional on valid vectors; abstentions are reported separately. The
threshold is chosen above calibration negatives only, never from test data.
This zero-calibration-accept threshold is an exploratory operating point, not a
false-accept guarantee or a proposed deployment threshold.

`phrase_probe.py` matches exact prompted wording/language using a private metadata
map. It compares averages of single-template scores (not maxima over unequal
reference counts). This measures target consistency, **not** rejection of other
speakers saying the same phrase. No classifier is trained.

## Limits

- Energy and pitch do not establish who is speaking. Source speaker labels may
  still need listening confirmation.
- Synthetic negative speakers do not establish household false-accept rates.
- Dependent duration crops and channel transforms are not independent trials.
- File duration includes silence. An onset crop is an energy-aligned diagnostic,
  not a causal streaming-VAD implementation or a latency measurement.
- Timing covers the graph on this host, not microphone-to-keystroke latency.
- Different preprocessing changes the embedding distribution. Never mix masked
  queries with old unmasked recognition profiles without qualification.

Tracked in voqx `trx-2mzg`; restricted derived data is registered separately.
