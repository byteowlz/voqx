import { StepRuntime } from './step-runtime.mjs';
import { poolFrames } from './frame-pool.mjs';
import { MODEL_SHA, TemporaryBank, preparePcm, QUERY_SAMPLES, REFERENCE_SAMPLES } from './live-core.mjs';
let runtime, bank, queue = Promise.resolve();

// Independent cold window: reset state, pad the final frame, FP64 frame mean, unit-normalized.
async function encode(pcm, limit, minimum) {
  const prepared = preparePcm(pcm, limit, minimum);
  if (prepared.status !== 'Ready') return { prepared };
  runtime.reset();
  const started = performance.now();
  const result = await runtime.push(prepared.pcm);
  const computeMs = performance.now() - started;
  if (result.pendingSamples || result.processedSamples !== prepared.pcm.length) throw new Error('Incomplete audio processing.');
  const vector = poolFrames(result.frames).mean;
  if (!vector) throw new Error('InsufficientAudio');
  return { prepared, vector, computeMs };
}

async function load() {
  const response = await fetch('/manifest.json');
  if (!response.ok) throw new Error('Model manifest unavailable.');
  const manifest = await response.json();
  if (manifest.onnx_sha256 !== MODEL_SHA) throw new Error('IncompatibleModel');
  bank = new TemporaryBank(manifest.onnx_sha256);
  runtime = await StepRuntime.create(manifest);
  return { ...bank.snapshot(), modelBytes: manifest.model_bytes, stateBytes: manifest.state_bytes, loadMs: runtime.loadMs };
}

async function command(type, args = {}) {
  if (type === 'load') return await load();
  if (!runtime) throw new Error('Load the model first.');
  if (type === 'inspect') return bank.snapshot();
  if (type === 'clear') { bank.clear(); runtime.reset(); return bank.snapshot(); }
  if (type === 'embed') {
    // Matching happens in the caller (voqx WASM); this returns only the vector.
    // Phrase spans may be shorter than the 1.52 s window; never below 0.6 s.
    const { prepared, vector, computeMs } = await encode(args.pcm, 160000, Math.max(9600, Math.min(args.minimum ?? QUERY_SAMPLES, QUERY_SAMPLES)));
    return vector ? { status: 'Ready', vector, samples: prepared.samples, paddedSamples: prepared.paddedSamples, rms: prepared.rms, computeMs } : prepared;
  }
  if (!['reference', 'score'].includes(type)) throw new Error('Unknown operation.');
  if (type === 'score' && !bank.references.length) return bank.score(null);
  const { prepared, vector, computeMs } = await encode(args.pcm, type === 'reference' ? REFERENCE_SAMPLES : QUERY_SAMPLES);
  if (!vector) return { ...prepared, referenceCount: bank.references.length };
  const evidence = type === 'reference' ? bank.add(vector, prepared.samples) : bank.score(vector);
  return { ...evidence, samples: prepared.samples, paddedSamples: prepared.paddedSamples, rms: prepared.rms, computeMs };
}
self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try { self.postMessage({ id: data.id, result: await command(data.type, data.args) }); }
    catch (error) { self.postMessage({ id: data.id, error: { code: 'SPEAKER_TEST_ERROR', message: error.message } }); }
  });
};
