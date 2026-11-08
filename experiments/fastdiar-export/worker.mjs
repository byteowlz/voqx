import { StepRuntime } from './step-runtime.mjs';
import { sha256 } from './protocol.mjs';
import { poolFrames, fromBase64, toBase64 } from './frame-pool.mjs';
let runtime, audio, queue = Promise.resolve();

async function run({ reset = true, chunkSize = 997, samples = audio.length } = {}) {
  if (!Number.isSafeInteger(chunkSize) || chunkSize < 1 || !Number.isSafeInteger(samples) || samples < 0 || samples > audio.length) throw new Error('Invalid fixture range');
  if (reset) runtime.reset();
  const frames = [], timings = [], started = performance.now();
  for (let offset = 0; offset < samples; offset += chunkSize) {
    const result = await runtime.push(audio.subarray(offset, Math.min(samples, offset + chunkSize)));
    frames.push(...result.frames); timings.push(...result.timings);
  }
  const embeddings = new Float32Array(frames.length * 192);
  frames.forEach((frame, i) => embeddings.set(frame, i * 192));
  return { embeddings, timings, elapsedMs: performance.now() - started,
    processedSamples: runtime.processedSamples, pendingSamples: runtime.pending.length };
}
async function embed(args) {
  const bytes = fromBase64(args.pcm);
  if (await sha256(bytes) !== args.pcm_sha256 || bytes.length === 0 || bytes.length % (1280 * 4)) throw new Error('IPC PCM hash/alignment mismatch');
  const pcm = new Float32Array(bytes.buffer);
  if (pcm.length > 160000) throw new Error('Private diagnostic window exceeds 10s');
  runtime.reset();
  const started = performance.now();
  const result = await runtime.push(pcm);
  const compute_ms = performance.now() - started;
  const flat = new Float32Array(result.frames.length * 192);
  result.frames.forEach((frame, i) => flat.set(frame, i * 192));
  return { frames: toBase64(new Uint8Array(flat.buffer)), pooled: poolFrames(result.frames),
    pcm_sha256: args.pcm_sha256, pending_samples: result.pendingSamples, processed_samples: result.processedSamples, compute_ms };
}
const simpleCommands = new Map([
  ['embed', embed], ['run', run],
  ['memory', () => ({ wasmLinearMemoryBytes: self.__ortLinearMemoryBytes?.() ?? null })],
]);
async function command(type, args) {
  if (simpleCommands.has(type)) return await simpleCommands.get(type)(args);
  if (type === 'boot') {
    const manifest = await (await fetch('/manifest.json')).json();
    const pcm = await (await fetch('/input.f32')).arrayBuffer();
    if (await sha256(pcm) !== manifest.input_sha256) throw new Error('Fixture hash mismatch');
    audio = new Float32Array(pcm); runtime = await StepRuntime.create(manifest);
    return { loadMs: runtime.loadMs, stepSamples: manifest.step_samples, stateBytes: manifest.state_bytes, stateSlots: manifest.state.length };
  }
  if (type === 'invalid') {
    try { await runtime.push(new Float32Array([NaN])); } catch { return { rejected: true }; }
    throw new Error('Nonfinite PCM was accepted');
  }
  if (type === 'race') {
    runtime.reset();
    const job = runtime.push(audio.subarray(0, 1280));
    let resetRejected = false;
    try { runtime.reset(); } catch { resetRejected = true; }
    const result = await job;
    if (!resetRejected) throw new Error('Reset destroyed an in-flight step');
    return { resetRejected, processedSamples: result.processedSamples };
  }
  if (type === 'bench') {
    const queries = [];
    for (let i = 0; i < 20; i++) {
      const result = await run({ reset: true, chunkSize: 1280 * 19, samples: 1280 * 19 });
      queries.push(result.elapsedMs);
    }
    return { querySamples: 1280 * 19, queryMs: queries };
  }
  if (type === 'close') { await runtime.close(); return { closed: true }; }
  throw new Error('Unknown probe command');
}
self.onmessage = ({ data: { id, type, args } }) => {
  queue = queue.then(async () => {
    try {
      const result = await command(type, args);
      self.postMessage({ id, result }, result.embeddings ? [result.embeddings.buffer] : []);
    } catch (error) { self.postMessage({ id, error: String(error.stack ?? error) }); }
  });
};
