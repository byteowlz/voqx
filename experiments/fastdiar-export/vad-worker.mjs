// CPU-only Silero VAD, same model and input contract as eaRS `dictation::client_vad`:
// snakers4/silero-vad@066409f3 (MIT), 512 samples at 16 kHz plus 64 samples of context,
// recurrent state [2,1,128] per stream. Separate worker so it never queues behind the encoder.
import * as ort from '/ort/ort.wasm.min.mjs';
import { sha256 } from './protocol.mjs';
import { VAD_SHA, VAD_FRAME, VAD_CONTEXT } from './clips-core.mjs';

ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = '/ort/';
let session, queue = Promise.resolve();
const streams = new Map();
const SR = new ort.Tensor('int64', BigInt64Array.from([16000n]), []);

async function load() {
  const response = await fetch('/vad.onnx');
  if (!response.ok) throw new Error('Speech detector model unavailable.');
  const bytes = await response.arrayBuffer();
  if (await sha256(bytes) !== VAD_SHA) throw new Error('Speech detector model hash mismatch.');
  session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
  for (const name of ['input', 'state', 'sr']) if (!session.inputNames.includes(name)) throw new Error('Unexpected speech detector inputs.');
  for (const name of ['output', 'stateN']) if (!session.outputNames.includes(name)) throw new Error('Unexpected speech detector outputs.');
  return { bytes: bytes.byteLength };
}

/** Probabilities for consecutive 512-sample frames of one stream; history persists across calls. */
async function scan({ stream, reset, pcm }) {
  if (!session) throw new Error('Load the speech detector first.');
  if (!(pcm instanceof Float32Array) || pcm.length % VAD_FRAME || pcm.length > VAD_FRAME * 4096 || !pcm.every(Number.isFinite)) throw new Error('Expected whole finite 512-sample frames.');
  if (reset || !streams.has(stream)) streams.set(stream, { context: new Float32Array(VAD_CONTEXT), state: new Float32Array(256) });
  const s = streams.get(stream), probs = new Float32Array(pcm.length / VAD_FRAME), input = new Float32Array(VAD_CONTEXT + VAD_FRAME);
  try {
    for (let f = 0; f < probs.length; f++) {
      const frame = pcm.subarray(f * VAD_FRAME, (f + 1) * VAD_FRAME);
      input.set(s.context); input.set(frame, VAD_CONTEXT);
      const out = await session.run({ input: new ort.Tensor('float32', input.slice(), [1, input.length]), state: new ort.Tensor('float32', s.state.slice(), [2, 1, 128]), sr: SR });
      const p = out.output.data[0];
      if (!(p >= 0 && p <= 1)) throw new Error('Invalid speech probability.');
      s.state = out.stateN.data.slice(); s.context = frame.slice(-VAD_CONTEXT); probs[f] = p;
    }
  } catch (error) { streams.delete(stream); throw error; } // never reuse stale recurrent state
  return { probs };
}

self.onmessage = ({ data }) => {
  queue = queue.then(async () => {
    try {
      const result = data.type === 'load' ? await load() : data.type === 'scan' ? await scan(data.args) : data.type === 'drop' ? streams.delete(data.args.stream) : (() => { throw new Error('Unknown operation.'); })();
      self.postMessage({ id: data.id, result }, result?.probs ? [result.probs.buffer] : []);
    } catch (error) { self.postMessage({ id: data.id, error: { code: 'VAD_ERROR', message: error.message } }); }
  });
};
