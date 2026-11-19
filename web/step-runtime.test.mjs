import { expect, test } from 'bun:test';
import { StepRuntime, MODEL_BYTES, MODEL_SHA256 } from './step-runtime.mjs';

const manifest = { format: 'fastdiar-export-probe/1', step_samples: 1280, sample_rate: 16000, embedding_dim: 192, onnx_sha256: MODEL_SHA256,
  state: Array.from({ length: 7 }, (_, i) => ({ input: `state_${i}`, output: `next_state_${i}`, field: ['audio_count', 'cms_count'][i] ?? `test_${i}`, dtype: i < 2 ? 'int64' : 'float32', shape: [], bytes: i < 2 ? 8 : 4 })), state_bytes: 36 };

test('hash/size checks fail before any session load, no download fallback', async () => {
  let loads = 0;
  const ort = { InferenceSession: { create() { loads++; throw Error('must not load'); } } };
  await expect(StepRuntime.create(ort, manifest, new Uint8Array(10))).rejects.toThrow();
  await expect(StepRuntime.create(ort, manifest, new Uint8Array(MODEL_BYTES))).rejects.toThrow();
  expect(loads).toBe(0);
});

test('invalid model output disposes tensors, faults until reset, close frees resources', async () => {
  let disposed = 0, released = 0;
  class Tensor { constructor(_, data) { this.data = data; } dispose() { disposed++; } }
  const output = { embedding: new Tensor('', new Float32Array(192).fill(NaN)) };
  for (const slot of manifest.state) output[slot.output] = new Tensor('', [0]);
  const session = { async run() { return output; }, async release() { released++; } };
  const runtime = new StepRuntime({ Tensor }, manifest, session, 0);
  await expect(runtime.push(new Float32Array(1280))).rejects.toThrow('Invalid embedding');
  expect(disposed).toBe(9); // audio input and every unadopted output
  await expect(runtime.push(new Float32Array(1280))).rejects.toThrow('faulted');
  runtime.reset();
  await runtime.close(); expect(released).toBe(1);
  expect(disposed).toBe(23); // seven old states, then seven new states
});
