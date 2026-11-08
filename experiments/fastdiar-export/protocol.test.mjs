import { test, expect } from 'bun:test';
import { validateManifest, zeroData } from './protocol.mjs';
const fixture = () => {
  const fields = ['audio_buffer', 'preemph_last', 'audio_sum', 'audio_sumsq', 'audio_count', 'cms_sum', 'cms_count'];
  const state = fields.map((field, i) => ({ input: 'state_' + i, output: 'next_state_' + i, field,
    dtype: field.endsWith('_count') ? 'int64' : 'float32', shape: [], bytes: field.endsWith('_count') ? 8 : 4 }));
  return { format: 'fastdiar-export-probe/1', sample_rate: 16000, step_samples: 1280, embedding_dim: 192,
    onnx_sha256: 'a'.repeat(64), state, state_bytes: 36 };
};
test('valid fixed CPU tensor contract', () => expect(validateManifest(fixture()).state.length).toBe(7));
test('incompatible rate, dimension, step or digest is rejected', () => {
  for (const mutation of [{ sample_rate: 48000 }, { embedding_dim: 256 }, { step_samples: 5120 }, { onnx_sha256: 'bad' }])
    expect(() => validateManifest({ ...fixture(), ...mutation })).toThrow();
});
test('duplicate bindings and float counters are rejected', () => {
  let m = fixture(); m.state[1].input = m.state[0].input; expect(() => validateManifest(m)).toThrow();
  m = fixture(); m.state[4].dtype = 'float32'; m.state[4].bytes = 4; expect(() => validateManifest(m)).toThrow();
});
test('oversize, negative and mismatched state is rejected', () => {
  for (const shape of [[-1], [1_000_001], [0]]) {
    const m = fixture(); m.state[0].shape = shape; expect(() => validateManifest(m)).toThrow();
  }
  const m = fixture(); m.state_bytes++; expect(() => validateManifest(m)).toThrow();
});
test('scalar int64 reset data uses BigInt, not lossy floats', () => {
  const data = zeroData({ dtype: 'int64', shape: [] });
  expect(data).toBeInstanceOf(BigInt64Array); expect([...data]).toEqual([0n]);
});
test('float reset data has exact bounded shape', () => {
  const data = zeroData({ dtype: 'float32', shape: [2, 3] });
  expect(data).toBeInstanceOf(Float32Array); expect([...data]).toEqual([0, 0, 0, 0, 0, 0]);
  expect(() => zeroData({ dtype: 'float16', shape: [] })).toThrow();
});
