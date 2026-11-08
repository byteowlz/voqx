import { test, expect } from 'bun:test';
import { poolFrames, fromBase64, toBase64 } from './frame-pool.mjs';
test('unit mean/last pooling and cancellation abstention', () => {
  const a = new Float32Array(192), b = new Float32Array(192); a[0] = 1; b[1] = 1;
  const pooled = poolFrames([a, b]);
  expect(Math.hypot(...pooled.mean)).toBeCloseTo(1, 12);
  expect(pooled.last[1]).toBe(1);
  b[1] = 0; b[0] = -1;
  expect(poolFrames([a, b]).mean).toBeNull();
});
test('invalid dimensions/nonfinite frames are rejected', () => {
  expect(() => poolFrames([])).toThrow();
  expect(() => poolFrames([new Float32Array(256)])).toThrow();
  expect(() => poolFrames([new Float32Array(192).fill(NaN)])).toThrow();
});
test('large binary IPC roundtrip preserves every byte without disk', () => {
  const input = Uint8Array.from({ length: 100000 }, (_, i) => i % 256);
  expect(fromBase64(toBase64(input))).toEqual(input);
  expect(() => fromBase64('x'.repeat(1000001))).toThrow();
});
