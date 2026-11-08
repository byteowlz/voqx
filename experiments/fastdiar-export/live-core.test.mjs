import { test, expect } from 'bun:test';
import { TemporaryBank, PcmRing, MODEL_SHA, QUERY_SAMPLES, preparePcm } from './live-core.mjs';
const vector = sign => [sign, ...new Array(191).fill(0)];
test('ring is bounded and preserves exact newest samples', () => {
  const ring = new PcmRing(5); ring.push([1, 2, 3]); expect(() => ring.tail(4)).toThrow();
  ring.push([4, 5, 6, 7]); expect([...ring.tail(5)]).toEqual([3, 4, 5, 6, 7]);
  ring.clear(); expect(ring.count).toBe(0); expect([...ring.data]).toEqual([0, 0, 0, 0, 0]);
});
test('missing references never snap a query to an identity', () => {
  expect(new TemporaryBank(MODEL_SHA).score(null)).toEqual({ status: 'NeedsReference', nearest: null, centroid: null, count: 0 });
});
test('nearest and centroid are distinct uncalibrated evidence', () => {
  const bank = new TemporaryBank(MODEL_SHA); bank.add(vector(1), 48640);
  expect(bank.score(vector(1)).nearest).toBe(1); expect(bank.score(vector(-1)).nearest).toBe(-1);
  bank.add(vector(-1), 48640);
  expect(bank.score(vector(1))).toEqual({ status: 'Uncalibrated', nearest: 1, centroid: null, count: 2 });
  bank.clear(); expect(bank.snapshot().references).toEqual([]);
});
test('model/dimension/finite guards and reference limit', () => {
  expect(() => new TemporaryBank('other')).toThrow('IncompatibleModel');
  const bank = new TemporaryBank(MODEL_SHA);
  expect(() => bank.add(new Array(256).fill(1), 48640)).toThrow();
  expect(() => bank.add(new Array(192).fill(NaN), 48640)).toThrow();
  for (let i = 0; i < 7; i++) bank.add(vector(1), 48640);
  expect(() => bank.add(vector(1), 48640)).toThrow();
});
test('audio duration and energy are not invented speech evidence', () => {
  expect(preparePcm(new Float32Array(100), QUERY_SAMPLES).status).toBe('InsufficientAudio');
  expect(preparePcm(new Float32Array(QUERY_SAMPLES), QUERY_SAMPLES).status).toBe('LowLevel');
  expect(() => preparePcm(new Float32Array(QUERY_SAMPLES).fill(NaN), QUERY_SAMPLES)).toThrow();
  const value = preparePcm(new Float32Array(25000).fill(.1), 25000);
  expect(value.samples).toBe(25000); expect(value.pcm.length % 1280).toBe(0);
  expect(value.paddedSamples).toBe(value.pcm.length - 25000);
});
