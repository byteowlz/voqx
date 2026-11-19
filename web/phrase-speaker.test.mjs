import { expect, test } from 'bun:test';
import { PhraseSpeaker, phraseSpace } from './phrase-speaker.mjs';
import { MODEL_SHA256 } from './step-runtime.mjs';

const space = phraseSpace(4800, 32000);
const observation = { generation: 1, start_sample: 1200, end_sample: 6001, activity: 'speech' };
const vector = () => Float32Array.from({ length: 192 }, (_, i) => i === 0 ? 1 : 0);
function fixture() {
  const calls = [];
  const runtime = { reset() { calls.push('reset'); }, async push(pcm) { calls.push(pcm); return { pendingSamples: 0, processedSamples: pcm.length, frames: Array.from({ length: pcm.length / 1280 }, vector) }; }, async close() { calls.push('close'); } };
  const matcher = { evaluate(values, json) { return JSON.stringify({ observation: JSON.parse(json), status: values.length ? 'uncalibrated' : 'unknown', voice_id: null }); }, free() { calls.push('free'); } };
  return { speaker: new PhraseSpeaker(runtime, matcher, space, 'fixture-revision'), calls };
}

test('native phrase descriptor pins duration/hash/reset/pooling; never aliases sandbox or WeSpeaker', () => {
  expect(space).toEqual({ engine: 'fastdiar-small-cold80-phrase-f32-v1', model_version: `cold80-mean-f64-l2-f32-v1:${MODEL_SHA256}:phrase-v1:4800-32000`, dimensions: 192, sample_rate: 16000, minimum_samples: 4800, maximum_samples: 32000 });
  for (const bounds of [[1279, 32000], [4800, 4799], [4800, 160001], [4800.5, 32000]]) expect(() => phraseSpace(...bounds)).toThrow();
});

test('cold reset, final-frame padding only, no onset trim, original scope survives', async () => {
  const { speaker, calls } = fixture();
  const pcm = new Float32Array(4801); pcm[0] = 0.1; pcm[4800] = 0.2;
  const result = await speaker.evaluate(pcm, observation);
  expect(result.observation).toEqual(observation);
  expect(calls[0]).toBe('reset');
  expect(calls[1].length).toBe(5120);
  expect(calls[1][4800]).toBe(pcm[4800]);
  expect(calls[1].subarray(4801).every(n => n === 0)).toBe(true);
  expect(pcm.length).toBe(4801);
  await speaker.close(); expect(calls.slice(-2)).toEqual(['close', 'free']);
});

test('activity and duration abstentions skip encoding; malformed/silent/oversized PCM fails', async () => {
  const { speaker, calls } = fixture();
  for (const activity of ['unknown', 'no_speech', 'overlap']) await speaker.evaluate(new Float32Array(4801), { ...observation, activity });
  await speaker.evaluate(new Float32Array(1), { ...observation, end_sample: 1201 });
  expect(calls.length).toBe(0);
  for (const [pcm, obs] of [[new Float32Array(4801), observation], [new Float32Array([NaN]), { ...observation, end_sample: 1201 }], [new Float32Array(32001), { ...observation, end_sample: 33201 }], [new Float32Array(4801), { ...observation, end_sample: 2 }]]) {
    await expect(speaker.evaluate(pcm, obs)).rejects.toThrow();
  }
  await speaker.close();
  await expect(speaker.evaluate(new Float32Array(4801), observation)).rejects.toThrow();
});

test('create rejects incompatible profiles before model load and never supplies thresholds', async () => {
  let constructed = 0;
  class Matcher { constructor(...args) { constructed++; expect(args[1]).toBeUndefined(); } free() {} }
  await expect(PhraseSpeaker.create({}, { snapshot: { space: { ...space, dimensions: 256 }, voices: [] }, SpeakerAttributor: Matcher })).rejects.toThrow();
  expect(constructed).toBe(0);
});
