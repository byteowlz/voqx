import { expect, test } from 'bun:test';
import { MODEL_SHA } from './live-core.mjs';
import {
  SPACE, WINDOW, HOP, cellCount, cellEnd, cellAt, vadActivity, speechSegments, snapshotJson, calibrationJson,
  describe, segments, toVtt, toEmbedding, DEFAULT_SETTINGS, PhraseSegmenter, PHRASE, PHRASE_SPACE, VAD_FRAME,
} from './clips-core.mjs';

const ev = (k, status, voice_id = null, candidates = []) => ({ observation: { start_sample: cellEnd(k) - WINDOW, end_sample: cellEnd(k), activity: 'speech' }, status, voice_id, candidates });
const names = new Map([['tmp-1', 'Anna'], ['tmp-2', 'Ben']]);

test('cells advance by 0.5 s and need a full 1.52 s window', () => {
  expect([cellCount(WINDOW - 1), cellCount(WINDOW), cellCount(WINDOW + HOP - 1), cellCount(WINDOW + HOP)]).toEqual([0, 1, 1, 2]);
  expect([cellAt(WINDOW - 1), cellAt(WINDOW), cellAt(WINDOW + HOP)]).toEqual([-1, 0, 1]);
  expect(cellEnd(3)).toBe(WINDOW + 3 * HOP);
});

test('space matches the native FASTDIAR window profile', () => {
  expect(SPACE).toEqual({ engine: 'fastdiar-small-cold80-mean-f32-v1', model_version: 'cold80-mean-f64-l2-f32-v1:' + MODEL_SHA, dimensions: 192, sample_rate: 16000, minimum_samples: 24320, maximum_samples: 160000 });
});

test('speech detector decides window activity and waits for coverage', () => {
  const probs = new Float32Array(100);
  const at = f => probs[f], s = DEFAULT_SETTINGS;
  // Window [0, 24320) has 47 whole frames (0..46).
  expect(vadActivity(at, 46, 0, WINDOW, s)).toBeNull();
  expect(vadActivity(at, 47, 0, WINDOW, s).activity).toBe('no_speech');
  probs.fill(0.9, 0, 24); // 24/47 >= 0.5
  expect(vadActivity(at, 47, 0, WINDOW, s).activity).toBe('speech');
  probs.fill(0, 0, 47); probs.fill(0.9, 0, 10); // 10/47 between 0.1 and 0.5
  expect(vadActivity(at, 47, 0, WINDOW, s).activity).toBe('unknown');
  expect(vadActivity(at, 47, 0, WINDOW, { ...s, minSpeech: 0.2 }).activity).toBe('speech');
  expect(vadActivity(at, 47, 0, WINDOW, { ...s, vadThreshold: 0.95 }).activity).toBe('no_speech');
  // A partial leading frame is excluded: window starting at 100 uses frames 1..46.
  probs.fill(0); probs[0] = 1;
  expect(vadActivity(at, 47, 100, WINDOW, s).fraction).toBe(0);
});

test('speech segments merge consecutive frames', () => {
  const probs = Float32Array.from([0, 0.9, 0.8, 0, 0.6, 0]);
  expect(speechSegments(f => probs[f], 0, 6, 0.5)).toEqual([{ start: 512, end: 1536 }, { start: 2048, end: 2560 }]);
});

test('snapshot includes only speakers with references; calibration is bound and clamped', () => {
  const vector = toEmbedding(Array.from({ length: 192 }, (_, i) => i === 0 ? 1 : 0));
  const snap = JSON.parse(snapshotJson([{ id: 'tmp-1', refs: [{ vector }] }, { id: 'tmp-2', refs: [] }]));
  expect(snap.voices.map(v => v.voice_id)).toEqual(['tmp-1']);
  expect(snap.voices[0].embeddings[0].length).toBe(192);
  const cal = JSON.parse(calibrationJson('abc', { ...DEFAULT_SETTINGS, tentative: 0.9, confident: 0.6 }));
  expect(cal).toEqual({ id: 'sandbox-exploratory-not-qualified', reference_revision: 'abc', minimum_samples: WINDOW, maximum_samples: WINDOW, confident: 0.6, tentative: 0.6, minimum_margin: 0.05, numerical_guard: 0.001 });
  expect(() => toEmbedding([1, 2])).toThrow();
});

test('only confident evidence names a person', () => {
  const cands = [{ voice_id: 'tmp-1', nearest: 0.8 }, { voice_id: 'tmp-2', nearest: 0.7 }];
  expect(describe(ev(0, 'confident', 'tmp-1', cands), names)).toEqual({ kind: 'speaker', id: 'tmp-1', text: 'Anna' });
  expect(describe(ev(0, 'tentative', null, cands), names).text).toBe('Maybe Anna');
  expect(describe(ev(0, 'ambiguous', null, cands), names).text).toBe('Unsure: Anna or Ben');
  expect(describe(ev(0, 'unknown', null, cands), names).kind).toBe('unknown');
  expect(describe(ev(0, 'unknown'), names).kind).toBe('quiet');
  expect(describe(ev(0, 'no_speech'), names).text).toBe('No speech');
  expect(describe(null, names).kind).toBe('pending');
});

test('segments merge contiguous cells and split on gaps or speaker changes', () => {
  const labels = new Map([[0, ev(0, 'confident', 'tmp-1')], [1, ev(1, 'confident', 'tmp-1')], [3, ev(3, 'confident', 'tmp-1')], [4, ev(4, 'confident', 'tmp-2')], [5, ev(5, 'unknown', null, [{ voice_id: 'tmp-1', nearest: 0.1 }])]]);
  const segs = segments(labels, names).map(s => [s.text, s.start, s.end]);
  expect(segs).toEqual([['Anna', WINDOW - HOP, WINDOW + HOP], ['Anna', WINDOW + 2 * HOP, WINDOW + 3 * HOP], ['Ben', WINDOW + 3 * HOP, WINDOW + 4 * HOP], ['Unknown voice', WINDOW + 4 * HOP, WINDOW + 5 * HOP]]);
  const vtt = toVtt(segments(labels, names));
  expect(vtt.startsWith('WEBVTT')).toBe(true);
  expect(vtt).toContain('00:00:01.020 --> 00:00:02.020\nAnna');
  expect(vtt).toContain('Unknown voice');
});

test('phrase segmenter: onset after two frames, end after 256 ms pause, short and long phrases', () => {
  const run = probs => { const seg = new PhraseSegmenter(0.5), events = []; probs.forEach((p, f) => { const e = seg.push(f, p); if (e) events.push(e); }); return { events, seg }; };
  const on = n => Array(n).fill(0.9), off = n => Array(n).fill(0.1);
  // 0.1 s silence, 0.8 s speech (25 frames) with a 5-frame internal gap, then silence.
  let { events } = run([...off(3), ...on(10), ...off(5), ...on(10), ...off(12)]);
  expect(events).toEqual([{ start: 3 * VAD_FRAME, end: 28 * VAD_FRAME, decidedAt: 36 * VAD_FRAME, reason: 'pause', short: false }]);
  expect((events[0].decidedAt - events[0].end) / 16).toBe(256);
  // A single speech frame never starts a phrase; a very short burst is flagged short.
  expect(run([...off(2), 0.9, ...off(20)]).events).toEqual([]);
  expect(run([...on(4), ...off(9)]).events[0].short).toBe(true);
  // Continuous speech is cut at the maximum length.
  const long = run(on(130)).events[0];
  expect(long.reason).toBe('max');
  expect(long.end - long.start).toBe(PHRASE.maxSpeech);
});

test('phrase profile is a distinct sandbox space with a 0.6 s floor', () => {
  expect(PHRASE_SPACE.model_version.endsWith(':sandbox-phrase')).toBe(true);
  expect(PHRASE_SPACE.minimum_samples).toBe(9600);
  expect(SPACE.minimum_samples).toBe(WINDOW);
});
