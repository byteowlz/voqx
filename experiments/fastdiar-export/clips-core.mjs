// Pure clip/microphone attribution helpers: windows, level gate, snapshots, timelines, exports.
// No audio is stored or sent anywhere; callers keep PCM and vectors in tab memory only.
import { MODEL_SHA } from './live-core.mjs';

export const RATE = 16000;
export const WINDOW = 24320;       // 1.52 s observation, exactly 19 encoder frames
export const REFERENCE = 48640;    // 3.04 s reference window
export const HOP = 8000;           // a new observation every 0.5 s of audio
export const MAX_SPEAKERS = 8;
export const MAX_REFERENCES = 7;
export const MAX_CLIP_SECONDS = 1800;
export const SANDBOX_POLICY = 'sandbox-exploratory-not-qualified';

// Must equal voqx_recognition::fastdiar::embedding_space() for the same graph.
function SPACE_BASE() {
  return { engine: 'fastdiar-small-cold80-mean-f32-v1', model_version: 'cold80-mean-f64-l2-f32-v1:' + MODEL_SHA, dimensions: 192, sample_rate: RATE, minimum_samples: WINDOW, maximum_samples: 160000 };
}
export const SPACE = Object.freeze(SPACE_BASE());

export const DEFAULT_SETTINGS = Object.freeze({ confident: 0.5, tentative: 0.4, margin: 0.05, stableSeconds: 0.5, vadThreshold: 0.5, minSpeech: 0.5 });

// Silero VAD pinned exactly as in eaRS docs/client-vad.md (snakers4/silero-vad@066409f3, MIT).
export const VAD_SHA = '1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3';
export const VAD_FRAME = 512, VAD_CONTEXT = 64;

// Per-phrase mode: Silero segments live audio; one decision per phrase, no window dwell.
export const PHRASE = Object.freeze({ startFrames: 2, endFrames: 8, minSpeech: 4800, maxSpeech: 64000, pre: 2400, post: 2400, minSpan: 9600 });
// Sandbox-only profile: same encoder, reset and pooling, but variable 0.6-10 s spans. Not the native window profile.
export const PHRASE_SPACE = Object.freeze({ ...SPACE_BASE(), model_version: SPACE_BASE().model_version + ':sandbox-phrase', minimum_samples: PHRASE.minSpan });

/** Cell k describes the window ending at WINDOW + k * HOP. */
export const cellEnd = k => WINDOW + k * HOP;
export const cellCount = samples => samples < WINDOW ? 0 : Math.floor((samples - WINDOW) / HOP) + 1;
export const cellAt = sample => sample < WINDOW ? -1 : Math.floor((sample - WINDOW) / HOP);

/**
 * Window activity from Silero frame probabilities. `prob(f)` covers samples [f*512, f*512+512).
 * Only frames fully inside the window count. Returns null until the detector has covered it.
 * Speech: at least `minSpeech` of frames at or above `vadThreshold` (natural speech has pauses).
 * Mostly non-speech windows abstain as no_speech; mixed windows abstain as unknown.
 */
export function vadActivity(prob, done, start, end, settings = DEFAULT_SETTINGS) {
  const first = Math.ceil(start / VAD_FRAME), last = Math.floor(end / VAD_FRAME);
  if (last > done) return null;
  if (last <= first) return { activity: 'unknown', fraction: 0 };
  let speech = 0;
  for (let f = first; f < last; f++) if (prob(f) >= settings.vadThreshold) speech++;
  const fraction = speech / (last - first);
  return { activity: fraction >= settings.minSpeech ? 'speech' : fraction <= 0.1 ? 'no_speech' : 'unknown', fraction };
}

/** Merge detector frames at or above threshold into [start, end) sample segments. */
export function speechSegments(prob, fromFrame, toFrame, threshold) {
  const out = [];
  for (let f = fromFrame; f < toFrame; f++) {
    if (prob(f) < threshold) continue;
    const start = f * VAD_FRAME, last = out.at(-1);
    if (last && last.end === start) last.end = start + VAD_FRAME; else out.push({ start, end: start + VAD_FRAME });
  }
  return out;
}

/**
 * Frame-by-frame phrase segmentation over Silero probabilities. A phrase starts after
 * `startFrames` speech frames and ends after `endFrames` non-speech frames (256 ms) or at
 * `maxSpeech`. Events carry the speech span and the sample at which the end was decided.
 */
export class PhraseSegmenter {
  constructor(threshold) { this.threshold = threshold; this.reset(); }
  reset() { this.inSpeech = false; this.run = 0; this.silence = 0; this.start = 0; this.last = 0; }
  push(frame, p) {
    const speech = p >= this.threshold;
    if (!this.inSpeech) {
      this.run = speech ? this.run + 1 : 0;
      if (this.run >= PHRASE.startFrames) { this.inSpeech = true; this.start = frame - this.run + 1; this.last = frame; this.silence = 0; }
      return null;
    }
    if (speech) { this.last = frame; this.silence = 0; } else this.silence++;
    const ended = this.silence >= PHRASE.endFrames, tooLong = (frame + 1 - this.start) * VAD_FRAME >= PHRASE.maxSpeech;
    if (!ended && !tooLong) return null;
    const event = { start: this.start * VAD_FRAME, end: (this.last + 1) * VAD_FRAME, decidedAt: (frame + 1) * VAD_FRAME, reason: ended ? 'pause' : 'max' };
    this.inSpeech = false; this.run = 0;
    return { ...event, short: event.end - event.start < PHRASE.minSpeech };
  }
}

/** Same arithmetic as the native window adapter: FP64 unit vector stored as f32. */
export function toEmbedding(unitVector) {
  if (!unitVector || unitVector.length !== 192) throw new Error('Expected a 192-dimensional vector.');
  const values = Float32Array.from(unitVector);
  if (!values.every(Number.isFinite)) throw new Error('Nonfinite vector.');
  return values;
}

export function snapshotJson(speakers, space = SPACE) {
  const voices = speakers.filter(s => s.refs.length).map(s => ({ voice_id: s.id, embeddings: s.refs.map(r => Array.from(r.vector)) }));
  return JSON.stringify({ space, voices });
}

export function calibrationJson(revision, settings, band = [WINDOW, WINDOW]) {
  return JSON.stringify({
    id: SANDBOX_POLICY, reference_revision: revision, minimum_samples: band[0], maximum_samples: band[1],
    confident: settings.confident, tentative: Math.min(settings.tentative, settings.confident),
    minimum_margin: settings.margin, numerical_guard: 0.001,
  });
}

/** Display label for one evidence record. Only stabilized confident evidence names a person. */
export function describe(evidence, names) {
  if (!evidence) return { kind: 'pending', text: 'Not analyzed yet' };
  const top = evidence.candidates?.[0], second = evidence.candidates?.[1];
  const name = id => names.get(id) ?? id;
  switch (evidence.status) {
    case 'confident': return { kind: 'speaker', id: evidence.voice_id, text: name(evidence.voice_id) };
    case 'tentative': return { kind: 'unsure', id: top?.voice_id, text: top ? 'Maybe ' + name(top.voice_id) : 'Unsure' };
    case 'ambiguous': return { kind: 'unsure', text: 'Unsure: ' + name(top.voice_id) + ' or ' + name(second.voice_id) };
    case 'unknown': return evidence.candidates?.length ? { kind: 'unknown', text: 'Unknown voice' } : { kind: 'quiet', text: 'Partly speech' };
    case 'no_speech': return { kind: 'quiet', text: 'No speech' };
    case 'overlap': return { kind: 'unsure', text: 'Overlapping speech' };
    case 'insufficient_audio': return { kind: 'quiet', text: 'Too little audio' };
    case 'incompatible_model': return { kind: 'unknown', text: 'Incompatible references' };
    default: return { kind: 'unsure', text: 'Uncalibrated' };
  }
}

/** Merge per-cell labels into contiguous segments. Each cell covers its newest 0.5 s. */
export function segments(labels, names) {
  const out = [];
  for (const [k, evidence] of [...labels.entries()].sort((a, b) => a[0] - b[0])) {
    const d = describe(evidence, names), start = cellEnd(k) - HOP, end = cellEnd(k);
    const key = d.kind + ':' + (d.kind === 'speaker' ? d.id : '');
    const last = out.at(-1);
    if (last && last.key === key && last.end === start) last.end = end;
    else out.push({ key, kind: d.kind, id: d.kind === 'speaker' ? d.id : null, text: d.kind === 'speaker' ? d.text : d.kind === 'unknown' ? 'Unknown voice' : d.text, start, end });
  }
  return out;
}

const stamp = samples => {
  const ms = Math.round(samples / RATE * 1000), h = Math.floor(ms / 3600000), m = Math.floor(ms / 60000) % 60, s = Math.floor(ms / 1000) % 60;
  return [h, m, s].map(n => String(n).padStart(2, '0')).join(':') + '.' + String(ms % 1000).padStart(3, '0');
};

export function toVtt(segs) {
  const lines = ['WEBVTT', '', 'NOTE Experimental voqx speaker test. Exploratory thresholds; not identification or authentication.', ''];
  for (const s of segs.filter(s => s.kind === 'speaker' || s.kind === 'unknown')) lines.push(stamp(s.start) + ' --> ' + stamp(s.end), s.text, '');
  return lines.join('\n');
}

export function toJson({ clip, revision, settings, labels, names, references }) {
  return JSON.stringify({
    format: 'voqx-speaker-timeline/1', experimental: true, policy: SANDBOX_POLICY,
    note: 'Exploratory sandbox thresholds. Cosine scores are not probabilities, identity proof or permission.',
    clip, model: { space: SPACE }, reference_revision: revision, settings, window_samples: WINDOW, hop_samples: HOP,
    speakers: [...names.entries()].map(([id, name]) => ({ id, name, references: references.filter(r => r.id === id).map(r => ({ start: r.start ?? null, end: r.end ?? null, source: r.source })) })),
    cells: [...labels.entries()].sort((a, b) => a[0] - b[0]).map(([k, e]) => ({
      start_sample: e.observation.start_sample, end_sample: e.observation.end_sample, paints: [cellEnd(k) - HOP, cellEnd(k)],
      activity: e.observation.activity, status: e.status, voice_id: e.voice_id, name: e.voice_id ? names.get(e.voice_id) : null,
      candidates: e.candidates.map(c => ({ voice_id: c.voice_id, name: names.get(c.voice_id), nearest: c.nearest, centroid: c.centroid })),
    })),
  }, null, 2);
}
