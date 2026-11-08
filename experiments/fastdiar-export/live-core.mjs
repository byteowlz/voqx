export const MODEL_SHA = 'e7e3878d00afb742094eb22a8edc3a7f756d8ae1a10ec16200d95e56cf6ec3b1';
export const QUERY_SAMPLES = 24320, REFERENCE_SAMPLES = 48640, MAX_REFERENCES = 7;
export class PcmRing {
  constructor(size = REFERENCE_SAMPLES) { this.data = new Float32Array(size); this.clear(); }
  clear() { this.data.fill(0); this.count = 0; this.at = 0; }
  push(pcm) { for (const sample of pcm) { this.data[this.at] = sample; this.at = (this.at + 1) % this.data.length; this.count = Math.min(this.count + 1, this.data.length); } }
  tail(n) {
    if (n > this.count) throw new Error('Wait for enough microphone audio.');
    return Float32Array.from({ length: n }, (_, i) => this.data[(this.at - n + i + this.data.length) % this.data.length]);
  }
}
export function preparePcm(pcm, limit, minimum = QUERY_SAMPLES) {
  if (!(pcm instanceof Float32Array) || !pcm.every(Number.isFinite)) throw new Error('Expected finite mono audio.');
  const actual = Math.min(pcm.length, limit);
  if (actual < minimum) return { status: 'InsufficientAudio', samples: actual };
  const input = pcm.subarray(0, actual);
  const rms = Math.sqrt(input.reduce((sum, n) => sum + n * n, 0) / input.length);
  if (rms < 0.0001) return { status: 'LowLevel', samples: actual, rms };
  const padded = new Float32Array(Math.ceil(actual / 1280) * 1280); padded.set(input);
  return { status: 'Ready', pcm: padded, samples: actual, paddedSamples: padded.length - actual, rms };
}
function unit(vector) {
  if (!Array.isArray(vector) || vector.length !== 192 || !vector.every(Number.isFinite)) throw new Error('Incompatible reference vector.');
  const norm = Math.hypot(...vector);
  if (!(norm > 1e-12)) throw new Error('Unusable reference vector.');
  return vector.map(n => n / norm);
}
const dot = (a, b) => a.reduce((sum, n, i) => sum + n * b[i], 0);
export class TemporaryBank {
  constructor(modelSha) { if (modelSha !== MODEL_SHA) throw new Error('IncompatibleModel'); this.clear(); }
  clear() { this.references = []; }
  add(vector, samples) {
    if (this.references.length >= MAX_REFERENCES) throw new Error('Seven references maximum. Clear the bank to start again.');
    this.references.push({ vector: unit(vector), samples });
    return this.snapshot();
  }
  snapshot() { return { modelSha256: MODEL_SHA, pooling: 'frame-mean-f64', status: 'Uncalibrated', references: this.references.map((r, i) => ({ number: i + 1, samples: r.samples, seconds: r.samples / 16000 })) }; }
  score(vector) {
    if (!this.references.length) return { status: 'NeedsReference', nearest: null, centroid: null, count: 0 };
    const query = unit(vector), scores = this.references.map(r => dot(r.vector, query));
    const mean = Array.from({ length: 192 }, (_, i) => this.references.reduce((sum, r) => sum + r.vector[i] / this.references.length, 0));
    const norm = Math.hypot(...mean);
    return { status: 'Uncalibrated', nearest: Math.max(...scores), centroid: norm > 1e-12 ? dot(mean.map(n => n / norm), query) : null, count: scores.length };
  }
}
