import { StepRuntime, MODEL_SHA256 } from './step-runtime.mjs';
import { poolFrames } from './frame-pool.mjs';

/** Exact native load_phrase descriptor. Bounds are profile semantics, not accuracy. */
export function phraseSpace(minimum, maximum) {
  if (!Number.isSafeInteger(minimum) || !Number.isSafeInteger(maximum) || minimum < 1280 || maximum < minimum || maximum > 160000) throw Error('Invalid phrase duration band');
  return Object.freeze({ engine: 'fastdiar-small-cold80-phrase-f32-v1',
    model_version: `cold80-mean-f64-l2-f32-v1:${MODEL_SHA256}:phrase-v1:${minimum}-${maximum}`,
    dimensions: 192, sample_rate: 16000, minimum_samples: minimum, maximum_samples: maximum });
}

/** One loaded CPU encoder + the actual voqx Rust/WASM one-shot matcher.
 * Supply bytes and an explicitly authorized snapshot; no fetch, capture, registry
 * write, auto-enrollment or threshold defaults. Run on a dedicated host worker. */
export class PhraseSpeaker {
  static async create(ort, { model, manifest, snapshot, calibration, SpeakerAttributor }) {
    snapshot = structuredClone(snapshot);
    const space = phraseSpace(snapshot.space.minimum_samples, snapshot.space.maximum_samples);
    if (Object.keys(snapshot.space).length !== Object.keys(space).length ||
        Object.keys(space).some(key => snapshot.space[key] !== space[key])) throw Error('Incompatible phrase profile');
    const matcher = new SpeakerAttributor(JSON.stringify(snapshot), calibration == null ? undefined : JSON.stringify(calibration), 1n, 16000n);
    try {
      const runtime = await StepRuntime.create(ort, manifest, model);
      return new PhraseSpeaker(runtime, matcher, space, matcher.referenceRevision());
    } catch (error) { matcher.free(); throw error; }
  }

  constructor(runtime, matcher, space, referenceRevision) {
    this.runtime = runtime; this.matcher = matcher; this.space = Object.freeze({ ...space });
    this.referenceRevision = referenceRevision; this.busy = false; this.closed = false;
  }

  async encode(pcm) {
    if (this.busy || this.closed || !(pcm instanceof Float32Array) ||
        pcm.length < this.space.minimum_samples || pcm.length > this.space.maximum_samples ||
        !pcm.every(Number.isFinite) || pcm.every(n => n === 0)) throw Error('Invalid phrase PCM or busy/closed encoder');
    this.busy = true;
    try {
      this.runtime.reset();
      // Only complete the final structural model frame. Never pad to the minimum.
      const padded = new Float32Array(Math.ceil(pcm.length / 1280) * 1280); padded.set(pcm);
      const result = await this.runtime.push(padded);
      if (result.pendingSamples || result.processedSamples !== padded.length) throw Error('Incomplete phrase encoding');
      const mean = poolFrames(result.frames).mean;
      if (!mean) throw Error('Invalid pooled embedding');
      return Float32Array.from(mean); // Native profile stores normalized FP64 pooling as f32.
    } finally { this.busy = false; }
  }

  async evaluate(pcm, observation) {
    if (this.closed || this.busy || !(pcm instanceof Float32Array) || !pcm.every(Number.isFinite) ||
        pcm.length > this.space.maximum_samples ||
        !['generation', 'start_sample', 'end_sample'].every(k => Number.isSafeInteger(observation[k]) && observation[k] >= 0) ||
        observation.end_sample - observation.start_sample !== pcm.length ||
        !['speech', 'no_speech', 'overlap', 'unknown'].includes(observation.activity)) throw Error('Invalid phrase observation');
    // Freeze scope before the first await; caller mutation cannot retime evidence.
    const json = JSON.stringify(observation);
    const needed = observation.activity === 'speech' && pcm.length >= this.space.minimum_samples;
    const values = needed ? await this.encode(pcm) : new Float32Array(0);
    return JSON.parse(this.matcher.evaluate(values, json));
  }

  async close() {
    if (this.busy) throw Error('Wait for current encoding before closing');
    if (this.closed) return;
    this.closed = true;
    try { await this.runtime.close(); } finally { this.matcher.free(); }
  }
}
