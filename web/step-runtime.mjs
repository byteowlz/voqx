import { validateManifest, zeroData, sha256 } from './tensor-contract.mjs';

export const MODEL_SHA256 = 'e7e3878d00afb742094eb22a8edc3a7f756d8ae1a10ec16200d95e56cf6ec3b1';
export const MODEL_BYTES = 10665298;

// Caller supplies graph bytes and ORT Web. No fetch, capture, GPU or server fallback.
export class StepRuntime {
  static async create(ort, manifest, bytes) {
    manifest = structuredClone(validateManifest(manifest));
    if (!(bytes instanceof Uint8Array) || bytes.byteLength !== MODEL_BYTES ||
        manifest.onnx_sha256 !== MODEL_SHA256 || await sha256(bytes) !== MODEL_SHA256) throw Error('Model hash/size mismatch');
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    const started = performance.now();
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    const inputs = ['audio', ...manifest.state.map(s => s.input)];
    const outputs = ['embedding', ...manifest.state.map(s => s.output)];
    if (JSON.stringify(session.inputNames) !== JSON.stringify(inputs) || JSON.stringify(session.outputNames) !== JSON.stringify(outputs)) {
      await session.release(); throw Error('Model tensor contract mismatch');
    }
    return new StepRuntime(ort, manifest, session, performance.now() - started);
  }
  constructor(ort, manifest, session, loadMs) {
    this.ort = ort; this.manifest = manifest; this.session = session; this.loadMs = loadMs;
    this.states = []; this.busy = false; this.closed = false; this.fault = false; this.reset();
  }
  reset() {
    if (this.busy || this.closed) throw new Error('Runtime is busy or closed');
    this.fault = false;
    for (const value of this.states) value.dispose();
    this.states = this.manifest.state.map(slot => new this.ort.Tensor(slot.dtype, zeroData(slot), slot.shape));
    this.pending = new Float32Array(0); this.processedSamples = 0;
  }
  async #step(pcm) {
    const input = new this.ort.Tensor('float32', pcm, [this.manifest.step_samples]);
    const feed = { audio: input };
    for (let i = 0; i < this.states.length; i++) feed[this.manifest.state[i].input] = this.states[i];
    const started = performance.now();
    let output;
    try { output = await this.session.run(feed); } finally { input.dispose(); }
    const elapsed = performance.now() - started;
    let adopted = false;
    try {
      const embedding = output.embedding.data.slice();
      if (embedding.length !== 192 || !embedding.every(Number.isFinite)) throw new Error('Invalid embedding output');
      const next = this.manifest.state.map(slot => output[slot.output]);
      for (let i = 0; i < next.length; i++) {
        const field = this.manifest.state[i].field;
        if (field === 'audio_count' && next[i].data[0] !== BigInt(this.processedSamples + pcm.length)) throw new Error('Frozen audio counter');
        if (field === 'cms_count' && next[i].data[0] !== BigInt((this.processedSamples + pcm.length) / 160)) throw new Error('Frozen spectral counter');
      }
      for (const value of this.states) value.dispose();
      this.states = next; this.processedSamples += pcm.length; adopted = true;
      return { embedding, elapsed };
    } finally {
      if (adopted) output.embedding.dispose();
      else for (const value of Object.values(output)) value.dispose();
    }
  }
  async push(pcm) {
    if (this.busy || this.closed || this.fault) throw new Error('Runtime is busy, closed or faulted');
    if (!(pcm instanceof Float32Array) || pcm.length > 16000 * 30 || !pcm.every(Number.isFinite)) throw new Error('Expected bounded finite mono PCM');
    this.busy = true;
    try { return await this.#consume(pcm); }
    catch (error) { this.fault = true; throw error; }
    finally { this.busy = false; }
  }
  async #consume(pcm) {
    const data = new Float32Array(this.pending.length + pcm.length);
    data.set(this.pending); data.set(pcm, this.pending.length);
    const frames = [], timings = [], samples = this.manifest.step_samples;
    let offset = 0;
    for (; offset + samples <= data.length; offset += samples) {
      const frame = await this.#step(data.subarray(offset, offset + samples));
      frames.push(frame.embedding); timings.push(frame.elapsed);
    }
    this.pending = data.slice(offset);
    return { frames, timings, processedSamples: this.processedSamples, pendingSamples: this.pending.length };
  }
  async close() {
    if (this.busy || this.closed) throw new Error('Runtime is busy or closed');
    this.closed = true;
    for (const value of this.states) value.dispose();
    this.states = []; this.pending = new Float32Array(0);
    await this.session.release();
  }
}
