import * as ort from '/ort/ort.wasm.min.mjs';
import { validateManifest, zeroData, sha256 } from './protocol.mjs';

// This worker runs CPU only. No WebGPU/WebNN fallback or inference service.
ort.env.wasm.numThreads = 1;
ort.env.wasm.proxy = false;
ort.env.wasm.wasmPaths = '/ort/';

export class StepRuntime {
  static async create(manifest) {
    validateManifest(manifest);
    const started = performance.now();
    const response = await fetch('/encoder.onnx');
    if (!response.ok) throw new Error(`Model fetch failed: ${response.status}`);
    const bytes = await response.arrayBuffer();
    if (await sha256(bytes) !== manifest.onnx_sha256) throw new Error('Model hash mismatch');
    const session = await ort.InferenceSession.create(bytes, { executionProviders: ['wasm'] });
    const inputs = ['audio', ...manifest.state.map(s => s.input)];
    const outputs = ['embedding', ...manifest.state.map(s => s.output)];
    if (JSON.stringify(session.inputNames) !== JSON.stringify(inputs) || JSON.stringify(session.outputNames) !== JSON.stringify(outputs)) {
      await session.release(); throw new Error('Model tensor contract mismatch');
    }
    return new StepRuntime(manifest, session, performance.now() - started);
  }
  constructor(manifest, session, loadMs) {
    this.manifest = manifest; this.session = session; this.loadMs = loadMs;
    this.states = []; this.busy = false; this.closed = false; this.fault = false; this.reset();
  }
  reset() {
    if (this.busy || this.closed) throw new Error('Runtime is busy or closed');
    this.fault = false;
    for (const value of this.states) value.dispose();
    this.states = this.manifest.state.map(slot => new ort.Tensor(slot.dtype, zeroData(slot), slot.shape));
    this.pending = new Float32Array(0); this.processedSamples = 0;
  }
  async #step(pcm) {
    const input = new ort.Tensor('float32', pcm, [this.manifest.step_samples]);
    const feed = { audio: input };
    for (let i = 0; i < this.states.length; i++) feed[this.manifest.state[i].input] = this.states[i];
    const started = performance.now();
    let output;
    try { output = await this.session.run(feed); } finally { input.dispose(); }
    const elapsed = performance.now() - started;
    const embedding = output.embedding.data.slice();
    output.embedding.dispose();
    if (embedding.length !== 192 || !embedding.every(Number.isFinite)) throw new Error('Invalid embedding output');
    const next = this.manifest.state.map(slot => output[slot.output]);
    for (let i = 0; i < next.length; i++) {
      const field = this.manifest.state[i].field;
      if (field === 'audio_count' && next[i].data[0] !== BigInt(this.processedSamples + pcm.length)) throw new Error('Frozen audio counter');
      if (field === 'cms_count' && next[i].data[0] !== BigInt((this.processedSamples + pcm.length) / 160)) throw new Error('Frozen spectral counter');
    }
    for (const value of this.states) value.dispose();
    this.states = next; this.processedSamples += pcm.length;
    return { embedding, elapsed };
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
