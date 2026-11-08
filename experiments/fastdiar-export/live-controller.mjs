import { HeadMicrophone } from '/heads-capture.mjs';
import { decodeFile } from '/capture.mjs';
import { PcmRing, QUERY_SAMPLES, REFERENCE_SAMPLES, MAX_REFERENCES } from './live-core.mjs';
import { WorkerRpc } from './worker-rpc.mjs';

export class LiveSpeakerTest {
  constructor(changed) {
    this.changed = changed; this.microphone = new HeadMicrophone(); this.ring = new PcmRing();
    this.state = { ready: false, active: false, micPending: false, busy: false, clearing: false, referenceIntent: false, captureSamples: 0, references: [], last: null, rms: 0, status: 'Loading the local CPU model…', error: '' };
    this.revision = 0;
    this.worker = new WorkerRpc('/live-worker.mjs', message => this.failWorker(message));
    this.loading = this.rpc('load').then(model => {
      this.state.model = model; this.state.ready = true;
      this.state.status = 'Ready. Start the microphone, then capture your reference.'; this.notify();
    }).catch(error => this.showError(error));
  }
  snapshot() { return structuredClone(this.state); }
  notify() { this.changed(this.snapshot()); }
  showError(error) { this.state.error = error.message; this.notify(); }
  guard() {
    if (!this.state.ready) throw new Error('Wait for the CPU model to load.');
    if (this.state.busy || this.state.clearing || this.state.micPending || this.state.referenceIntent) throw new Error('Wait for the current operation to finish, or press Stop.');
    this.state.error = '';
  }
  rpc(type, args = {}) { return this.worker.call(type, args); }
  failWorker(message) {
    this.stop(); this.state.ready = false; this.state.references = []; this.state.last = null; this.state.error = message; this.notify();
  }
  async start() { this.guard(); return await this.openMicrophone(); }
  async openMicrophone() {
    if (this.state.active) return true;
    if (this.state.micPending) throw new Error('Microphone permission is still pending.');
    this.state.micPending = true; this.state.status = 'Waiting for microphone permission…'; this.ring.clear(); this.notify();
    try {
      this.state.active = await this.microphone.start(pcm => this.frame(pcm));
      this.state.status = this.state.active ? 'Microphone on. Capture your reference, or say a new sentence.' : 'Microphone stopped.';
      return this.state.active;
    } finally { this.state.micPending = false; this.notify(); }
  }
  frame(pcm) {
    if (!this.state.active) return;
    this.ring.push(pcm);
    this.state.rms = Math.sqrt(pcm.reduce((n, value) => n + value * value, 0) / pcm.length);
    if (this.capture) {
      const remaining = this.capture.pcm.length - this.capture.used, n = Math.min(remaining, pcm.length);
      this.capture.pcm.set(pcm.subarray(0, n), this.capture.used); this.capture.used += n;
      this.state.captureSamples = this.capture.used;
      if (this.capture.used === this.capture.pcm.length) {
        const capture = this.capture; this.capture = null; capture.resolve(capture.pcm);
      }
    }
    this.notify();
  }
  stop() {
    this.microphone.stop(); this.ring.clear(); this.state.active = false; this.state.rms = 0;
    if (this.capture) {
      this.capture.pcm.fill(0); this.capture.reject(Object.assign(new Error('Capture cancelled.'), { code: 'CANCELLED' })); this.capture = null;
    }
    this.state.status = this.state.busy ? 'Microphone stopped. Finishing the current CPU check…' : 'Microphone stopped. References remain until cleared or refreshed.';
    this.notify();
  }
  async reference() {
    this.guard();
    if (this.state.references.length >= MAX_REFERENCES) throw new Error('Seven references maximum. Clear the bank to start again.');
    this.state.referenceIntent = true; this.state.captureSamples = 0; this.notify();
    try {
      if (!await this.openMicrophone()) return { status: 'Cancelled' };
      this.state.status = 'Speak now for about three seconds.'; this.notify();
      const pcm = await new Promise((resolve, reject) => { this.capture = { pcm: new Float32Array(REFERENCE_SAMPLES), used: 0, resolve, reject }; });
      return await this.infer('reference', pcm);
    } catch (error) { if (error.code === 'CANCELLED') return { status: 'Cancelled' }; throw error; }
    finally { this.state.referenceIntent = false; this.notify(); }
  }
  async check() {
    this.guard();
    if (!this.state.active) throw new Error('Start the microphone first.');
    if (this.ring.count < QUERY_SAMPLES) throw new Error('Speak for at least 1.52 seconds first.');
    return await this.infer('score', this.ring.tail(QUERY_SAMPLES));
  }
  async fromPcm(type, pcm) { this.guard(); return await this.infer(type, pcm); }
  async file(type, file) {
    this.guard(); this.stop(); this.state.busy = true; this.notify();
    const revision = this.revision;
    try {
      const pcm = await decodeFile(file);
      if (revision !== this.revision) return { status: 'Cancelled' };
      return await this.infer(type, pcm);
    }
    finally { this.state.busy = false; this.notify(); }
  }
  async infer(type, pcm) {
    if (!['reference', 'score'].includes(type)) throw new Error('Unknown operation.');
    this.state.busy = true; this.state.error = ''; this.state.status = 'Processing on this device’s CPU…'; this.notify();
    const revision = this.revision;
    try {
      const result = await this.rpc(type, { pcm: pcm.slice() });
      if (this.state.clearing || revision !== this.revision) return result;
      if (result.references) {
        this.state.references = result.references; this.state.last = null; this.ring.clear();
        this.state.status = this.state.active ? 'Reference added. Say something new to see your scores.' : 'Reference added. Start the microphone to try a new sentence.';
      } else {
        this.state.last = { ...result, at: new Date().toISOString() };
        this.state.status = result.status === 'Uncalibrated' ? 'Similarity updated · uncalibrated.' : result.status === 'LowLevel' ? 'Audio level is very low. Move closer or speak a little louder.' : result.status === 'NeedsReference' ? 'Capture your reference first.' : 'Not enough usable audio.';
      }
      return result;
    } finally { this.state.busy = false; this.notify(); }
  }
  async clear() {
    if (this.state.clearing || !this.state.ready) throw new Error('The model is not ready to clear.');
    ++this.revision; this.state.clearing = true; this.stop(); this.notify();
    try {
      await this.rpc('clear'); this.state.references = []; this.state.last = null; this.state.captureSamples = 0;
      this.state.status = 'References cleared. No audio or vectors were saved.';
    } finally { this.state.clearing = false; this.notify(); }
  }
}
