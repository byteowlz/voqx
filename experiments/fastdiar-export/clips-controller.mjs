// Clip/microphone speaker attribution with temporary named references.
// Encoder: FASTDIAR in a CPU-only ORT Web worker. Matching/stabilization: the voqx Rust/WASM library.
// Embeddings are cached per 0.5 s cell, so reference and threshold changes only re-run matching.
import init, { SpeakerAttributor } from '/voqx/voqx_recognition.js';
import { HeadMicrophone } from '/heads-capture.mjs';
import { WorkerRpc } from './worker-rpc.mjs';
import { PhraseDecider } from './phrase-decider.mjs';
import {
  RATE, WINDOW, REFERENCE, HOP, MAX_SPEAKERS, MAX_REFERENCES, MAX_CLIP_SECONDS, DEFAULT_SETTINGS,
  cellEnd, cellCount, cellAt, vadActivity, VAD_FRAME, toEmbedding, snapshotJson, calibrationJson,
  PhraseSegmenter, PHRASE, PHRASE_SPACE,
} from './clips-core.mjs';

const MIC_SECONDS = 30, MIC_VAD_FRAMES = 1024;

class Track {
  constructor(kind) { this.kind = kind; this.reset(); }
  reset() { this.cache = new Map(); this.labels = new Map(); this.labeler = null; this.revision = (this.revision ?? 0) + 1; }
}

export class ClipSpeakerTest {
  constructor(changed) {
    this.changed = changed;
    this.state = { ready: false, status: 'Loading the local CPU model…', error: '', speakers: [], settings: { ...DEFAULT_SETTINGS },
      clip: null, analysis: { running: false }, mic: { active: false, pending: false, total: 0, mode: 'phrase' }, capture: null, revision: null, busy: false, phrases: [], enrolling: null };
    this.clip = new Track('clip'); this.micTrack = new Track('mic');
    this.pcm = null; this.media = null; this.truth = null;
    this.microphone = new HeadMicrophone(); this.ring = new Float32Array(RATE * MIC_SECONDS); this.micTotal = 0;
    this.generation = 0n; this.inflight = false; this.nextSpeaker = 1;
    this.worker = new WorkerRpc('/live-worker.mjs', message => this.fail(message));
    this.vad = new WorkerRpc('/vad-worker.mjs', message => this.fail(message));
    this.clipVad = { probs: new Float32Array(0), done: 0, frames: 0 };
    this.micVadRing = new Float32Array(MIC_VAD_FRAMES); this.micVadDone = 0; this.micPending = []; this.micSession = 0;
    this.segmenter = new PhraseSegmenter(DEFAULT_SETTINGS.vadThreshold); this.arrivals = [];
    this.phrases = new PhraseDecider(this);
    this.loading = Promise.all([init(), this.rpc('load'), this.vad.call('load')]).then(([, model]) => {
      this.state.model = model; this.state.ready = true;
      this.state.status = 'Ready. Load a clip, then add the speakers you want to recognise.';
      this.timer = setInterval(() => this.pump(), 100);
      this.notify();
    }).catch(error => this.fail(error.message));
  }

  // ---------- plumbing ----------
  snapshot() {
    const s = structuredClone({ ...this.state, speakers: this.state.speakers.map(sp => ({ ...sp, refs: sp.refs.map(({ vector, ...rest }) => rest) })) });
    s.clipCells = { cached: this.clip.cache.size, labeled: this.clip.labels.size, total: this.pcm ? cellCount(this.pcm.length) : 0 };
    s.micCells = { cached: this.micTrack.cache.size, labeled: this.micTrack.labels.size };
    s.vad = { clipFrames: this.clipVad.frames, clipDone: this.clipVad.done, micDone: this.micVadDone };
    return s;
  }
  notify() { this.changed?.(this); }
  showError(error) { this.state.error = error.message ?? String(error); this.notify(); }
  fail(message) {
    clearInterval(this.timer); this.stopMicrophone();
    this.state.ready = false; this.state.error = message; this.notify();
  }
  rpc(type, args = {}) { return this.worker.call(type, args); }
  names() { return new Map(this.state.speakers.map(s => [s.id, s.name])); }
  clipProb(f) { return this.clipVad.probs[f]; }
  micProb(f) { return this.micVadRing[f % MIC_VAD_FRAMES]; }
  /** Silero speech decision for one window; null until the detector has covered it. */
  activity(track, k) {
    const end = cellEnd(k);
    if (track === this.clip) return this.pcm && end <= this.pcm.length ? vadActivity(f => this.clipProb(f), this.clipVad.done, end - WINDOW, end, this.state.settings) : null;
    if (end > this.micTotal || this.micVadDone - Math.floor((end - WINDOW) / VAD_FRAME) > MIC_VAD_FRAMES) return null;
    return vadActivity(f => this.micProb(f), this.micVadDone, end - WINDOW, end, this.state.settings);
  }

  // ---------- speakers and references ----------
  addSpeaker(name) {
    const clean = String(name ?? '').trim().slice(0, 40);
    if (!clean) throw new Error('Enter a name first.');
    if (this.state.speakers.length >= MAX_SPEAKERS) throw new Error('Eight speakers maximum in this test.');
    if (this.state.speakers.some(s => s.name.toLowerCase() === clean.toLowerCase())) throw new Error('That name is already in the list.');
    const speaker = { id: 'tmp-' + this.nextSpeaker++, name: clean, refs: [] };
    this.state.speakers.push(speaker); this.state.error = ''; this.rebank();
    return speaker.id;
  }
  removeSpeaker(id) { this.state.speakers = this.state.speakers.filter(s => s.id !== id); this.rebank(); }
  removeReference(id, index) {
    const speaker = this.speaker(id); speaker.refs.splice(index, 1); this.rebank();
  }
  speaker(id) {
    const speaker = this.state.speakers.find(s => s.id === id);
    if (!speaker) throw new Error('Unknown speaker.');
    if (speaker.refs.length >= MAX_REFERENCES) throw new Error('Seven references per speaker maximum.');
    return speaker;
  }
  async embedReference(pcm, speech) {
    if (!speech) throw new Error('Speech detection has not reached this moment yet. Try again in a second.');
    if (speech.activity !== 'speech') throw new Error('The speech detector found too little speech here (' + Math.round(speech.fraction * 100) + '%). Pick a moment where only this person is talking.');
    const result = await this.rpc('embed', { pcm: pcm.slice() });
    if (result.status !== 'Ready') throw new Error('Not enough usable audio for a reference.');
    return { vector: toEmbedding(result.vector), computeMs: result.computeMs };
  }
  /** Reference = the 3.04 s (or at least 1.52 s) of the clip ending at `endSeconds`. */
  async referenceFromClip(id, endSeconds = this.playhead() / RATE) {
    this.speaker(id);
    if (!this.pcm) throw new Error('Load a clip first.');
    const end = Math.min(this.pcm.length, Math.round(endSeconds * RATE)), start = Math.max(0, end - REFERENCE);
    if (end - start < WINDOW) throw new Error('Play at least 1.5 seconds of the clip first.');
    const revision = this.clip.revision;
    const ref = await this.embedReference(this.pcm.subarray(start, end), vadActivity(f => this.clipProb(f), this.clipVad.done, start, end, this.state.settings));
    if (revision !== this.clip.revision) throw new Error('The clip changed; reference discarded.');
    const speaker = this.speaker(id);
    speaker.refs.push({ ...ref, source: 'clip', start, end }); this.rebank();
    this.state.status = 'Added a reference for ' + speaker.name + ' (' + (start / RATE).toFixed(1) + '–' + (end / RATE).toFixed(1) + ' s).';
    this.notify(); return speaker.refs.length;
  }
  referenceWakeTakes(id, takes = 3) { return this.phrases.referenceWakeTakes(id, takes); }
  async referenceFromMicrophone(id) {
    const speaker = this.speaker(id);
    if (this.state.capture) throw new Error('A recording is already running.');
    const wasActive = this.state.mic.active;
    const pcm = new Float32Array(REFERENCE);
    try {
      const done = new Promise((resolve, reject) => { this.state.capture = { id, used: 0 }; this.captureBuffer = { pcm, resolve, reject }; });
      this.state.status = 'Recording ' + speaker.name + ' for three seconds — ask them to speak normally.'; this.notify();
      if (!wasActive && !await this.startMicrophone({ listen: false })) throw new Error('Microphone unavailable.');
      await done;
    } finally {
      this.state.capture = null; this.captureBuffer = null;
      if (!wasActive) this.stopMicrophone();
      this.notify();
    }
    const stream = 'ref-' + (++this.micSession);
    const { probs } = await this.vad.call('scan', { stream, reset: true, pcm: pcm.slice() });
    void this.vad.call('drop', { stream });
    const ref = await this.embedReference(pcm, vadActivity(f => probs[f], probs.length, 0, REFERENCE, this.state.settings));
    this.speaker(id).refs.push({ ...ref, source: 'microphone' }); this.rebank();
    this.state.status = 'Added a microphone reference for ' + speaker.name + '.'; this.notify();
    return speaker.refs.length;
  }

  // ---------- matching (WASM library) ----------
  rebank() {
    const usable = this.state.speakers.filter(s => s.refs.length);
    this.bank = null; this.state.revision = null;
    if (usable.length) {
      const snapshot = snapshotJson(this.state.speakers);
      const probe = new SpeakerAttributor(snapshot, undefined, ++this.generation, 1n);
      const revision = probe.referenceRevision(); probe.free();
      const phraseSnapshot = snapshotJson(this.state.speakers, PHRASE_SPACE);
      const phraseProbe = new SpeakerAttributor(phraseSnapshot, undefined, ++this.generation, 1n);
      this.bank = { snapshot, revision, phraseSnapshot, phraseRevision: phraseProbe.referenceRevision() }; phraseProbe.free();
      this.state.revision = revision;
    }
    for (const track of [this.clip, this.micTrack]) this.relabel(track);
    this.rebuildPhraseMatcher();
    this.notify();
  }
  /** One-shot matcher for delimited phrases: same references, variable-length sandbox profile. */
  rebuildPhraseMatcher() {
    this.phraseMatcher?.free(); this.phraseMatcher = null;
    if (!this.bank) return;
    const band = [PHRASE.minSpan, PHRASE_SPACE.maximum_samples];
    this.phraseMatcher = new SpeakerAttributor(this.bank.phraseSnapshot, calibrationJson(this.bank.phraseRevision, this.state.settings, band), ++this.generation, 1n);
  }
  newLabeler() {
    if (!this.bank) return null;
    const stable = BigInt(Math.max(1, Math.round(this.state.settings.stableSeconds * RATE)));
    const generation = ++this.generation;
    return { attributor: new SpeakerAttributor(this.bank.snapshot, calibrationJson(this.bank.revision, this.state.settings), generation, stable), generation, last: -1 };
  }
  relabel(track) {
    track.labeler?.attributor.free();
    track.labels = new Map(); track.labeler = this.newLabeler();
    if (!track.labeler) return;
    for (const k of [...track.cache.keys()].sort((a, b) => a - b)) this.observe(track, k);
  }
  observe(track, k) {
    if (!track.labeler) return;
    if (k <= track.labeler.last) { this.relabel(track); return; }
    const cell = track.cache.get(k), end = cellEnd(k);
    const observation = { generation: Number(track.labeler.generation), start_sample: end - WINDOW, end_sample: end, activity: cell.activity };
    const evidence = JSON.parse(track.labeler.attributor.observe(cell.vector ?? new Float32Array(0), JSON.stringify(observation)));
    track.labeler.last = k; track.labels.set(k, evidence);
  }
  setSettings(partial) {
    const { vadThreshold, minSpeech } = this.state.settings;
    this.state.settings = { ...this.state.settings, ...partial };
    if (this.state.settings.vadThreshold !== vadThreshold || this.state.settings.minSpeech !== minSpeech) this.regate();
    this.segmenter.threshold = this.state.settings.vadThreshold;
    for (const track of [this.clip, this.micTrack]) this.relabel(track);
    this.rebuildPhraseMatcher();
    this.notify();
  }
  /** Speech settings change which windows need embeddings; recompute decisions from stored probabilities. */
  regate() {
    for (const track of [this.clip, this.micTrack]) {
      for (const [k, cell] of track.cache) {
        const speech = this.activity(track, k);
        if (!speech || (speech.activity === 'speech' && !cell.vector)) track.cache.delete(k);
        else track.cache.set(k, { ...cell, ...speech });
      }
    }
  }

  // ---------- clip source ----------
  async loadFile(file) {
    if (!file) throw new Error('Choose a file.');
    if (file.size > 600 * 1024 * 1024) throw new Error('Choose a file smaller than 600 MB, or export the audio track.');
    this.state.status = 'Decoding ' + file.name + ' on this device…'; this.notify();
    const pcm = await decode(await file.arrayBuffer());
    this.setClip({ name: file.name, pcm, url: URL.createObjectURL(file), video: file.type.startsWith('video/'), truth: null });
  }
  async loadDemo() {
    this.state.status = 'Loading the demo clip…'; this.notify();
    const [audio, truth] = await Promise.all([fetch('/demo.wav'), fetch('/demo.json')]);
    if (!audio.ok || !truth.ok) throw new Error('Demo clip unavailable on this server.');
    const pcm = await decode(await audio.arrayBuffer());
    this.setClip({ name: 'Demo: three fictional synthetic voices', pcm, url: '/demo.wav', video: false, truth: await truth.json() });
  }
  loadPcm(name, pcm) { this.setClip({ name, pcm: Float32Array.from(pcm), url: null, video: false, truth: null }); }
  setClip({ name, pcm, url, video, truth }) {
    if (pcm.length / RATE > MAX_CLIP_SECONDS) throw new Error('Clips up to 30 minutes in this test.');
    if (this.objectUrl) URL.revokeObjectURL(this.objectUrl);
    this.objectUrl = url?.startsWith('blob:') ? url : null;
    this.stopAnalysis(); this.clip.labeler?.attributor.free(); this.clip.reset();
    // References are vectors, not clip audio, so they remain valid for other clips; clip spans are no longer shown.
    for (const s of this.state.speakers) for (const r of s.refs) if (r.source === 'clip') r.otherClip = true;
    this.pcm = pcm; this.truth = truth;
    this.state.clip = { name, seconds: pcm.length / RATE, url, video, hasTruth: Boolean(truth) };
    this.state.status = 'Clip ready. Add speakers and take a reference while each one is talking.';
    this.relabel(this.clip); this.notify();
    void this.scanClip();
  }
  /** Silero over the whole clip in order, one recurrent stream; much faster than real time. */
  async scanClip() {
    const revision = this.clip.revision, pcm = this.pcm, frames = Math.floor(pcm.length / VAD_FRAME), stream = 'clip-' + revision, chunk = 256;
    this.clipVad = { probs: new Float32Array(frames), done: 0, frames };
    try {
      for (let f = 0; f < frames; f += chunk) {
        const n = Math.min(chunk, frames - f);
        const { probs } = await this.vad.call('scan', { stream, reset: f === 0, pcm: pcm.slice(f * VAD_FRAME, (f + n) * VAD_FRAME) });
        if (revision !== this.clip.revision) return;
        this.clipVad.probs.set(probs, f); this.clipVad.done = f + n;
        if (f % (chunk * 8) === 0) this.notify();
      }
      void this.vad.call('drop', { stream });
      this.notify();
    } catch (error) { if (revision === this.clip.revision) this.showError(error); }
  }
  attachMedia(element) { this.media = element; }
  playhead() {
    if (!this.pcm) return 0;
    return this.media ? Math.min(this.pcm.length, Math.max(0, Math.round(this.media.currentTime * RATE))) : this.pcm.length;
  }
  clipWindow(k) { return this.pcm && cellEnd(k) <= this.pcm.length ? this.pcm.subarray(cellEnd(k) - WINDOW, cellEnd(k)) : null; }
  analyze() {
    if (!this.pcm) throw new Error('Load a clip first.');
    if (cellCount(this.pcm.length) === 0) throw new Error('The clip is shorter than 1.5 seconds.');
    this.state.analysis = { running: true, started: performance.now() }; this.notify();
  }
  stopAnalysis() { this.state.analysis = { running: false }; this.notify?.(); }
  current() {
    if (this.state.mic.active && this.state.mic.mode === 'phrase') {
      const last = this.state.phrases[0], fresh = last && (this.micTotal - last.end) / RATE < 6;
      return { source: 'phrase', speaking: this.state.mic.speaking, phrase: fresh ? last : null, evidence: fresh ? last.evidence ?? null : null };
    }
    if (this.state.mic.active) {
      const k = cellAt(this.micTotal), labels = this.micTrack.labels;
      const latest = Math.max(-1, ...[...labels.keys()].filter(key => key >= k - 2));
      return { source: 'microphone', cell: latest, evidence: latest >= 0 ? labels.get(latest) : null, t: this.micTotal / RATE };
    }
    const sample = this.playhead(), k = cellAt(sample);
    return { source: 'clip', cell: k, evidence: k >= 0 ? this.clip.labels.get(k) ?? null : null, t: sample / RATE, before: sample < WINDOW };
  }

  // ---------- microphone source ----------
  setMode(mode) {
    if (this.state.mic.active) throw new Error('Stop the microphone before switching mode.');
    if (!['phrase', 'continuous'].includes(mode)) throw new Error('Unknown microphone mode.');
    this.state.mic.mode = mode; this.notify();
  }
  async startMicrophone({ listen = true } = {}) {
    if (this.state.mic.active || this.state.mic.pending) return this.state.mic.active;
    this.state.mic.pending = true; this.notify();
    try {
      if (listen) {
        this.micTrack.labeler?.attributor.free(); this.micTrack.reset(); this.micTotal = 0; this.ring.fill(0); this.relabel(this.micTrack); this.media?.pause();
        this.micSession++; this.micVadDone = 0; this.micPending = []; this.micVadRing.fill(0);
        this.segmenter.reset(); this.phrases.reset(); this.arrivals = []; this.state.mic.speaking = false;
      }
      this.state.mic.listening = listen;
      this.state.mic.active = await this.microphone.start(pcm => this.frame(pcm));
      if (listen && this.state.mic.active) this.state.status = this.state.mic.mode === 'phrase' ? 'Listening. Say a phrase or the wake word, then pause: one decision per phrase.' : 'Listening. Results use the latest 1.5 seconds, updated every 0.5 seconds.';
      return this.state.mic.active;
    } finally { this.state.mic.pending = false; this.notify(); }
  }
  stopMicrophone() {
    const wasListening = this.state.mic.listening;
    this.microphone.stop(); this.state.mic.active = false; this.state.mic.listening = false; this.state.mic.rms = 0;
    if (wasListening) this.state.status = 'Microphone stopped. The clip timeline is unchanged.';
    if (this.captureBuffer) { this.captureBuffer.reject(new Error('Recording cancelled.')); this.captureBuffer = null; }
    this.phrases.cancelEnrollment();
    this.state.mic.speaking = false;
    this.notify?.();
  }
  frame(pcm) {
    if (!this.state.mic.active) return;
    let sum = 0;
    for (const v of pcm) sum += v * v;
    this.state.mic.rms = Math.sqrt(sum / pcm.length);
    if (this.captureBuffer) {
      const c = this.captureBuffer, n = Math.min(pcm.length, c.pcm.length - this.state.capture.used);
      c.pcm.set(pcm.subarray(0, n), this.state.capture.used); this.state.capture.used += n;
      if (this.state.capture.used === c.pcm.length) c.resolve();
    }
    if (this.state.mic.listening) {
      for (const v of pcm) { this.ring[this.micTotal % this.ring.length] = v; this.micTotal++; }
      this.state.mic.total = this.micTotal;
      this.arrivals.push([this.micTotal, performance.now()]); if (this.arrivals.length > 400) this.arrivals.shift();
      this.micPending.push(pcm.slice()); void this.scanMicrophone();
    }
  }
  /** Streaming Silero over live audio in whole 512-sample frames, one call in flight. */
  async scanMicrophone() {
    if (this.micVadBusy) return;
    const total = this.micPending.reduce((n, c) => n + c.length, 0), n = Math.floor(total / VAD_FRAME) * VAD_FRAME;
    if (n < VAD_FRAME * 4) return;
    const all = new Float32Array(total); let at = 0;
    for (const c of this.micPending) { all.set(c, at); at += c.length; }
    this.micPending = [all.slice(n)];
    const session = this.micSession, stream = 'mic-' + session;
    this.micVadBusy = true;
    try {
      const { probs } = await this.vad.call('scan', { stream, reset: this.micVadDone === 0, pcm: all.slice(0, n) });
      if (session !== this.micSession) return;
      for (let i = 0; i < probs.length; i++) {
        const frame = this.micVadDone + i;
        this.micVadRing[frame % MIC_VAD_FRAMES] = probs[i];
        if (this.state.mic.mode === 'phrase' || this.state.enrolling) {
          const event = this.segmenter.push(frame, probs[i]);
          if (event) this.phrases.enqueue(event);
        }
      }
      this.micVadDone += probs.length;
      this.state.mic.speaking = this.segmenter.inSpeech;
    } catch (error) { this.showError(error); }
    finally { this.micVadBusy = false; }
  }
  micWindow(k) {
    const end = cellEnd(k);
    if (end > this.micTotal || this.micTotal - (end - WINDOW) > this.ring.length) return null;
    const out = new Float32Array(WINDOW);
    for (let i = 0; i < WINDOW; i++) out[i] = this.ring[(end - WINDOW + i) % this.ring.length];
    return out;
  }

  // ---------- scheduling: one CPU inference at a time ----------
  /** Newest missing window among the last `count` whose speech decision is available. */
  recent(track, k, count) {
    for (let back = 0; back < count && k - back >= 0; back++) if (!track.cache.has(k - back) && this.activity(track, k - back)) return [track, k - back];
    return null;
  }
  wholeClipJob(total) {
    const k = [...Array(total).keys()].find(i => !this.clip.cache.has(i));
    if (k !== undefined) return this.activity(this.clip, k) ? [this.clip, k] : null;
    this.state.analysis = { running: false, done: true };
    this.state.status = 'Whole clip analysed. Change references or thresholds freely; matching updates instantly.';
    this.notify(); return null;
  }
  nextJob() {
    if (this.state.mic.listening && this.state.mic.active) return this.state.mic.mode === 'continuous' && !this.state.enrolling ? this.recent(this.micTrack, cellAt(this.micTotal), 3) : null;
    if (!this.pcm) return null;
    const total = cellCount(this.pcm.length);
    if (this.state.analysis.running) return this.wholeClipJob(total);
    return this.recent(this.clip, Math.min(cellAt(this.playhead()), total - 1), 4);
  }
  async pump() {
    if (!this.state.ready || this.inflight) return;
    const job = this.nextJob();
    if (!job) { if (this.state.mic.active || (this.media && !this.media.paused)) this.notify(); return; }
    const [track, k] = job, revision = track.revision;
    const pcm = track === this.clip ? this.clipWindow(k) : this.micWindow(k), speech = this.activity(track, k);
    if (!pcm || !speech) return;
    this.inflight = true;
    try {
      let vector = null, computeMs = 0;
      // Non-speech windows never reach the speaker encoder, so they cannot produce a name.
      if (speech.activity === 'speech') {
        const result = await this.rpc('embed', { pcm: pcm.slice() });
        if (result.status === 'Ready') { vector = toEmbedding(result.vector); computeMs = result.computeMs; } else speech.activity = 'unknown';
      }
      if (revision !== track.revision) return;
      track.cache.set(k, { ...speech, vector });
      this.state.lastComputeMs = computeMs || this.state.lastComputeMs;
      this.observe(track, k);
    } catch (error) { this.showError(error); this.stopAnalysis(); }
    finally { this.inflight = false; this.notify(); }
  }
}

async function decode(buffer) {
  const decoded = await new OfflineAudioContext(1, 1, RATE).decodeAudioData(buffer);
  if (!decoded.length) throw new Error('The file contains no audio.');
  if (decoded.duration > MAX_CLIP_SECONDS) throw new Error('Clips up to 30 minutes in this test.');
  const pcm = new Float32Array(decoded.length);
  for (let c = 0; c < decoded.numberOfChannels; c++) {
    const data = decoded.getChannelData(c);
    for (let i = 0; i < pcm.length; i++) pcm[i] += data[i] / decoded.numberOfChannels;
  }
  return pcm;
}
