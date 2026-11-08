// Per-phrase speaker decisions and wake-word enrollment for the clip test page.
// The host owns the microphone, its ring buffer, Silero segmentation and the one-shot matcher.
import { RATE, PHRASE, MAX_REFERENCES, toEmbedding } from './clips-core.mjs';

export class PhraseDecider {
  constructor(host) { this.host = host; this.queue = []; this.busy = false; this.enrollment = null; }
  reset() { this.queue = []; this.host.state.phrases = []; }
  enqueue(event) { this.queue.push(event); void this.pump(); }
  cancelEnrollment() { if (this.enrollment) this.finishEnrollment(new Error('Recording cancelled.')); }
  arrivalOf(sample) { return (this.host.arrivals.find(([total]) => total >= sample) ?? this.host.arrivals.at(-1) ?? [0, performance.now()])[1]; }
  async pump() {
    if (this.busy) return;
    this.busy = true;
    try { while (this.queue.length) await this.decidePhrase(this.queue.shift()); }
    catch (error) { this.host.showError(error); }
    finally { this.busy = false; this.host.notify(); }
  }
  phrasePcm(event) {
    const oldest = Math.max(0, this.host.micTotal - this.host.ring.length + 1);
    const start = Math.max(oldest, event.start - PHRASE.pre), end = Math.min(this.host.micTotal, event.end + PHRASE.post);
    const pcm = new Float32Array(Math.max(0, end - start));
    for (let i = 0; i < pcm.length; i++) pcm[i] = this.host.ring[(start + i) % this.host.ring.length];
    return { pcm, start, end };
  }
  async decidePhrase(event) {
    const session = this.host.micSession, speechSeconds = (event.end - event.start) / RATE;
    const record = { start: event.start, end: event.end, speechSeconds, reason: event.reason, endWaitMs: (event.decidedAt - event.end) / RATE * 1000 };
    const { pcm, start, end } = this.phrasePcm(event);
    if (event.short || pcm.length < PHRASE.minSpan) return this.addPhrase(session, { ...record, status: 'too_short' });
    const result = await this.host.rpc('embed', { pcm, minimum: PHRASE.minSpan });
    if (session !== this.host.micSession || result.status !== 'Ready') return this.addPhrase(session, { ...record, status: 'too_short' });
    const vector = toEmbedding(result.vector), computeMs = result.computeMs;
    if (this.host.state.enrolling) return this.enrollTake(vector, speechSeconds);
    let evidence = null;
    if (this.host.phraseMatcher) evidence = JSON.parse(this.host.phraseMatcher.evaluate(vector, JSON.stringify({ generation: 1, start_sample: start, end_sample: end, activity: 'speech' })));
    this.addPhrase(session, { ...record, status: evidence ? 'decided' : 'no_references', evidence, computeMs, latencyMs: performance.now() - this.arrivalOf(event.end) });
  }
  addPhrase(session, record) {
    if (session !== this.host.micSession) return;
    this.host.state.phrases = [{ ...record, at: Date.now() }, ...this.host.state.phrases].slice(0, 10); this.host.notify();
  }
  /** Record `takes` short phrases (e.g. the wake word) from the microphone as references. */
  async referenceWakeTakes(id, takes = 3) {
    const speaker = this.host.speaker(id);
    if (this.host.state.enrolling || this.host.state.capture) throw new Error('A recording is already running.');
    const need = Math.min(takes, MAX_REFERENCES - speaker.refs.length), stopAfter = !this.host.state.mic.active;
    if (this.host.state.mic.active && this.host.state.mic.mode !== 'phrase') throw new Error('Switch the microphone to per-phrase mode or stop it first.');
    const done = new Promise((resolve, reject) => { this.enrollment = { id, need, got: 0, resolve, reject, stopAfter, timer: setTimeout(() => this.finishEnrollment(new Error('No phrase heard for 30 seconds.')), 30000) }; });
    this.host.state.enrolling = { id, need, got: 0 };
    this.host.state.status = 'Say the wake word for ' + speaker.name + ', then pause. Take 1 of ' + need + '.'; this.host.notify();
    if (stopAfter && !await this.host.startMicrophone()) this.finishEnrollment(new Error('Microphone unavailable.'));
    return await done;
  }
  enrollTake(vector, speechSeconds) {
    const e = this.enrollment, speaker = this.host.state.speakers.find(s => s.id === e?.id);
    if (!e || !speaker) return;
    speaker.refs.push({ vector, source: 'wake', seconds: speechSeconds }); e.got++; this.host.state.enrolling.got = e.got;
    this.host.state.status = e.got < e.need ? 'Got take ' + e.got + ' (' + speechSeconds.toFixed(1) + ' s). Say it again: take ' + (e.got + 1) + ' of ' + e.need + '.' : 'Added ' + e.need + ' wake-word references for ' + speaker.name + '.';
    this.host.rebank();
    if (e.got >= e.need) this.finishEnrollment(null);
  }
  finishEnrollment(error) {
    const e = this.enrollment;
    if (!e) return;
    this.enrollment = null; this.host.state.enrolling = null; clearTimeout(e.timer);
    if (e.stopAfter && this.host.state.mic.active) { this.host.stopMicrophone(); }
    if (error) e.reject(error); else e.resolve(e.got);
    this.host.notify();
  }
}
