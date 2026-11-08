import { ClipSpeakerTest } from './clips-controller.mjs';
import { RATE, HOP, WINDOW, VAD_FRAME, DEFAULT_SETTINGS, MAX_SPEAKERS, MAX_REFERENCES, describe, segments, speechSegments, toVtt, toJson, cellEnd } from './clips-core.mjs';

const el = id => document.getElementById(id);
const make = (tag, props = {}, ...children) => { const node = Object.assign(document.createElement(tag), props); node.append(...children); return node; };
const seconds = s => (s < 60 ? s.toFixed(1) + ' s' : Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0'));
const MIC_VIEW = 30 * RATE;
let test, speakerKey = '', timelineKey = '', nowKey = '', playheads = [];

async function action(fn) {
  try { test.state.error = ''; return await fn(); }
  catch (error) { test.showError(error); return { error: error.message }; }
  finally { render(); }
}

function renderSpeakers(s) {
  const key = JSON.stringify([s.speakers.map(sp => [sp.id, sp.name, sp.refs.map(r => [r.source, r.start, r.end, r.otherClip])]), Boolean(s.clip), s.capture?.id, s.enrolling, s.ready, s.mic.active, s.mic.mode]);
  if (key === speakerKey) return;
  speakerKey = key;
  el('speaker-count').textContent = s.speakers.length + ' / ' + MAX_SPEAKERS;
  el('add').disabled = !s.ready || s.speakers.length >= MAX_SPEAKERS;
  if (!s.speakers.length) { el('speakers').replaceChildren(make('li', { className: 'hint', textContent: 'No speakers yet. Add a name, then take a reference while that person talks.' })); return; }
  el('speakers').replaceChildren(...s.speakers.map(sp => speakerItem(sp, s)));
}

const refLabel = r => r.source === 'clip' ? (r.otherClip ? 'earlier clip' : seconds(r.start / RATE) + '–' + seconds(r.end / RATE)) : r.source === 'wake' ? 'wake word ' + r.seconds.toFixed(1) + ' s' : 'microphone sentence';

function button(text, label, disabled, run) {
  const node = make('button', { textContent: text, disabled });
  node.setAttribute('aria-label', label); node.onclick = () => action(run);
  return node;
}

function speakerItem(sp, s) {
  const full = sp.refs.length >= MAX_REFERENCES, busy = Boolean(s.capture || s.enrolling);
  const recordingHere = s.enrolling?.id === sp.id;
  const actions = make('div', { className: 'actions' },
    button(recordingHere ? 'Say it now… ' + (s.enrolling.got + 1) + '/' + s.enrolling.need : 'Record wake word ×3', 'Record the wake word three times for ' + sp.name,
      full || busy || (s.mic.active && s.mic.mode !== 'phrase'), () => test.referenceWakeTakes(sp.id, 3)),
    button(s.capture?.id === sp.id ? 'Recording…' : 'Record sentence (3 s)', 'Record a 3 second sentence from the microphone for ' + sp.name, full || busy, () => test.referenceFromMicrophone(sp.id)),
    button('Reference from clip', 'Reference for ' + sp.name + ' from the 3 seconds before the playhead', !s.clip || full || busy, () => test.referenceFromClip(sp.id)),
    button('Remove', 'Remove ' + sp.name, false, () => test.removeSpeaker(sp.id)));
  const refs = make('ul', { className: 'refs' }, ...sp.refs.map((r, i) =>
    make('li', {}, make('span', { textContent: refLabel(r) }), button('×', 'Remove reference ' + (i + 1) + ' of ' + sp.name, false, () => test.removeReference(sp.id, i)))));
  const count = sp.refs.length ? sp.refs.length + ' reference' + (sp.refs.length > 1 ? 's' : '') : 'needs a reference';
  return make('li', {}, make('div', { className: 'speaker-head' }, make('strong', { textContent: sp.name }), make('span', { textContent: count })), sp.refs.length ? refs : '', actions);
}

function renderNow(s) {
  const names = test.names(), current = test.current(), usable = s.speakers.some(sp => sp.refs.length);
  let kind = 'pending', text = '—', detail = '';
  if (!usable) { text = s.speakers.length ? 'Add a reference' : 'Add a speaker'; detail = 'Labels appear once at least one speaker has a reference.'; }
  else if (current.source === 'phrase') { ({ kind, text, detail } = phraseNow(current, names)); }
  else if (current.source === 'clip' && !s.clip) { text = 'Load a clip'; }
  else if (current.before) { text = 'Starting…'; detail = 'The first label needs 1.5 seconds of audio.'; }
  else if (!current.evidence) { text = current.source === 'microphone' ? 'Listening…' : 'Analysing…'; detail = current.source === 'clip' ? 'Play the clip or analyse the whole clip.' : ''; }
  else {
    const d = describe(current.evidence, names); kind = d.kind; text = d.text;
    const end = current.evidence.observation.end_sample;
    detail = (current.source === 'clip' ? 'Window ' + seconds((end - WINDOW) / RATE) + '–' + seconds(end / RATE) : 'Microphone · latest 1.5 s') + (current.evidence.status === 'tentative' && current.evidence.candidates?.[0]?.nearest >= s.settings.confident ? ' · confirming' : '');
  }
  const now = el('now');
  if (now.textContent !== text) now.textContent = text;
  now.dataset.kind = kind; el('now-detail').textContent = detail;
  const candidates = new Map((current.evidence?.candidates ?? []).map(c => [c.voice_id, c.nearest]));
  const rows = s.speakers.filter(sp => sp.refs.length).map(sp => {
    const value = candidates.get(sp.id);
    const meter = make('meter', { min: 0, max: 1, low: s.settings.tentative, high: s.settings.confident, optimum: 1, value: Number.isFinite(value) ? Math.max(0, value) : 0 });
    meter.setAttribute('aria-label', sp.name + ' similarity');
    return make('div', { className: 'score-row' }, make('span', { className: 'score-name', textContent: sp.name }), meter, make('output', { className: 'score-value', textContent: Number.isFinite(value) ? value.toFixed(2) : '—' }));
  });
  el('scores').replaceChildren(...rows);
}

function phraseNow(current, names) {
  if (current.speaking) return { kind: 'pending', text: 'Speaking…', detail: 'A decision follows when you pause.' };
  if (!current.phrase) return { kind: 'pending', text: 'Listening…', detail: 'Say the wake word or a short phrase, then pause.' };
  const p = current.phrase;
  if (p.status === 'too_short') return { kind: 'quiet', text: 'Too short', detail: p.speechSeconds.toFixed(2) + ' s of speech; at least 0.3 s is needed.' };
  const d = describe(p.evidence, names);
  return { kind: d.kind, text: d.text, detail: phraseTiming(p) };
}

const phraseTiming = p => p.speechSeconds.toFixed(2) + ' s phrase · shown ' + (p.latencyMs / 1000).toFixed(2) + ' s after you stopped (pause detection ' + (p.endWaitMs / 1000).toFixed(2) + ' s, CPU ' + Math.round(p.computeMs) + ' ms)';

function renderPhrases(s) {
  const names = test.names();
  el('phrases').replaceChildren(...s.phrases.slice(0, 6).map(p => {
    if (p.status !== 'decided') return make('li', {}, make('span', { textContent: p.status === 'too_short' ? 'Too short (' + p.speechSeconds.toFixed(2) + ' s)' : 'No references yet' }));
    const d = describe(p.evidence, names), [top, next] = p.evidence.candidates;
    const scores = top ? ' ' + top.nearest.toFixed(2) + (next ? ' · next ' + (names.get(next.voice_id) ?? next.voice_id) + ' ' + next.nearest.toFixed(2) : '') : '';
    return make('li', {}, make('strong', { className: d.kind === 'speaker' ? '' : 'unknown', textContent: d.text }), make('span', { textContent: scores }), make('span', { className: 'meta', textContent: phraseTiming(p) }));
  }));
  el('mode-hint').textContent = s.mic.mode === 'phrase' ? 'Per phrase: one decision each time you pause, like after a wake word, with the measured delay. Phrases need at least 0.3 s of speech.' : 'Continuous: a label every 0.5 s from the latest 1.5 s; needs two agreeing windows, so names appear about 1.5–2.5 s after someone starts talking.';
}

function lane(name, minor, segs, duration, offset, refs = []) {
  const track = make('div', { className: 'track' });
  track.setAttribute('role', 'img');
  track.setAttribute('aria-label', name + ': ' + (segs.length ? segs.map(g => seconds((g.start - offset) / RATE) + '–' + seconds((g.end - offset) / RATE)).join(', ') : 'none'));
  const pct = v => Math.max(0, Math.min(100, (v - offset) / duration * 100));
  for (const g of segs) {
    if (g.end <= offset) continue;
    const seg = make('div', { className: 'seg ' + (g.cls ?? ''), title: name + ' · ' + seconds((g.start - offset) / RATE) + '–' + seconds((g.end - offset) / RATE) });
    seg.style.left = pct(g.start) + '%'; seg.style.width = Math.max(0.15, pct(g.end) - pct(g.start)) + '%'; track.append(seg);
  }
  for (const r of refs) { const box = make('div', { className: 'ref', title: 'Reference used for ' + name }); box.style.left = pct(r.start) + '%'; box.style.width = (pct(r.end) - pct(r.start)) + '%'; track.append(box); }
  const head = make('div', { className: 'playhead' }); track.append(head); playheads.push(head);
  track.onclick = event => {
    if (!test.media || test.state.mic.active) return;
    const box = track.getBoundingClientRect();
    test.media.currentTime = Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)) * duration / RATE;
  };
  return [make('div', { className: 'lane-name' + (minor ? ' minor' : ''), textContent: name }), track];
}

function merge(cells, cls) {
  const out = [];
  for (const k of [...cells].sort((a, b) => a - b)) { const start = cellEnd(k) - HOP, end = cellEnd(k), last = out.at(-1); if (last && last.end === start) last.end = end; else out.push({ start, end, cls }); }
  return out;
}

function renderTimeline(s) {
  const mic = s.mic.active && s.mic.listening, track = mic ? test.micTrack : test.clip;
  const total = mic ? Math.max(test.micTotal, MIC_VIEW) : test.pcm?.length ?? 0;
  const vadDone = mic ? test.micVadDone : test.clipVad.done;
  const key = JSON.stringify([mic, mic ? Math.floor(test.micTotal / HOP) : 0, s.clip?.name, total, track.labels.size, track.cache.size, track.labeler?.generation?.toString(), speakerKey, Math.floor(vadDone / 64), s.settings.vadThreshold]);
  el('json').disabled = el('vtt').disabled = mic || !test.clip.labels.size;
  if (key === timelineKey) return;
  timelineKey = key; playheads = [];
  if (!total || (!mic && !s.clip)) { el('timeline').replaceChildren(make('p', { className: 'hint', textContent: 'Load a clip to see the timeline.' })); return; }
  const duration = mic ? MIC_VIEW : total, offset = mic ? Math.max(0, test.micTotal - MIC_VIEW) : 0;
  const names = test.names(), segs = segments(track.labels, names);
  const children = [];
  for (const sp of s.speakers) {
    const refs = mic ? [] : sp.refs.filter(r => r.source === 'clip' && !r.otherClip);
    children.push(...lane(sp.name, false, segs.filter(g => g.kind === 'speaker' && g.id === sp.id), duration, offset, refs));
  }
  children.push(...lane('Unknown voice', true, segs.filter(g => g.kind === 'unknown').map(g => ({ ...g, cls: 'unknown' })), duration, offset));
  children.push(...lane('Unsure', true, segs.filter(g => g.kind === 'unsure').map(g => ({ ...g, cls: 'unsure' })), duration, offset));
  if (!mic && test.truth) {
    for (const voice of test.truth.voices) children.push(...lane('Script: ' + voice.label, true, test.truth.turns.filter(t => t.voice === voice.id).map(t => ({ start: Math.round(t.start * RATE), end: Math.round(t.end * RATE), cls: 'truth' })), duration, offset));
  }
  const prob = mic ? f => test.micProb(f) : f => test.clipProb(f), fromFrame = mic ? Math.max(0, test.micVadDone - Math.floor(MIC_VIEW / VAD_FRAME)) : 0;
  children.push(...lane('Speech (VAD)', true, speechSegments(prob, fromFrame, vadDone, s.settings.vadThreshold).map(g => ({ ...g, cls: 'speech' })), duration, offset));
  children.push(...lane('Analysed', true, merge(track.cache.keys(), 'cover'), duration, offset));
  children.push(make('span'), make('div', { className: 'axis' }, make('span', { textContent: mic ? '−30 s' : '0:00' }), make('span', { textContent: mic ? 'now' : seconds(total / RATE) })));
  el('timeline').replaceChildren(...children);
  el('timeline-note').textContent = mic ? 'Microphone · last 30 seconds' : 'Each block shows the newest 0.5 s of a 1.5 s window · click to seek';
}

function movePlayheads() {
  if (test?.state) {
    const mic = test.state.mic.active && test.state.mic.listening;
    const left = mic ? 100 : test.pcm ? test.playhead() / test.pcm.length * 100 : 0;
    for (const head of playheads) head.style.left = left + '%';
    if (test.media && !test.media.paused) {
      const c = test.current(), key = c.cell + ':' + (c.evidence ? 1 : 0);
      if (key !== nowKey) { nowKey = key; renderNow(test.state); }
    }
  }
  requestAnimationFrame(movePlayheads);
}

const SETTINGS = [['confident', 'confident', v => v.toFixed(2)], ['tentative', 'tentative', v => v.toFixed(2)], ['margin', 'margin', v => v.toFixed(2)], ['stable', 'stableSeconds', v => v.toFixed(1) + ' s'], ['vad', 'vadThreshold', v => v.toFixed(2)], ['speech', 'minSpeech', v => Math.round(v * 100) + '%']];

function renderControls(s) {
  const listening = s.mic.active && s.mic.listening;
  el('demo').disabled = el('file').disabled = !s.ready || s.mic.active;
  el('analyze').disabled = !s.ready || !s.clip || s.analysis.running || s.mic.active;
  el('analyze').hidden = s.analysis.running;
  el('stop-analysis').hidden = !s.analysis.running;
  el('mic').disabled = !s.ready || s.mic.pending || Boolean(s.capture);
  el('mic').hidden = listening;
  el('mic-stop').hidden = !listening;
  for (const mode of ['phrase', 'continuous']) { el('mode-' + mode).setAttribute('aria-pressed', String(s.mic.mode === mode)); el('mode-' + mode).disabled = s.mic.active; }
  el('mic-level').textContent = s.mic.active ? 'Level ' + (20 * Math.log10(Math.max(s.mic.rms ?? 0, 1e-6))).toFixed(0) + ' dBFS' : '';
}

function progressText(s) {
  const vad = test.clipVad, windows = test.pcm ? Math.max(1, Math.floor((test.pcm.length - WINDOW) / HOP) + 1) : 1;
  if (vad.frames && vad.done < vad.frames) return 'Detecting speech ' + Math.round(vad.done / vad.frames * 100) + '%';
  if (s.analysis.running) return 'Analysed ' + Math.round(test.clip.cache.size / windows * 100) + '% on this CPU';
  if (s.analysis.done) return 'Whole clip analysed';
  return s.lastComputeMs ? 'Last window: ' + s.lastComputeMs.toFixed(0) + ' ms CPU' : '';
}

function renderSettings(s) {
  for (const [id, key, format] of SETTINGS) {
    if (document.activeElement !== el(id)) el(id).value = s.settings[key];
    el(id + '-value').textContent = format(s.settings[key]);
  }
}

function render() {
  if (!test) return;
  const s = test.state;
  if (lastClip !== s.clip) { lastClip = s.clip; mountMedia(s.clip); }
  el('status').textContent = s.status;
  el('error').hidden = !s.error; el('error').textContent = s.error;
  el('progress').textContent = progressText(s);
  el('clip-name').textContent = s.clip ? s.clip.name + ' · ' + seconds(s.clip.seconds) : 'No clip loaded. The demo has three fictional synthetic voices taking turns.';
  renderControls(s); renderSettings(s); renderSpeakers(s); renderNow(s); renderPhrases(s); renderTimeline(s);
}

function mountMedia(clip) {
  const player = el('player');
  if (!clip?.url) { player.replaceChildren(); test.attachMedia(null); return; }
  const media = make(clip.video ? 'video' : 'audio', { controls: true, preload: 'auto', src: clip.url });
  media.setAttribute('aria-label', 'Clip player');
  if (clip.video) media.playsInline = true;
  media.onseeked = media.onpause = media.onplay = () => render();
  player.replaceChildren(media); test.attachMedia(media);
}

function download(name, type, body) {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const link = make('a', { href: url, download: name }); document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function exportArgs() {
  const s = test.state, names = test.names();
  return { clip: { name: s.clip?.name, seconds: s.clip?.seconds }, revision: s.revision, settings: s.settings, labels: test.clip.labels, names,
    references: s.speakers.flatMap(sp => sp.refs.map(r => ({ id: sp.id, source: r.source, start: r.otherClip ? null : r.start, end: r.otherClip ? null : r.end }))) };
}

let lastClip = null;
test = new ClipSpeakerTest(() => render());

el('file').onchange = () => action(async () => { const file = el('file').files[0]; if (file) await test.loadFile(file); });
el('demo').onclick = () => action(() => test.loadDemo());
el('add-form').onsubmit = event => { event.preventDefault(); void action(() => { test.addSpeaker(el('name').value); el('name').value = ''; el('name').focus(); }); };
el('analyze').onclick = () => action(() => test.analyze());
el('stop-analysis').onclick = () => action(() => test.stopAnalysis());
el('mic').onclick = () => action(() => test.startMicrophone());
for (const mode of ['phrase', 'continuous']) el('mode-' + mode).onclick = () => action(() => test.setMode(mode));
el('mic-stop').onclick = () => action(() => test.stopMicrophone());
el('reset-settings').onclick = () => action(() => test.setSettings({ ...DEFAULT_SETTINGS }));
for (const [id, key] of SETTINGS) {
  el(id).oninput = () => action(() => test.setSettings({ [key]: Number(el(id).value) }));
}
el('json').onclick = () => download('speaker-timeline.json', 'application/json', toJson(exportArgs()));
el('vtt').onclick = () => download('speakers.vtt', 'text/vtt', toVtt(segments(test.clip.labels, test.names())));

// Semantic operations for agents and automated checks; same code paths as the controls.
window.speakerClips = {
  snapshot: () => test.snapshot(),
  ready: () => test.loading.then(() => test.state.ready),
  loadDemo: () => action(() => test.loadDemo()),
  addSpeaker: name => test.addSpeaker(name),
  referenceFromClip: (id, endSeconds) => test.referenceFromClip(id, endSeconds),
  analyze: () => test.analyze(),
  setMode: mode => test.setMode(mode),
  referenceWakeTakes: (id, n) => test.referenceWakeTakes(id, n),
  phrases: () => test.state.phrases.map(({ evidence, ...p }) => ({ ...p, label: evidence ? describe(evidence, test.names()) : null, candidates: evidence?.candidates ?? [] })),
  setSettings: settings => test.setSettings(settings),
  seek: t => { if (test.media) test.media.currentTime = t; },
  current: () => { const c = test.current(); return { ...c, label: describe(c.evidence, test.names()) }; },
  segments: () => segments(test.clip.labels, test.names()),
  exportJson: () => toJson(exportArgs()),
  exportVtt: () => toVtt(segments(test.clip.labels, test.names())),
};
window.addEventListener('pagehide', () => test.stopMicrophone());
render();
requestAnimationFrame(movePlayheads);
