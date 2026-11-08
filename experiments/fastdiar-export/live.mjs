import { LiveSpeakerTest } from './live-controller.mjs';
import { QUERY_SAMPLES, REFERENCE_SAMPLES } from './live-core.mjs';
const el = id => document.getElementById(id);
let controller;
const number = value => Number.isFinite(value) ? value.toFixed(3) : '—';
function render(s) {
  const blocked = !s.ready || s.busy || s.clearing || s.micPending || s.referenceIntent;
  el('start').disabled = blocked || s.active;
  el('stop').disabled = !s.active && !s.micPending;
  el('reference').disabled = blocked || s.references.length >= 7;
  el('clear').disabled = !s.ready || s.clearing || (!s.references.length && !s.busy && !s.referenceIntent);
  el('check').disabled = blocked || !s.active || !s.references.length || controller.ring.count < QUERY_SAMPLES;
  el('file-reference').disabled = blocked || !el('file').files.length || s.references.length >= 7;
  el('file-check').disabled = blocked || !el('file').files.length || !s.references.length;
  el('status').textContent = s.status;
  el('error').hidden = !s.error; el('error').textContent = s.error;
  el('level').value = s.rms; el('level-value').textContent = number(s.rms);
  el('reference-count').textContent = s.references.length + ' / 7';
  el('reference-list').replaceChildren(...(s.references.length ? s.references.map(r => {
    const item = document.createElement('li'); item.textContent = 'Reference ' + r.number + ' · ' + r.seconds.toFixed(2) + ' seconds · this browser only'; return item;
  }) : [Object.assign(document.createElement('li'), { textContent: 'No reference yet. Say a short sentence after pressing Capture.' })]));
  el('capture-progress').textContent = s.referenceIntent ? 'Reference audio: ' + (s.captureSamples / 16000).toFixed(2) + ' / ' + (REFERENCE_SAMPLES / 16000).toFixed(2) + ' seconds' : 'Use another reference for a different distance or speaking style. Nothing is saved.';
  for (const name of ['nearest', 'centroid']) {
    el(name).textContent = number(s.last?.[name]); el(name + '-meter').value = Number.isFinite(s.last?.[name]) ? s.last[name] : -1;
  }
  el('score-note').textContent = s.last?.status === 'Uncalibrated' ? 'Last check ' + new Date(s.last.at).toLocaleTimeString() + ' · raw cosine similarities, not probabilities or identity decisions.' : s.references.length ? 'Say a new sentence. Live checks use the latest 1.52 seconds.' : 'Capture a reference first, then say something new. No identity or acceptance decision is made.';
  el('timing').textContent = s.last?.computeMs !== undefined ? 'Latest CPU inference: ' + s.last.computeMs.toFixed(1) + ' ms · ' + (s.last.samples / 16000).toFixed(2) + ' seconds observed · ' + s.last.paddedSamples + ' padding samples' : '';
  if (s.model) el('model-info').textContent = 'FASTDIAR-small · ' + (s.model.modelBytes / 1e6).toFixed(2) + ' MB model file (not RAM) · WASM CPU, one thread · 192 dimensions · 80 ms frames. Uncalibrated temporary bank; not your saved voqx profiles.';
}
controller = new LiveSpeakerTest(render);
window.speakerTest = {
  snapshot: () => controller.snapshot(), start: () => controller.start(), stop: () => controller.stop(),
  captureReference: () => controller.reference(), check: () => controller.check(), clear: () => controller.clear(),
  referencePcm: pcm => controller.fromPcm('reference', pcm), queryPcm: pcm => controller.fromPcm('score', pcm),
};
async function action(fn) {
  try { return await fn(); } catch (error) { controller.showError(error); return { status: 'Error', message: error.message }; }
}
el('start').onclick = () => action(() => controller.start());
el('stop').onclick = () => controller.stop();
el('reference').onclick = () => action(() => controller.reference());
el('check').onclick = () => action(() => controller.check());
el('clear').onclick = () => action(() => controller.clear());
el('file').onchange = () => render(controller.snapshot());
el('file-reference').onclick = () => action(() => controller.file('reference', el('file').files[0]));
el('file-check').onclick = () => action(() => controller.file('score', el('file').files[0]));
setInterval(() => {
  const s = controller.state;
  if (el('auto').checked && s.active && s.references.length && !s.busy && !s.referenceIntent && !s.clearing && controller.ring.count >= QUERY_SAMPLES) void action(() => controller.check());
}, 2000);
window.addEventListener('pagehide', () => controller.stop());
document.addEventListener('visibilitychange', () => { if (document.hidden) controller.stop(); });
render(controller.snapshot());
