// Actual Chromium/WASM check of the clip speaker page on the synthetic demo clip.
// Fictional voices only; accuracy numbers describe this synthetic arrangement, not real people.
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import path from 'node:path';
import { consoleSeverity } from './console-severity.mjs';
const base = process.env.LIVE_BASE_URL ?? 'http://127.0.0.1:8068';
const { EVIDENCE_DIR: evidence, PLAYWRIGHT, CHROMIUM, DEMO_DIR } = process.env;
if (!evidence || !PLAYWRIGHT || !CHROMIUM || !DEMO_DIR) throw new Error('Set EVIDENCE_DIR (new), PLAYWRIGHT, CHROMIUM and DEMO_DIR.');
await mkdir(evidence, { recursive: false, mode: 0o700 });
const script = JSON.parse(await readFile(path.join(DEMO_DIR, 'demo.json'), 'utf8'));
const { chromium } = await import(PLAYWRIGHT);
const browser = await chromium.launch({ executablePath: CHROMIUM, headless: true, args: ['--disable-gpu', '--autoplay-policy=no-user-gesture-required', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream', '--use-file-for-fake-audio-capture=' + path.join(DEMO_DIR, 'demo.wav')] });
const checks = [], errors = [], warnings = [], requests = [], report = { format: 'voqx-speaker-clips-check/1', base };
const assert = (value, name) => { if (!value) throw new Error('FAILED: ' + name); checks.push(name); };
const shots = process.env.SKIP_SHOTS !== '1';
try {
  const context = await browser.newContext({ ignoreHTTPSErrors: process.env.ALLOW_LOCAL_CERT === '1', permissions: ['microphone'], viewport: { width: 1280, height: 1000 }, serviceWorkers: 'block' });
  await context.route('**/*', route => {
    const r = route.request(), url = new URL(r.url());
    if (url.protocol !== 'blob:') requests.push({ method: r.method(), path: url.pathname });
    if (url.protocol !== 'blob:' && url.origin !== new URL(base).origin) { errors.push('External request ' + url.origin); return route.abort(); }
    return route.continue();
  });
  const page = await context.newPage();
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', e => { if (e.type() === 'error') (consoleSeverity(e.text()) === 'warning' ? warnings : errors).push(e.text()); });
  await page.goto(base + '/clips');
  await page.waitForFunction(() => window.speakerClips && window.speakerClips.snapshot().ready, undefined, { timeout: 60000 });
  assert(await page.evaluate(() => isSecureContext && crossOriginIsolated), 'Secure isolated context');
  assert(await page.evaluate(() => document.getElementById('now').textContent === 'Add a speaker'), 'Empty state asks for a speaker, no pre-enrolled identities');
  if (shots) await page.screenshot({ path: path.join(evidence, 'desktop-empty.png'), fullPage: true });

  await page.getByRole('button', { name: 'Load demo clip' }).click();
  await page.waitForFunction(() => window.speakerClips.snapshot().clip?.seconds > 60, undefined, { timeout: 30000 });
  assert(Math.abs((await page.evaluate(() => window.speakerClips.snapshot().clip.seconds)) - script.seconds) < 0.01, 'Demo decodes to its exact length at 16 kHz');

  for (const voice of script.voices) { await page.locator('#name').fill(voice.label); await page.getByRole('button', { name: 'Add speaker' }).click(); }
  const ids = await page.evaluate(() => window.speakerClips.snapshot().speakers.map(s => s.id));
  assert(ids.length === 3, 'Three named speakers added through the form');
  assert(await page.evaluate(() => document.getElementById('now').textContent === 'Add a reference'), 'Speakers without references never produce labels');

  // UI path for Voice A: seek inside its first turn, then "Reference from clip".
  const firstTurn = v => script.turns.find(t => t.voice === v);
  const refEnd = v => firstTurn(v).start + 3.5;
  await page.evaluate(t => new Promise(resolve => { const m = document.querySelector('#player audio'); m.addEventListener('seeked', resolve, { once: true }); m.currentTime = t; }), refEnd(script.voices[0].id));
  await page.getByRole('button', { name: 'Reference for Voice A from the 3 seconds before the playhead' }).click();
  await page.waitForFunction(() => window.speakerClips.snapshot().speakers[0].refs.length === 1, undefined, { timeout: 20000 });
  for (let i = 1; i < 3; i++) await page.evaluate(([id, t]) => window.speakerClips.referenceFromClip(id, t), [ids[i], refEnd(script.voices[i].id)]);
  const refs = await page.evaluate(() => window.speakerClips.snapshot().speakers.map(s => s.refs.map(r => [r.start, r.end])));
  assert(refs.every(r => r.length === 1 && r[0][1] - r[0][0] === 48640), 'Each reference is the 3.04 s before the chosen moment');

  // Playback path: live cells are computed causally while the clip plays.
  await page.evaluate(() => { const m = document.querySelector('#player audio'); m.muted = true; m.currentTime = 30; return m.play(); });
  await page.waitForTimeout(3500);
  await page.evaluate(() => document.querySelector('#player audio').pause());
  const live = await page.evaluate(() => ({ cached: window.speakerClips.snapshot().clipCells.cached, current: window.speakerClips.current() }));
  assert(live.cached >= 3 && live.current.evidence, 'Playback computes live windows near the playhead');
  report.playback = { cachedCells: live.cached, label: live.current.label.text };
  if (shots) await page.screenshot({ path: path.join(evidence, 'desktop-playback.png'), fullPage: true });

  const started = Date.now();
  await page.getByRole('button', { name: 'Analyse whole clip' }).click();
  await page.waitForFunction(() => window.speakerClips.snapshot().analysis.done, undefined, { timeout: 240000 });
  const snap = await page.evaluate(() => window.speakerClips.snapshot());
  assert(snap.clipCells.cached === snap.clipCells.total && snap.clipCells.labeled === snap.clipCells.total, 'Whole clip analysed: every window cached and labelled');
  report.analysis = { cells: snap.clipCells.total, wallSeconds: (Date.now() - started) / 1000, lastComputeMs: snap.lastComputeMs };

  // Score against the turn script: only cells whose painted 0.5 s lies inside one turn and whose
  // whole 1.52 s window lies inside that same turn (no carry-over from the previous speaker).
  const score = async () => {
    const segs = await page.evaluate(() => JSON.parse(window.speakerClips.exportJson()).cells);
    const label = new Map(script.voices.map((v, i) => [ids[i], v.id]));
    const out = { inTurnWindows: 0, correct: 0, wrong: 0, unsure: 0, unknown: 0, quiet: 0, byVoice: {} };
    for (const c of segs) {
      const s = c.start_sample / 16000, e = c.end_sample / 16000;
      const turn = script.turns.find(t => s >= t.start && e <= t.end);
      if (!turn) continue;
      out.inTurnWindows++;
      const v = (out.byVoice[turn.voice] ??= { n: 0, correct: 0, wrong: 0 }); v.n++;
      if (c.status === 'confident') { if (label.get(c.voice_id) === turn.voice) { out.correct++; v.correct++; } else { out.wrong++; v.wrong++; } }
      else if (c.status === 'tentative' || c.status === 'ambiguous') out.unsure++;
      else if (c.status === 'unknown' && c.candidates.length) out.unknown++;
      else out.quiet++;
    }
    return out;
  };
  report.defaults = await score();
  assert(report.defaults.inTurnWindows > 60, 'Enough fully in-turn windows to score');
  assert(report.defaults.wrong <= report.defaults.inTurnWindows * 0.05, 'Wrong-speaker labels at most 5% of in-turn windows at defaults');
  assert(report.defaults.correct >= report.defaults.inTurnWindows * 0.5, 'At least half of in-turn windows correctly named at defaults');
  if (shots) await page.screenshot({ path: path.join(evidence, 'desktop-analysed.png'), fullPage: true });

  // Threshold changes re-run matching only (instant), and stricter settings name fewer windows.
  const t0 = Date.now();
  await page.evaluate(() => window.speakerClips.setSettings({ confident: 0.95, tentative: 0.9 }));
  report.strictRelabelMs = Date.now() - t0;
  report.strict = await score();
  assert(report.strict.correct + report.strict.wrong < report.defaults.correct + report.defaults.wrong, 'Stricter threshold names fewer windows without recomputing embeddings');
  assert((await page.evaluate(() => window.speakerClips.snapshot().clipCells.cached)) === snap.clipCells.total, 'Embedding cache survives threshold changes');
  await page.getByText('Matching settings (exploratory)').click();
  await page.getByRole('button', { name: 'Reset to defaults' }).click();
  assert(JSON.stringify(await score()) === JSON.stringify(report.defaults), 'Reset restores identical labels');

  // Open set: remove Voice C; its turns must never be labelled Voice C, and we report A/B false names.
  const before = await page.evaluate(() => window.speakerClips.segments().filter(s => s.text === 'Voice C').length);
  await page.getByRole('button', { name: 'Remove Voice C' }).click();
  const removed = await score();
  report.voiceCRemoved = { cTurns: removed.byVoice.de02, voiceCSegmentsBefore: before };
  assert(await page.evaluate(() => !window.speakerClips.segments().some(s => s.text === 'Voice C')), 'Removed speaker is never named');

  const json = JSON.parse(await page.evaluate(() => window.speakerClips.exportJson()));
  const vtt = await page.evaluate(() => window.speakerClips.exportVtt());
  assert(json.format === 'voqx-speaker-timeline/1' && json.experimental && json.cells.length === snap.clipCells.total && !JSON.stringify(json).includes('embeddings'), 'JSON export: timeline only, no voiceprints');
  assert(vtt.startsWith('WEBVTT') && vtt.includes('Voice A'), 'WebVTT export');

  // Non-speech must never produce a name: demo + 5 s synthetic chords + 4 s noise bursts + 2 s silence.
  const demoPcm = new Int16Array((await readFile(path.join(DEMO_DIR, 'demo.wav'))).buffer.slice(44));
  const tailStart = demoPcm.length, extra = 11 * 16000, mixed = new Int16Array(tailStart + extra);
  mixed.set(demoPcm);
  let seed = 7; const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;
  for (let i = 0; i < extra; i++) {
    const t = i / 16000; let v = 0;
    if (t < 5) { const chord = [[220, 277.2, 329.6], [196, 246.9, 293.7], [174.6, 220, 261.6]][Math.floor(t / 1.67) % 3]; const env = Math.exp(-3 * (t % 1.67)); for (const f of chord) for (let h = 1; h <= 4; h++) v += env * 0.06 / h * Math.sin(2 * Math.PI * f * h * t); }
    else if (t < 9) { const burst = (t * 4) % 1 < 0.35 ? 1 : 0.15; v = 0.12 * burst * rand(); }
    mixed[tailStart + i] = Math.max(-32767, Math.min(32767, Math.round(v * 32767)));
  }
  const wav = Buffer.alloc(44 + mixed.length * 2);
  wav.write('RIFF', 0); wav.writeUInt32LE(36 + mixed.length * 2, 4); wav.write('WAVEfmt ', 8); wav.writeUInt32LE(16, 16); wav.writeUInt16LE(1, 20); wav.writeUInt16LE(1, 22);
  wav.writeUInt32LE(16000, 24); wav.writeUInt32LE(32000, 28); wav.writeUInt16LE(2, 32); wav.writeUInt16LE(16, 34); wav.write('data', 36); wav.writeUInt32LE(mixed.length * 2, 40);
  Buffer.from(mixed.buffer).copy(wav, 44);
  // What the earlier loudness gate (half the 80 ms frames above -45 dBFS) would have sent to the encoder.
  let levelWouldAnalyse = 0, tailWindows = 0;
  for (let end = 24320; end <= mixed.length; end += 8000) {
    if (end - 24320 < tailStart) continue;
    tailWindows++; let loud = 0;
    for (let f = end - 24320; f + 1280 <= end; f += 1280) { let sum = 0; for (let i = f; i < f + 1280; i++) sum += (mixed[i] / 32768) ** 2; if (Math.sqrt(sum / 1280) >= 10 ** (-45 / 20)) loud++; }
    if (loud / 19 >= 0.5) levelWouldAnalyse++;
  }
  await page.locator('#file').setInputFiles({ name: 'demo-plus-nonspeech.wav', mimeType: 'audio/wav', buffer: wav });
  await page.waitForFunction(n => window.speakerClips.snapshot().clip?.name === 'demo-plus-nonspeech.wav' && window.speakerClips.snapshot().vad.clipDone === window.speakerClips.snapshot().vad.clipFrames, undefined, { timeout: 60000 });
  await page.getByRole('button', { name: 'Analyse whole clip' }).click();
  await page.waitForFunction(() => window.speakerClips.snapshot().analysis.done, undefined, { timeout: 240000 });
  const tail = (await page.evaluate(() => JSON.parse(window.speakerClips.exportJson()).cells)).filter(c => c.start_sample >= tailStart);
  const named = tail.filter(c => ['confident', 'tentative', 'ambiguous'].includes(c.status));
  report.nonSpeech = { tailWindows: tail.length, levelGateWouldAnalyse: levelWouldAnalyse, activity: tail.reduce((m, c) => ({ ...m, [c.activity]: (m[c.activity] ?? 0) + 1 }), {}), namedOrMaybe: named.length };
  assert(tail.length === tailWindows && tail.length >= 15, 'Non-speech tail windows analysed');
  assert(named.length === 0, 'Music, noise and silence never produce a name or a maybe');
  if (shots) await page.screenshot({ path: path.join(evidence, 'desktop-nonspeech.png'), fullPage: true });

  // Microphone paths (fake capture device plays the demo audio). Continuous first.
  await page.getByRole('button', { name: 'Continuous' }).click();
  await page.getByRole('button', { name: 'Listen with microphone' }).click();
  await page.waitForFunction(() => window.speakerClips.snapshot().micCells.labeled >= 3, undefined, { timeout: 30000 });
  report.microphone = { label: (await page.evaluate(() => window.speakerClips.current())).label.text, cells: (await page.evaluate(() => window.speakerClips.snapshot().micCells)) };
  assert(true, 'Microphone listening labels live windows with the same bank');
  if (shots) await page.screenshot({ path: path.join(evidence, 'desktop-microphone.png'), fullPage: true });
  await page.getByRole('button', { name: 'Stop microphone' }).click();
  assert(!(await page.evaluate(() => window.speakerClips.snapshot().mic.active)), 'Stop microphone releases capture');

  // Per-phrase mode: Silero segments live audio; one decision per phrase with measured delay.
  await page.getByRole('button', { name: 'Per phrase' }).click();
  await page.getByRole('button', { name: 'Listen with microphone' }).click();
  await page.waitForFunction(() => window.speakerClips.phrases().filter(p => p.status === 'decided').length >= 4, undefined, { timeout: 45000 });
  const phrases = (await page.evaluate(() => window.speakerClips.phrases())).filter(p => p.status === 'decided');
  report.phrases = phrases.map(p => ({ speech: +p.speechSeconds.toFixed(2), label: p.label.text, top: p.candidates[0]?.nearest?.toFixed(2), latencyMs: Math.round(p.latencyMs), endWaitMs: Math.round(p.endWaitMs), computeMs: Math.round(p.computeMs) }));
  const latencies = phrases.map(p => p.latencyMs).sort((a, b) => a - b);
  report.phraseLatencyMs = { min: Math.round(latencies[0]), median: Math.round(latencies[Math.floor(latencies.length / 2)]), max: Math.round(latencies.at(-1)) };
  assert(phrases.every(p => p.endWaitMs === 256 && p.computeMs > 0 && p.latencyMs >= p.endWaitMs), 'Each phrase reports pause detection, CPU time and total delay');
  assert(report.phraseLatencyMs.median < 1000, 'Median phrase decision under 1 s after speech ends');
  if (shots) await page.screenshot({ path: path.join(evidence, 'desktop-phrases.png'), fullPage: true });
  await page.getByRole('button', { name: 'Stop microphone' }).click();

  // Wake-word enrollment: three phrases from the microphone become references.
  await page.locator('#name').fill('Mic tester'); await page.getByRole('button', { name: 'Add speaker' }).click();
  await page.getByRole('button', { name: 'Record the wake word three times for Mic tester' }).click();
  await page.waitForFunction(() => window.speakerClips.snapshot().speakers.find(s => s.name === 'Mic tester')?.refs.length === 3, undefined, { timeout: 45000 });
  const wakeRefs = await page.evaluate(() => window.speakerClips.snapshot().speakers.find(s => s.name === 'Mic tester').refs);
  report.wakeTakes = wakeRefs.map(r => +r.seconds.toFixed(2));
  assert(wakeRefs.every(r => r.source === 'wake' && r.seconds >= 0.3), 'Record wake word ×3 adds three microphone phrase references');
  await page.waitForFunction(() => !window.speakerClips.snapshot().mic.active, undefined, { timeout: 5000 });
  assert(true, 'Wake enrollment releases the microphone it opened');

  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'Phone layout has no horizontal overflow');
  if (shots) await page.screenshot({ path: path.join(evidence, 'phone.png'), fullPage: true });
  assert(requests.every(r => ['GET', 'HEAD'].includes(r.method)), 'Read-only requests; no audio uploads');
  assert(!errors.length, 'No page errors: ' + errors.join(' | '));
  report.result = 'pass';
} catch (error) {
  report.result = 'fail'; report.failure = error.message;
} finally {
  Object.assign(report, { checks, errors, warnings, requests: [...new Set(requests.map(r => r.method + ' ' + r.path))] });
  await writeFile(path.join(evidence, 'report.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
  await browser.close();
  console.log(JSON.stringify(report, null, 1));
  if (report.result !== 'pass') process.exitCode = 1;
}
