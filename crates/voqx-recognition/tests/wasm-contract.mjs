// Actual Chromium execution of the portable Rust/WASM matcher, not encoder inference.
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';

const { WASM_PKG, NATIVE_FIXTURE, CHROMIUM, PLAYWRIGHT, EVIDENCE_DIR } = process.env;
if (![WASM_PKG, NATIVE_FIXTURE, CHROMIUM, PLAYWRIGHT, EVIDENCE_DIR].every(Boolean)) {
  throw new Error('Set WASM_PKG, NATIVE_FIXTURE, CHROMIUM, PLAYWRIGHT and a NEW EVIDENCE_DIR');
}
const fixture = JSON.parse(await readFile(NATIVE_FIXTURE, 'utf8'));
assert.equal(fixture.format, 'voqx-speaker-contract/1');
await mkdir(EVIDENCE_DIR, { recursive: false });
const files = new Map([
  ['/sdk.js', [resolve(WASM_PKG, 'voqx_recognition.js'), 'text/javascript']],
  ['/voqx_recognition_bg.wasm', [resolve(WASM_PKG, 'voqx_recognition_bg.wasm'), 'application/wasm']],
]);
const requests = [];
const server = Bun.serve({ hostname: '127.0.0.1', port: 0, async fetch(request) {
  requests.push({ method: request.method, path: new URL(request.url).pathname });
  if (request.method !== 'GET') return new Response('read only', { status: 405 });
  const path = new URL(request.url).pathname;
  if (path === '/') return new Response('<!doctype html><title>WASM contract check</title>', { headers: { 'Content-Type': 'text/html' } });
  if (path === '/favicon.ico') return new Response(null, { status: 204 });
  const file = files.get(path);
  return file ? new Response(await readFile(file[0]), { headers: { 'Content-Type': file[1] } }) : new Response('not found', { status: 404 });
}});
let browser;
try {
  const { chromium } = await import(PLAYWRIGHT);
  browser = await chromium.launch({ executablePath: CHROMIUM, headless: true });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(server.url.href);
  const actual = await page.evaluate(async fixture => {
    const { default: init, SpeakerAttributor } = await import('/sdk.js');
    await init();
    const cases = [];
    for (const config of fixture.cases) {
      const scorer = new SpeakerAttributor(JSON.stringify(config.snapshot),
        config.calibration ? JSON.stringify(config.calibration) : undefined,
        BigInt(config.generation), BigInt(config.stable_samples));
      const revision = scorer.referenceRevision();
      const results = [];
      for (const action of config.actions) {
        try {
          if (action.operation === 'reset') {
            scorer.reset(BigInt(action.generation));
            results.push({ ok: { reset: true } });
          } else {
            const { operation, values, ...observation } = action;
            results.push({ ok: JSON.parse(scorer.observe(new Float32Array(values), JSON.stringify(observation))) });
          }
        } catch (error) { results.push({ error: error.message }); }
      }
      const oneShot = config.actions.filter(a => a.operation === 'observe').map(action => {
        const { operation, values, ...observation } = action;
        try { return { ok: JSON.parse(scorer.evaluate(new Float32Array(values), JSON.stringify(observation))) }; }
        catch (error) { return { error: error.message }; }
      });
      scorer.free();
      cases.push({ revision, results, oneShot });
    }
    const base = fixture.cases[1];
    const invalid = [
      { snapshot: { ...base.snapshot, space: { ...base.snapshot.space, dimensions: 0 } }, calibration: undefined, stable: 4n },
      { snapshot: { ...base.snapshot, voices: [base.snapshot.voices[0], base.snapshot.voices[0]] }, calibration: undefined, stable: 4n },
      { snapshot: base.snapshot, calibration: { ...base.calibration, reference_revision: '0'.repeat(64) }, stable: 4n },
      { snapshot: base.snapshot, calibration: base.calibration, stable: 0n },
    ];
    const constructorErrors = invalid.map(config => {
      try {
        const value = new SpeakerAttributor(JSON.stringify(config.snapshot),
          config.calibration ? JSON.stringify(config.calibration) : undefined, 1n, config.stable);
        value.free();
        return null;
      } catch (error) { return error.message; }
    });
    return { cases, constructorErrors, userAgent: navigator.userAgent };
  }, fixture);
  for (let i = 0; i < fixture.cases.length; i++) {
    assert.equal(actual.cases[i].revision, fixture.cases[i].revision, 'native/WASM exact bank fingerprint');
    assert.deepEqual(actual.cases[i].results, fixture.cases[i].expected, 'native/WASM full evidence and stale-error parity');
    assert.deepEqual(actual.cases[i].oneShot, fixture.cases[i].one_shot, 'native/WASM one-shot (untracked) evidence parity');
  }
  assert.equal(actual.constructorErrors.length, 4);
  assert(actual.constructorErrors.every(message => /invalid (recognition profile|speaker policy)/.test(message)));
  assert.deepEqual(errors, []);
  assert(requests.every(request => request.method === 'GET'));
  const report = { format: 'voqx-speaker-wasm-contract/1', scope: 'portable matcher/tracker only; not encoder or speaker accuracy',
    cases: fixture.cases.length, actions: fixture.cases.reduce((sum, config) => sum + config.actions.length, 0),
    exactNativeEvidenceParity: true, oneShotActions: fixture.cases.reduce((n, c) => n + c.one_shot.length, 0), constructorRejections: actual.constructorErrors.length, userAgent: actual.userAgent, requests };
  await writeFile(resolve(EVIDENCE_DIR, 'report.json'), JSON.stringify(report, null, 2), { flag: 'wx' });
  console.log(JSON.stringify(report));
} finally {
  await browser?.close();
  await server.stop(true);
}
