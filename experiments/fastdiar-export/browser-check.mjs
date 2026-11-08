import { parseArgs } from 'node:util';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { sha256, validateManifest } from './protocol.mjs';
import { consoleSeverity } from './console-severity.mjs';
import { runCohort } from './cohort-runner.mjs';

const { values } = parseArgs({ options: { artifacts: { type: 'string' }, ort: { type: 'string' }, evidence: { type: 'string' }, cohort: { type: 'boolean' }, source: { type: 'string' }, registered: { type: 'string' }, 'fastdiar-source': { type: 'string' }, checkpoint: { type: 'string' } } });
if (!values.artifacts || !values.ort || !values.evidence || !process.env.PLAYWRIGHT || !process.env.CHROMIUM) throw new Error('Provide --artifacts --ort --evidence and PLAYWRIGHT/CHROMIUM executable paths');
if (values.cohort && ![values.source, values.registered, values['fastdiar-source'], values.checkpoint].every(Boolean)) throw new Error('Cohort replay requires all four private input paths');
if (values.cohort) { process.env.DEBUG = ''; process.env.PWDEBUG = '0'; }
const root = path.dirname(fileURLToPath(import.meta.url));
const artifacts = path.resolve(values.artifacts), evidence = path.resolve(values.evidence);
await mkdir(evidence, { recursive: false, mode: 0o700 });
const manifest = validateManifest(await Bun.file(path.join(artifacts, 'manifest.json')).json());
const expectedBytes = await Bun.file(path.join(artifacts, 'expected.f32')).arrayBuffer();
if (await sha256(expectedBytes) !== manifest.expected_sha256) throw new Error('Golden hash mismatch');
const expected = new Float32Array(expectedBytes);
const assets = new Map();
for (const name of ['worker.mjs', 'page.mjs', 'protocol.mjs', 'step-runtime.mjs', 'frame-pool.mjs']) assets.set('/' + name, path.join(root, name));
for (const name of ['encoder.onnx', 'manifest.json', 'input.f32']) assets.set('/' + name, path.join(artifacts, name));
const ortNames = ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'];
for (const name of ortNames) assets.set('/ort/' + name, path.join(path.resolve(values.ort), name));
// Read-only factory observation: kernels, options and returned module are unchanged.
const memoryShim = `import factory from './original-ort-wasm-simd-threaded.mjs';
export default async function(options) {
  const module = await factory(options);
  globalThis.__ortLinearMemoryBytes = () => module.HEAPU8?.byteLength ?? null;
  return module;
}`;
assets.set('/ort/original-ort-wasm-simd-threaded.mjs', path.join(path.resolve(values.ort), 'ort-wasm-simd-threaded.mjs'));
const headers = {
  'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp', 'Cross-Origin-Resource-Policy': 'same-origin',
  'Content-Security-Policy': "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; worker-src 'self'; connect-src 'self'; frame-ancestors 'none'",
  'Cache-Control': 'no-store',
};
const server = Bun.serve({ hostname: '127.0.0.1', port: 0, fetch(request) {
  if (!['GET', 'HEAD'].includes(request.method)) return new Response('Method not allowed', { status: 405, headers });
  const url = new URL(request.url);
  if (url.pathname === '/') return new Response('<!doctype html><script type="module" src="/page.mjs"></script>', { headers: { ...headers, 'Content-Type': 'text/html' } });
  if (url.pathname === '/ort/ort-wasm-simd-threaded.mjs') return new Response(memoryShim, { headers: { ...headers, 'Content-Type': 'application/javascript' } });
  if (url.pathname === '/favicon.ico') return new Response(null, { status: 204, headers });
  const file = assets.get(url.pathname);
  if (!file) return new Response('Not found', { status: 404, headers });
  const type = file.endsWith('.mjs') ? 'application/javascript' : file.endsWith('.wasm') ? 'application/wasm' : file.endsWith('.json') ? 'application/json' : 'application/octet-stream';
  return new Response(Bun.file(file), { headers: { ...headers, 'Content-Type': type } });
} });
const origin = server.url.origin;
const { chromium } = await import(process.env.PLAYWRIGHT);
let browser, cdp;
const errors = [], warnings = [], checkpoints = [], workerMemory = [];
const report = { manifest: { onnxSha256: manifest.onnx_sha256, modelBytes: manifest.model_bytes, stateBytes: manifest.state_bytes, stateSlots: manifest.state.length }, checks: [] };
const assert = (condition, message) => { if (!condition) throw new Error(message); report.checks.push(message); };
const quantile = (values, q) => [...values].sort((a, b) => a - b)[Math.min(values.length - 1, Math.floor(values.length * q))];
async function workerHeap(label) {
  const { targetInfos } = await cdp.send('Target.getTargets');
  const target = targetInfos.find(t => t.type === 'worker' && t.url === origin + '/worker.mjs');
  if (!target) { workerMemory.push({ label, available: false }); return; }
  const { sessionId } = await cdp.send('Target.attachToTarget', { targetId: target.targetId, flatten: false });
  let id = 0;
  const send = method => new Promise((resolve, reject) => {
    const requestId = ++id;
    const listener = event => {
      if (event.sessionId !== sessionId) return;
      const message = JSON.parse(event.message);
      if (message.id !== requestId) return;
      cdp.off('Target.receivedMessageFromTarget', listener); clearTimeout(timer);
      if (message.error) reject(new Error(message.error.message)); else resolve(message.result);
    };
    const timer = setTimeout(() => { cdp.off('Target.receivedMessageFromTarget', listener); reject(new Error('Worker heap request timed out')); }, 10000);
    cdp.on('Target.receivedMessageFromTarget', listener);
    cdp.send('Target.sendMessageToTarget', { sessionId, message: JSON.stringify({ id: requestId, method }) }).catch(reject);
  });
  try {
    await send('HeapProfiler.collectGarbage');
    workerMemory.push({ label, available: true, afterExplicitGC: true, ...await send('Runtime.getHeapUsage') });
  } finally { await cdp.send('Target.detachFromTarget', { sessionId }); }
}
async function snapshot(label) {
  const { processInfo } = await cdp.send('SystemInfo.getProcessInfo');
  const ids = processInfo.filter(p => p.type === 'renderer').map(p => p.id);
  const command = ids.length ? Bun.spawnSync(['ps', '-o', 'pid=,rss=', '-p', ids.join(',')]) : null;
  const rss = command ? command.stdout.toString().trim().split('\n').filter(Boolean).map(line => {
    const [pid, kib] = line.trim().split(/\s+/).map(Number); return { pid, rssBytes: kib * 1024 };
  }) : [];
  checkpoints.push({ label, rendererProcesses: rss, rendererRssBytes: rss.reduce((n, r) => n + r.rssBytes, 0) });
}
function parity(result, label) {
  assert(result.embeddings.length === expected.length, label + ': exact frame count');
  let maximum = 0, sum = 0;
  for (let i = 0; i < expected.length; i++) {
    const delta = Math.abs(result.embeddings[i] - expected[i]);
    if (!Number.isFinite(delta)) throw new Error('Nonfinite browser output');
    maximum = Math.max(maximum, delta); sum += delta;
  }
  const mean = sum / expected.length;
  assert(maximum <= 1e-4 && mean <= 1e-5, label + ': Torch embedding parity');
  assert(result.processedSamples === manifest.fixture_frames * manifest.step_samples && result.pendingSamples === 0, label + ': exact coverage');
  return { maxAbs: maximum, meanAbs: mean, frames: result.embeddings.length / 192, stepMsP50: quantile(result.timings.slice(10), 0.5), stepMsP95: quantile(result.timings.slice(10), 0.95), firstStepMs: result.timings[0], elapsedMs: result.elapsedMs };
}
try {
  browser = await chromium.launch({ executablePath: process.env.CHROMIUM, headless: true, args: ['--disable-gpu', '--disable-background-networking', '--enable-precise-memory-info'] });
  cdp = await browser.newBrowserCDPSession();
  const context = await browser.newContext({ serviceWorkers: 'block' });
  await context.route('**/*', route => {
    if (new URL(route.request().url()).origin !== origin) { errors.push('External request blocked'); return route.abort(); }
    return route.continue();
  });
  const page = await context.newPage();
  page.on('pageerror', error => errors.push(String(error)));
  page.on('console', event => {
    if (event.type() === 'error') (consoleSeverity(event.text()) === 'warning' ? warnings : errors).push(event.text());
    else if (event.type() === 'warning') warnings.push(event.text());
  });
  await page.goto(origin); await page.waitForFunction(() => window.probeReady);
  const call = async (type, args) => {
    let timer;
    try { return await Promise.race([
      page.evaluate(async ({ type, args }) => { const result = await window.probe(type, args); return { ...result, embeddings: result.embeddings ? Array.from(result.embeddings) : undefined }; }, { type, args }),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(type + ' timed out')), 90000); timer.unref(); }),
    ]); } finally { clearTimeout(timer); }
  };
  report.browser = { version: browser.version(), ...await page.evaluate(() => ({ userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency, isolated: crossOriginIsolated })) };
  assert(report.browser.isolated, 'Cross-origin isolated worker fixture');
  report.linearMemory = [{ label: 'before-model', ...await call('memory') }];
  await workerHeap('before-model'); await snapshot('before-model');
  const started = performance.now(); report.boot = await call('boot'); report.boot.wallMs = performance.now() - started;
  report.linearMemory.push({ label: 'model-loaded', ...await call('memory') });
  await workerHeap('model-loaded'); await snapshot('model-loaded');
  report.microchunks = parity(await call('run', { chunkSize: 997 }), '997-sample fragments');
  const partial = await call('run', { samples: 1279, chunkSize: 1279 });
  assert(partial.embeddings.length === 0 && partial.pendingSamples === 1279 && partial.processedSamples === 0, 'Partial frame held, not padded or dropped');
  report.resetJumbo = parity(await call('run', { chunkSize: 19001 }), 'Reset then jumbo fragments');
  assert((await call('invalid')).rejected, 'Nonfinite PCM rejected');
  assert((await call('race')).resetRejected, 'Reset fenced during inference');
  const bench = await call('bench');
  report.query = { samples: bench.querySamples, seconds: bench.querySamples / 16000, repetitions: bench.queryMs.length, computeMsP50: quantile(bench.queryMs, 0.5), computeMsP95: quantile(bench.queryMs, 0.95) };
  report.linearMemory.push({ label: 'after-two-streams-and-query-benchmark', ...await call('memory') });
  await workerHeap('after-two-streams-and-query-benchmark'); await snapshot('after-two-streams-and-query-benchmark');
  await call('bench');
  report.linearMemory.push({ label: 'after-second-query-burst', ...await call('memory') });
  await workerHeap('after-second-query-burst'); await snapshot('after-second-query-burst');
  const warmCapacity = report.linearMemory.find(m => m.label === 'after-two-streams-and-query-benchmark').wasmLinearMemoryBytes;
  assert(Number.isSafeInteger(warmCapacity) && warmCapacity > 0, 'Actual WASM linear-memory capacity observed');
  assert(report.linearMemory.find(m => m.label === 'after-second-query-burst').wasmLinearMemoryBytes === warmCapacity, 'No linear-heap growth between measured query bursts');
  if (values.cohort) {
    report.cohort = await runCohort(call, values, root, evidence);
    report.linearMemory.push({ label: 'after-cohort', ...await call('memory') });
    await workerHeap('after-cohort'); await snapshot('after-cohort');
  }
  await call('close');
  report.linearMemory.push({ label: 'session-released', ...await call('memory') }); await workerHeap('session-released'); await snapshot('session-released');
  assert(errors.length === 0, 'No browser errors or external inference requests');
  const packageInfo = await Bun.file(path.resolve(values.ort, '../package.json')).json();
  report.host = { platform: process.platform, arch: process.arch, bun: Bun.version };
  report.ort = { version: packageInfo.version, executionProviders: ['wasm'], threads: 1, assetHashes: {}, memoryInstrumentation: { kind: 'read-only Emscripten factory return observer', shimSha256: await sha256(new TextEncoder().encode(memoryShim)) } };
  for (const name of ortNames) report.ort.assetHashes[name] = await sha256(await Bun.file(path.join(values.ort, name)).arrayBuffer());
  report.rssSnapshots = checkpoints; report.workerHeapSnapshots = workerMemory; report.consoleWarnings = warnings;
  report.harnessHashes = {};
  for (const name of ['browser-check.mjs', 'worker.mjs', 'step-runtime.mjs', 'protocol.mjs', 'console-severity.mjs', 'page.mjs', 'frame-pool.mjs', 'cohort-runner.mjs']) report.harnessHashes[name] = await sha256(await Bun.file(path.join(root, name)).arrayBuffer());
  report.limits = ['Base parity fixtures are synthetic; optional registered real-owner/synthetic-non-target cohort results are reported separately', 'One desktop Chromium on the producing host; no phone/Safari/Firefox qualification', 'Renderer RSS includes browser/worker/runtime overhead and shared pages; not model-only or peak RAM. Worker heap/backing storage excludes some native/runtime overhead', 'Compute excludes audio acquisition; no microphone, enrollment, filtering or default changes'];
  await writeFile(path.join(evidence, 'report.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify({ evidence, checks: report.checks.length, microchunks: report.microchunks, resetJumbo: report.resetJumbo, query: report.query, workerHeapSnapshots: workerMemory, linearMemory: report.linearMemory, rssSnapshots: checkpoints }, null, 2));
} catch (error) {
  await writeFile(path.join(evidence, 'failure.json'), JSON.stringify({ error: String(error.stack ?? error), errors, warnings, report, checkpoints }, null, 2));
  throw error;
} finally { if (browser) await browser.close(); server.stop(true); }
