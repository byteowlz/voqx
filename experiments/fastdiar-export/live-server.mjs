import { parseArgs } from 'node:util';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { isIP } from 'node:net';
import { validateManifest, sha256 } from './protocol.mjs';
import { MODEL_SHA } from './live-core.mjs';

const TYPES = { '.mjs': 'application/javascript', '.js': 'application/javascript', '.css': 'text/css', '.html': 'text/html', '.wasm': 'application/wasm', '.wav': 'audio/wav', '.json': 'application/json' };
const contentType = file => TYPES[path.extname(file)] ?? 'application/octet-stream';

/** Explicit read-only allowlist; optional pages appear only when their inputs are supplied. */
function staticAssets({ root, artifacts, ortDist, captureRoot, tokens, resampler, speakerWasm, vad, demo }) {
  const assets = new Map([
    ['/', path.join(root, 'live.html')], ['/encoder.onnx', path.join(artifacts, 'encoder.onnx')],
    ['/tokens.css', tokens], ['/style.css', path.join(captureRoot, 'style.css')], ['/resampler.js', resampler],
  ]);
  for (const name of ['live.mjs', 'live-controller.mjs', 'live-worker.mjs', 'live-core.mjs', 'live.css', 'step-runtime.mjs', 'frame-pool.mjs', 'protocol.mjs', 'worker-rpc.mjs']) assets.set('/' + name, path.join(root, name));
  // Optional clip page: needs the voqx-recognition WASM package and the pinned Silero VAD directory.
  if (speakerWasm && vad) {
    assets.set('/vad.onnx', path.join(vad, 'silero_vad.onnx'));
    assets.set('/vad-worker.mjs', path.join(root, 'vad-worker.mjs'));
    assets.set('/clips', path.join(root, 'clips.html'));
    for (const name of ['clips.mjs', 'clips-controller.mjs', 'clips-core.mjs', 'clips.css', 'phrase-decider.mjs']) assets.set('/' + name, path.join(root, name));
    for (const name of ['voqx_recognition.js', 'voqx_recognition_bg.wasm']) assets.set('/voqx/' + name, path.join(speakerWasm, name));
  }
  // Optional synthetic demo clip (fictional voices) with its turn script.
  if (demo) { assets.set('/demo.wav', path.join(demo, 'demo.wav')); assets.set('/demo.json', path.join(demo, 'demo.json')); }
  for (const name of ['heads-capture.mjs', 'capture.mjs', 'capture-worklet.js']) assets.set('/' + name, path.join(captureRoot, name));
  for (const name of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) assets.set('/ort/' + name, path.join(ortDist, name));
  return assets;
}

export async function createLiveHandler({ artifacts, ortDist, captureRoot, tokens, resampler, speakerWasm, vad, demo, publicHost = 'localhost' }) {
  const root = path.dirname(fileURLToPath(import.meta.url));
  const manifest = validateManifest(await Bun.file(path.join(artifacts, 'manifest.json')).json());
  if (manifest.onnx_sha256 !== MODEL_SHA || await sha256(await Bun.file(path.join(artifacts, 'encoder.onnx')).arrayBuffer()) !== MODEL_SHA) throw new Error('Incompatible FASTDIAR artifact.');
  if (speakerWasm && !vad) throw new Error('The clip page needs --vad with the pinned Silero model.');
  const assets = staticAssets({ root, artifacts, ortDist, captureRoot, tokens, resampler, speakerWasm, vad, demo });
  for (const file of assets.values()) if (!await Bun.file(file).exists()) throw new Error('Required static asset is missing: ' + path.basename(file));
  const headers = {
    'Cross-Origin-Opener-Policy': 'same-origin', 'Cross-Origin-Embedder-Policy': 'require-corp',
    'Cross-Origin-Resource-Policy': 'same-origin', 'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
    'Permissions-Policy': 'microphone=(self), camera=(), display-capture=()',
    'Content-Security-Policy': "default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'; worker-src 'self'; connect-src 'self'; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
  };
  const metadata = { format: manifest.format, sample_rate: manifest.sample_rate, step_samples: manifest.step_samples,
    embedding_dim: manifest.embedding_dim, state: manifest.state, state_bytes: manifest.state_bytes,
    model_bytes: manifest.model_bytes, onnx_sha256: manifest.onnx_sha256 };
  return request => {
    const url = new URL(request.url);
    if (!['localhost', '127.0.0.1', publicHost].includes(url.hostname)) return new Response('Unknown host', { status: 404, headers });
    if (!['GET', 'HEAD'].includes(request.method)) return new Response('Read-only static service; no audio uploads', { status: 405, headers });
    if (url.pathname === '/favicon.ico') return new Response(null, { status: 204, headers });
    if (url.pathname === '/manifest.json') return Response.json(metadata, { headers });
    if (url.pathname === '/health') return Response.json({ ok: true, service: 'voqx-speaker-test', modelSha256: MODEL_SHA, inference: 'browser-wasm-cpu-only', audioUploads: false, persistence: false, registryWrites: false }, { headers });
    const file = assets.get(url.pathname);
    if (!file) return new Response('Not found', { status: 404, headers });
    return new Response(request.method === 'HEAD' ? null : Bun.file(file), { headers: { ...headers, 'Content-Type': contentType(file) } });
  };
}
if (import.meta.main) {
  const { values } = parseArgs({ options: { artifacts: { type: 'string' }, ort: { type: 'string' }, capture: { type: 'string' }, tokens: { type: 'string' }, resampler: { type: 'string' }, 'speaker-wasm': { type: 'string' }, vad: { type: 'string' }, demo: { type: 'string' }, port: { type: 'string', default: '8067' }, host: { type: 'string', default: '127.0.0.1' }, 'public-host': { type: 'string' }, cert: { type: 'string' }, key: { type: 'string' } } });
  for (const key of ['artifacts', 'ort', 'capture', 'tokens', 'resampler']) if (!values[key]) throw new Error('Required: --' + key);
  const port = Number(values.port);
  if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error('Invalid port.');
  const tls = values.cert && values.key ? { cert: Bun.file(values.cert), key: Bun.file(values.key) } : undefined;
  if (!['localhost', '127.0.0.1'].includes(values.host) && (!tls || !isIP(values.host) || ['0.0.0.0', '::'].includes(values.host))) throw new Error('A specific interface and explicit TLS certificate/key are required for remote access.');
  const handler = await createLiveHandler({ artifacts: values.artifacts, ortDist: values.ort, captureRoot: values.capture, tokens: values.tokens, resampler: values.resampler, speakerWasm: values['speaker-wasm'], vad: values.vad, demo: values.demo, publicHost: values['public-host'] ?? values.host });
  const server = Bun.serve({ hostname: values.host, port, tls, fetch: handler });
  console.log(JSON.stringify({ url: server.url.href, clips: Boolean(values['speaker-wasm']), demo: Boolean(values.demo), service: 'voqx-speaker-test', audioUploads: false, persistence: false }));
}
