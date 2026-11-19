// Deterministic double-precision pooling over FP32 encoder output; no identity decision.
export function poolFrames(frames) {
  if (!frames.length || frames.some(f => f.length !== 192 || !f.every(Number.isFinite))) throw new Error('Invalid frame vectors');
  const mean = new Array(192).fill(0);
  for (const frame of frames) for (let d = 0; d < 192; d++) mean[d] += frame[d] / frames.length;
  const unit = values => {
    const norm = Math.hypot(...values);
    return norm > 1e-12 ? Array.from(values, value => value / norm) : null;
  };
  return { mean: unit(mean), last: unit(frames.at(-1)) };
}
export function fromBase64(value) {
  if (typeof value !== 'string' || value.length > 1_000_000) throw new Error('Oversized IPC payload');
  const raw = atob(value);
  return Uint8Array.from(raw, c => c.charCodeAt(0));
}
export function toBase64(bytes) {
  const parts = [];
  for (let i = 0; i < bytes.length; i += 8192) parts.push(String.fromCharCode(...bytes.subarray(i, i + 8192)));
  return btoa(parts.join(''));
}
