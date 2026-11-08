// Experimental tensor contract only; never identity or action authority.
const TYPES = { float32: 4, int64: 8 };
export function validateManifest(m) {
  if (m.format !== 'fastdiar-export-probe/1' || m.step_samples !== 1280 || m.sample_rate !== 16000 || m.embedding_dim !== 192 ||
      !/^[a-f0-9]{64}$/.test(m.onnx_sha256 ?? '') || !Array.isArray(m.state) || m.state.length < 7 || m.state.length > 128) throw new Error('Unsupported FASTDIAR export manifest');
  const names = new Set(), fields = new Set();
  let bytes = 0;
  for (const slot of m.state) {
    if (!/^state_\d+$/.test(slot.input) || slot.output !== `next_${slot.input}` || names.has(slot.input) || fields.has(slot.field) ||
        !TYPES[slot.dtype] || !Array.isArray(slot.shape) || slot.shape.length > 6 || slot.shape.some(n => !Number.isSafeInteger(n) || n < 1)) throw new Error('Invalid or duplicate state slot');
    if (['audio_count', 'cms_count'].includes(slot.field) && (slot.dtype !== 'int64' || slot.shape.length !== 0)) throw new Error('Counters must be int64 scalars');
    const elements = slot.shape.reduce((a, b) => a * b, 1);
    if (elements > 1_000_000 || slot.bytes !== elements * TYPES[slot.dtype]) throw new Error('State size mismatch');
    names.add(slot.input); fields.add(slot.field); bytes += slot.bytes;
  }
  if (!fields.has('audio_count') || !fields.has('cms_count') || bytes !== m.state_bytes || bytes > 8 * 1024 ** 2) throw new Error('Incomplete or oversized state');
  return m;
}
export function zeroData(slot) {
  const n = slot.shape.reduce((a, b) => a * b, 1);
  if (slot.dtype === 'float32') return new Float32Array(n);
  if (slot.dtype === 'int64') return new BigInt64Array(n);
  throw new Error('Unsupported tensor type');
}
export async function sha256(data) {
  const digest = await crypto.subtle.digest('SHA-256', data);
  return [...new Uint8Array(digest)].map(n => n.toString(16).padStart(2, '0')).join('');
}
