// Non-visual numerical test fixture; no microphone access or product UI.
const worker = new Worker('/worker.mjs', { type: 'module' });
const waiting = new Map();
let next = 0;
worker.onmessage = ({ data }) => {
  const request = waiting.get(data.id); waiting.delete(data.id);
  if (data.error) request.reject(new Error(data.error)); else request.resolve(data.result);
};
worker.onerror = error => {
  for (const request of waiting.values()) request.reject(new Error(error.message));
  waiting.clear();
};
window.probe = (type, args) => new Promise((resolve, reject) => {
  const id = ++next; waiting.set(id, { resolve, reject }); worker.postMessage({ id, type, args });
});
window.probeReady = true;
