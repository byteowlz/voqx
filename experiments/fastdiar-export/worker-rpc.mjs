// One queued request/response channel to a CPU worker; any timeout or crash fails every pending call.
export class WorkerRpc {
  constructor(url, onFail, timeoutMs = 60000) {
    this.onFail = onFail; this.timeoutMs = timeoutMs; this.pending = new Map(); this.next = 0;
    this.worker = new Worker(url, { type: 'module' });
    this.worker.onmessage = ({ data }) => {
      const request = this.pending.get(data.id);
      if (!request) return;
      this.pending.delete(data.id); clearTimeout(request.timer);
      if (data.error) request.reject(Object.assign(new Error(data.error.message), { code: data.error.code })); else request.resolve(data.result);
    };
    this.worker.onerror = () => this.fail('The CPU worker stopped. Reload the page to restart.');
  }
  call(type, args = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.next;
      const timer = setTimeout(() => this.fail('The CPU worker timed out. Reload the page to restart.'), this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      try { this.worker.postMessage({ id, type, args }, args.pcm ? [args.pcm.buffer] : []); }
      catch (error) { clearTimeout(timer); this.pending.delete(id); reject(error); }
    });
  }
  fail(message) {
    this.worker.terminate();
    for (const request of this.pending.values()) { clearTimeout(request.timer); request.reject(new Error(message)); }
    this.pending.clear(); this.onFail(message);
  }
}
