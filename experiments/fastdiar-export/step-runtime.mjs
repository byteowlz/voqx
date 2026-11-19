import * as ort from '/ort/ort.wasm.min.mjs';
import { StepRuntime as CpuRuntime } from '../../web/step-runtime.mjs';
ort.env.wasm.wasmPaths = '/ort/';

// Sandbox compatibility wrapper; reusable runtime takes caller-owned bytes/ORT.
export class StepRuntime {
  static async create(manifest) {
    const response = await fetch('/encoder.onnx');
    if (!response.ok) throw Error(`Model fetch failed: ${response.status}`);
    return CpuRuntime.create(ort, manifest, new Uint8Array(await response.arrayBuffer()));
  }
}
