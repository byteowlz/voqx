import { test, expect } from 'bun:test';
import { consoleSeverity } from './console-severity.mjs';
test('known native WASM CPU warning retains warning severity despite stderr routing', () => {
  expect(consoleSeverity('\u001b[0;93m2026-11-06 13:55:02.648495 [W:onnxruntime:Default, cpuid_info.cc:91 LogEarlyWarning] Unknown CPU vendor. cpuinfo_vendor value: 0\u001b[m')).toBe('warning');
});
test('native errors, JS exceptions and other warnings are not hidden', () => {
  for (const message of ['TypeError: failed', '[E:onnxruntime] unsupported operator', '[W:onnxruntime] different warning', 'Unknown CPU vendor'])
    expect(consoleSeverity(message)).toBe('error');
});
