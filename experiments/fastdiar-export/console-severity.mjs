// Emscripten routes native stderr (including warnings) through console.error.
export function consoleSeverity(message) {
  const plain = message.replace(/\u001b\[[0-9;]*m/g, '').trim();
  const vendorWarning = /^\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+ \[W:onnxruntime:Default, cpuid_info\.cc:91 LogEarlyWarning\] Unknown CPU vendor\. cpuinfo_vendor value: 0$/;
  return vendorWarning.test(plain) ? 'warning' : 'error';
}
