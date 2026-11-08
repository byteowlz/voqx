"""Export an experimental CPU FASTDIAR step with every streaming cache explicit.

Weights and synthetic parity fixtures stay outside git. This is not a shipped model,
recognition profile, license approval or speaker filtering qualification.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'short-window'))
from audio import sha256
from fastdiar_probe import load_model

ATTRS = ('audio_buffer', 'preemph_last', 'audio_sum', 'audio_sumsq', 'audio_count', 'cms_sum', 'cms_count')
COUNTERS = {'audio_count', 'cms_count'}


class ExplicitStep(torch.nn.Module):
    def __init__(self, model, samples):
        super().__init__()
        from fastdiar.encoder import StreamingInference, StreamingState
        self.model = model
        self.stream = StreamingInference(model)
        self.state_type = StreamingState
        with torch.no_grad():
            self.stream.process_chunk(torch.zeros(samples))
        self.keys = sorted(self.stream.state.conv_buffers)
        self.fields = list(ATTRS) + self.keys
        self.initial = []
        self.slots = []
        for index, field in enumerate(self.fields):
            if field in COUNTERS:
                tensor = torch.tensor(0, dtype=torch.int64)
            else:
                value = getattr(self.stream.state, field) if field in ATTRS else self.stream.state.conv_buffers[field]
                tensor = torch.zeros_like(value).contiguous()
            self.initial.append(tensor)
            self.slots.append({'input': f'state_{index}', 'output': f'next_state_{index}', 'field': field,
                               'shape': list(tensor.shape), 'dtype': 'int64' if tensor.dtype == torch.int64 else 'float32',
                               'bytes': tensor.numel() * tensor.element_size()})

    def forward(self, audio, *slots):
        state = self.state_type()
        for index, field in enumerate(ATTRS):
            # Upstream += mutates Tensor counters; the exported operation is a pure state transition.
            setattr(state, field, slots[index].clone() if field in COUNTERS else slots[index])
        state.conv_buffers = dict(zip(self.keys, slots[len(ATTRS):]))
        self.stream.state = state
        embedding = self.stream.process_chunk(audio)
        values = [getattr(state, field) for field in ATTRS] + [state.conv_buffers[key] for key in self.keys]
        return (embedding, *values)


def fixture(frames, samples):
    rng = np.random.default_rng(17)
    t = np.arange(frames * samples, dtype=np.float32)
    signal = (0.12 * np.sin(t * 0.071) + 0.03 * np.sin(t * 0.013)).astype(np.float32)
    quarter = len(signal) // 4
    signal[quarter:2 * quarter] = 0
    signal[2 * quarter:3 * quarter] = rng.normal(0, 0.08, quarter).astype(np.float32)
    signal[3 * quarter:] += 0.08
    return signal


def native_check(step, model, audio, samples, file):
    from fastdiar.encoder import StreamingInference
    options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
    ort.disable_telemetry_events()
    started = time.perf_counter()
    session = ort.InferenceSession(str(file), sess_options=options, providers=['CPUExecutionProvider'])
    load_ms = (time.perf_counter() - started) * 1000
    expected_inputs = ['audio'] + [s['input'] for s in step.slots]
    if [i.name for i in session.get_inputs()] != expected_inputs:
        raise ValueError('Export discarded or reordered state inputs')
    control = StreamingInference(model)
    states = [v.numpy().copy() for v in step.initial]
    golden, actual, timings = [], [], []
    for offset in range(0, len(audio), samples):
        chunk = audio[offset:offset + samples]
        with torch.no_grad():
            reference = control.process_chunk(torch.from_numpy(chunk.copy())).numpy().copy()
        feeds = {'audio': chunk, **dict(zip(expected_inputs[1:], states))}
        started = time.perf_counter(); output = session.run(None, feeds); timings.append((time.perf_counter() - started) * 1000)
        states = output[1:]
        for descriptor, value in zip(step.slots, states):
            if descriptor['field'] == 'audio_count' and int(value) != offset + samples:
                raise ValueError('Export froze its audio counter')
            if descriptor['field'] == 'cms_count' and int(value) != (offset + samples) // 160:
                raise ValueError('Export froze its spectral counter')
        golden.append(reference.reshape(-1)); actual.append(output[0].reshape(-1))
    golden, actual = np.asarray(golden), np.asarray(actual)
    delta = np.abs(golden - actual)
    if delta.max() > 1e-4 or delta.mean() > 1e-5:
        raise ValueError(f'Native ONNX parity failed: max={delta.max()} mean={delta.mean()}')
    return golden, {'ort_version': ort.__version__, 'threads': 1, 'load_ms': load_ms,
                    'embedding_max_abs': float(delta.max()), 'embedding_mean_abs': float(delta.mean()),
                    'step_ms_p50': float(np.median(timings)), 'step_ms_p95': float(np.percentile(timings, 95))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'checkpoint', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    _, model, _, provenance = load_model(args.source, args.checkpoint)
    samples, frames = 1280, 240
    step = ExplicitStep(model, samples).eval()
    file = args.out / 'encoder.onnx'
    with torch.no_grad():
        torch.onnx.export(step, (torch.zeros(samples), *step.initial), str(file), dynamo=False,
                          export_params=True, opset_version=17, do_constant_folding=True,
                          input_names=['audio'] + [s['input'] for s in step.slots],
                          output_names=['embedding'] + [s['output'] for s in step.slots], external_data=False)
    graph = onnx.load(file)
    onnx.checker.check_model(graph)
    if any(t.external_data for t in graph.graph.initializer):
        raise ValueError('Expected a self-contained research artifact')
    audio = fixture(frames, samples)
    golden, native = native_check(step, model, audio, samples, file)
    audio.astype('<f4').tofile(args.out / 'input.f32'); golden.astype('<f4').tofile(args.out / 'expected.f32')
    manifest = {'format': 'fastdiar-export-probe/1', 'sample_rate': 16000, 'step_samples': samples,
                'embedding_dim': 192, 'fixture_frames': frames, 'state': step.slots,
                'state_bytes': sum(s['bytes'] for s in step.slots), 'provenance': provenance,
                'native': native, 'model_bytes': file.stat().st_size, 'onnx_sha256': sha256(file),
                'input_sha256': sha256(args.out / 'input.f32'), 'expected_sha256': sha256(args.out / 'expected.f32'),
                'exporter_sha256': sha256(__file__), 'onnx_version': onnx.__version__, 'onnx_ir_version': graph.ir_version,
                'opsets': {op.domain: op.version for op in graph.opset_import},
                'limits': ['Synthetic numerical fixtures only, not speaker accuracy', 'Experimental local export; redistribution not approved',
                           'No VAD, diarizer, identity enrollment, admission policy or production default changes']}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'artifact': str(args.out), 'model_bytes': manifest['model_bytes'], 'state_slots': len(step.slots),
                      'state_bytes': manifest['state_bytes'], 'native': native}))


if __name__ == '__main__':
    main()
