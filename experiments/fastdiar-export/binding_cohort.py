"""Private, transient three-binding cohort replay over immutable registered audio.

IPC only: never write PCM or vectors, and never expose audio over HTTP. Report the
prespecified diagnostic grid; do not install profiles or select settings from test.
"""
import argparse
import base64
from itertools import product
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'short-window'))
from audio import crop, sha256
from bank_probe import score_rows, verify_inputs
from fastdiar_probe import SOURCE_MANIFEST, VECTOR_MANIFEST, checked_audio, compare, load_model
from metrics import check_split

MODEL_SHA = 'e7e3878d00afb742094eb22a8edc3a7f756d8ae1a10ec16200d95e56cf6ec3b1'
# Frozen before replay; a conservative numerical diagnostic, NOT a qualified FAR gate.
SCORE_MARGIN = 0.001
BACKENDS = ('torch80', 'native_ort', 'browser_wasm')


def pool_frames(frames):
    values = np.asarray(frames, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 192 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected finite frame vectors')
    result = {}
    for mode, value in [('mean', values.mean(axis=0)), ('last', values[-1])]:
        norm = float(np.linalg.norm(value))
        result[mode] = value / norm if norm > 1e-12 else None
    return result


def encode_frames(value):
    return base64.b64encode(np.asarray(value, dtype='<f4').tobytes()).decode('ascii')


def decode_frames(value, shape):
    raw = base64.b64decode(value, validate=True)
    if len(raw) != int(np.prod(shape)) * 4:
        raise ValueError('Frame byte count mismatch')
    result = np.frombuffer(raw, dtype='<f4').reshape(shape)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite binding output')
    return result


def count_at(rows, threshold):
    return {role: {
        'positive_n': sum(r['target'] for r in rows if r['role'] == role),
        'negative_n': sum(not r['target'] for r in rows if r['role'] == role),
        'positive_accept': sum(r['target'] and r['score'] is not None and r['score'] >= threshold for r in rows if r['role'] == role),
        'negative_accept': sum(not r['target'] and r['score'] is not None and r['score'] >= threshold for r in rows if r['role'] == role),
        'abstain': sum(r['score'] is None for r in rows if r['role'] == role),
    } for role in ('calibration', 'test')}


def disagreements(control, rows, threshold):
    if [r['id'] for r in control] != [r['id'] for r in rows]:
        raise ValueError('Cannot compare different query orders')
    def accepts(row):
        return row['score'] is not None and row['score'] >= threshold
    return {role: sum(accepts(a) != accepts(b) for a, b in zip(control, rows) if a['role'] == role)
            for role in ('calibration', 'test')}


def compare_bindings(scenario, lookup, banks):
    independent = {name: compare(scenario, lookup, *banks[name]) for name in BACKENDS}
    fixed = []
    for config in independent['torch80']:
        seconds, alignment, pooling = config['seconds'], config['alignment'], config['pooling']
        threshold = config['threshold_calibration_zero_fa']
        if threshold is None:
            raise ValueError('Calibration cannot produce an operating point')
        scored = {}
        for backend in BACKENDS:
            refs, queries = banks[backend]
            vectors = [refs[(i, alignment, pooling)] for i in config['selected_references']]
            rows = [{'id': i, 'target': lookup[i]['speaker'] == 'owner', 'role': lookup[i]['role'],
                     'vector': queries[(i, seconds, alignment, pooling)]} for i in scenario['queries']]
            scored[backend] = score_rows(rows, vectors, config['method'])
        control = scored['torch80']
        result = {k: config[k] for k in ('seconds', 'alignment', 'pooling', 'bank_size', 'method', 'selected_references')}
        result.update(torch_threshold=threshold, guarded_threshold=threshold + SCORE_MARGIN, bindings={})
        for backend, rows in scored.items():
            deltas = [abs(a['score'] - b['score']) for a, b in zip(control, rows) if a['score'] is not None and b['score'] is not None]
            result['bindings'][backend] = {
                'raw': count_at(rows, threshold), 'guarded': count_at(rows, threshold + SCORE_MARGIN),
                'raw_disagreements': disagreements(control, rows, threshold),
                'guarded_disagreements': disagreements(control, rows, threshold + SCORE_MARGIN),
                'max_abs_score_delta': max(deltas, default=0),
                'abstention_mismatch': sum((a['score'] is None) != (b['score'] is None) for a, b in zip(control, rows)),
            }
        fixed.append(result)
    return {'independent_calibration': independent, 'fixed_torch_bank_and_threshold': fixed}


def jobs(source, config, scenario):
    ids = sorted(scenario['enrollment']['owner'])
    lookup = {r['id']: r for r in config['records']}
    for identifier in ids + scenario['queries']:
        audio = checked_audio(source, lookup[identifier])
        durations = (10.0,) if identifier in ids else (1.0, 1.5, 2.6)
        for seconds, alignment in product(durations, ('prefix', 'onset')):
            pcm, _ = crop(audio, seconds, alignment)
            padding = (-len(pcm)) % 1280
            yield (identifier, seconds, alignment), np.pad(pcm, (0, padding)).astype('<f4'), padding


class NativeStep:
    def __init__(self, artifacts, manifest):
        options = ort.SessionOptions(); options.intra_op_num_threads = 1; options.inter_op_num_threads = 1
        ort.disable_telemetry_events()
        self.session = ort.InferenceSession(str(artifacts / 'encoder.onnx'), sess_options=options, providers=['CPUExecutionProvider'])
        self.slots = manifest['state']
        if self.session.get_providers() != ['CPUExecutionProvider']:
            raise ValueError('Unexpected inference provider')

    def infer(self, pcm):
        states = [np.zeros(s['shape'], np.int64 if s['dtype'] == 'int64' else np.float32) for s in self.slots]
        frames = []
        for offset in range(0, len(pcm), 1280):
            feeds = {'audio': pcm[offset:offset + 1280], **dict(zip((s['input'] for s in self.slots), states))}
            output = self.session.run(None, feeds); states = output[1:]
            frames.append(output[0].reshape(192))
        return np.asarray(frames)


def torch_frames(model, pcm):
    from fastdiar.encoder import StreamingInference
    engine = StreamingInference(model)
    with torch.no_grad():
        return np.asarray([engine.process_chunk(torch.from_numpy(pcm[i:i + 1280].copy())).numpy().reshape(192)
                           for i in range(0, len(pcm), 1280)])


def receive():
    line = sys.stdin.readline(2_000_001)
    if not line or len(line) > 2_000_000:
        raise ValueError('Missing or oversized IPC response')
    return json.loads(line)


def emit(value):
    print(json.dumps(value, allow_nan=False), flush=True)


def prepare(args):
    hashes = verify_inputs(args.registered, VECTOR_MANIFEST)
    if sha256(args.source / 'manifest.jsonl') != SOURCE_MANIFEST or sha256(args.artifacts / 'encoder.onnx') != MODEL_SHA:
        raise ValueError('Pinned source/model hash mismatch')
    manifest = json.loads((args.artifacts / 'manifest.json').read_text())
    if manifest['onnx_sha256'] != MODEL_SHA or manifest['step_samples'] != 1280 or manifest['embedding_dim'] != 192:
        raise ValueError('Incompatible export')
    config = json.loads((args.registered / 'config.json').read_text())
    scenario = next(s for s in config['scenarios'] if s['name'] == 'owner-mixed')
    enrolled = sorted(scenario['enrollment']['owner'])
    lookup = {r['id']: r for r in config['records']}
    check_split(config['records'], enrolled, scenario['queries'])
    if any(lookup[i]['role'] != 'enroll' or lookup[i]['speaker'] != 'owner' for i in enrolled):
        raise ValueError('Reference sources are not target enrollment')
    if any(lookup[i]['role'] not in ('calibration', 'test') for i in scenario['queries']):
        raise ValueError('Queries require frozen roles')
    args.out.mkdir(parents=True, exist_ok=False)
    plan = {'input_hashes': hashes, 'source_manifest': SOURCE_MANIFEST, 'model_sha256': MODEL_SHA,
            'score_margin': SCORE_MARGIN, 'pool_math': 'float64; mean/last normalized, same frame grid across bindings',
            'grid': {'seconds': [1, 1.5, 2.6], 'alignment': ['prefix', 'onset'], 'pooling': ['mean', 'last'],
                     'banks': [[1, 'nearest'], [7, 'nearest'], [7, 'centroid']]},
            'script_hashes': {name: sha256(Path(__file__).with_name(name)) for name in ('binding_cohort.py', 'export_step.py')},
            'shared_helper_hashes': {name: sha256(Path(__file__).parent.parent / 'short-window' / name)
                                     for name in ('audio.py', 'bank_probe.py', 'metrics.py', 'fastdiar_probe.py')},
            'policy': 'Calibration-only bank/threshold selection; fixed Torch bank raw and +0.001 controls; never select on test'}
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    return config, scenario, lookup, enrolled, manifest, plan


def infer_pair(model, native, pcm):
    frames, timing = {}, {}
    for name, inference in [('torch80', lambda: torch_frames(model, pcm)), ('native_ort', lambda: native.infer(pcm))]:
        started = time.perf_counter(); frames[name] = inference()
        if frames[name].shape != (len(pcm) // 1280, 192) or not np.isfinite(frames[name]).all():
            raise ValueError('Invalid native/Torch frame output')
        timing[name] = (time.perf_counter() - started) * 1000
    return frames, timing


def update_parity(parity, frames, index):
    for name in BACKENDS[1:]:
        delta = np.abs(frames[name].astype(np.float64) - frames['torch80'])
        if delta.max() > 1e-4 or delta.mean() > 1e-5:
            raise ValueError(f'{name} numerical parity failed at job {index}: max {delta.max()} mean {delta.mean()}')
        parity[name]['max_abs'] = max(parity[name]['max_abs'], float(delta.max()))
        parity[name]['sum_abs'] += float(delta.sum()); parity[name]['components'] += delta.size


def store_pools(banks, enrolled, key, pcm, frames):
    is_reference = key[0] in enrolled
    vector_key = (key[0], key[2]) if is_reference else key
    has_signal = bool(np.any(pcm))
    pooled = {}
    for name, values in frames.items():
        pooled[name] = pool_frames(values) if has_signal else {'mean': None, 'last': None}
        destination = banks[name][0 if is_reference else 1]
        for mode, value in pooled[name].items():
            destination[(*vector_key, mode)] = value
    return pooled['browser_wasm']


def check_pool_math(pooled, response):
    maximum = 0.0
    for mode, value in pooled.items():
        if value is None:
            continue
        reported = np.asarray(response['pooled'][mode], dtype=np.float64)
        if reported.shape != (192,):
            raise ValueError('Invalid JavaScript pooling dimension')
        difference = float(np.max(np.abs(value - reported)))
        if difference > 1e-10:
            raise ValueError('JavaScript pooling does not match the float64 control')
        maximum = max(maximum, difference)
    return maximum


def timing_summary(timing):
    result = {}
    for name, values in timing.items():
        result[name] = {}
        for seconds in (1.0, 1.5, 2.6, 10.0):
            rows = [r['ms'] for r in values if r['seconds'] == seconds]
            result[name][str(seconds)] = {'n': len(rows), 'p50_ms': float(np.median(rows)),
                                         'p95_ms': float(np.percentile(rows, 95))}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'registered', 'artifacts', 'fastdiar-source', 'checkpoint', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if any(args.out.resolve().is_relative_to(p.resolve()) for p in (args.source, args.registered, args.artifacts)):
        raise ValueError('Outputs must not modify frozen inputs')
    config, scenario, lookup, enrolled, manifest, plan = prepare(args)
    _, model, _, provenance = load_model(args.fastdiar_source, args.checkpoint)
    native = NativeStep(args.artifacts, manifest)
    banks = {name: ({}, {}) for name in BACKENDS}
    timing = {name: [] for name in BACKENDS}
    parity = {name: {'max_abs': 0.0, 'sum_abs': 0.0, 'components': 0} for name in BACKENDS[1:]}
    max_pool_math_delta = 0.0
    count = len(enrolled) * 2 + len(scenario['queries']) * 6
    emit({'ready': True, 'jobs': count, 'plan_sha256': sha256(args.out / 'plan.json')})
    for index, (key, pcm, padding) in enumerate(jobs(args.source, config, scenario)):
        frames, elapsed = infer_pair(model, native, pcm)
        for name, milliseconds in elapsed.items():
            timing[name].append({'seconds': key[1], 'ms': milliseconds})
        digest = hashlib.sha256(pcm.tobytes()).hexdigest()
        emit({'index': index, 'pcm': encode_frames(pcm), 'pcm_sha256': digest})
        response = receive()
        if response.get('index') != index or response.get('pcm_sha256') != digest or response.get('pending_samples') != 0 or response.get('processed_samples') != len(pcm):
            raise ValueError('Browser IPC sample/key coverage mismatch')
        frames['browser_wasm'] = decode_frames(response['frames'], frames['torch80'].shape)
        timing['browser_wasm'].append({'seconds': key[1], 'ms': response['compute_ms']})
        update_parity(parity, frames, index)
        pooled = store_pools(banks, enrolled, key, pcm, frames)
        max_pool_math_delta = max(max_pool_math_delta, check_pool_math(pooled, response))
        emit({'accepted': index, 'last': index + 1 == count})
    if receive().get('command') != 'finish':
        raise ValueError('Missing finish fence')
    for values in parity.values():
        values['mean_abs'] = values.pop('sum_abs') / values['components']
    report = {'plan': plan, 'provenance': provenance, 'jobs': count, 'parity': parity,
              'max_js_pool_math_delta': max_pool_math_delta, 'results': compare_bindings(scenario, lookup, banks),
              'compute': timing_summary(timing),
              'limits': ['Real owner and synthetic non-target cohort; not real household FAR',
                         'Inherited session dependence, exposed test set and repeated exploratory searches',
                         '0.001 is a predeclared numerical diagnostic, not a proven FAR ceiling',
                         'No live profiles, thresholds, enrollment, training or new retained corpus; vectors/PCM only in RAM']}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    emit({'finished': True, 'jobs': count, 'report': str(args.out / 'report.json')})


if __name__ == '__main__':
    main()
