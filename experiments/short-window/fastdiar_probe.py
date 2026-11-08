"""CPU-only FASTDIAR-small verification diagnostic; no saved audio/embedding corpus.

Requires an explicit pinned source checkout and checkpoint. Uses an existing Torch
interpreter without installing/updating dependencies. The source's causal streaming
encoder is tested, NOT its 960-ms-delayed VAD/clustering diarizer. Mean/last pooling
are exploratory identity baselines, not approved voqx profiles or enrollment.
"""
import argparse
from itertools import product
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np

from audio import crop, read_wav, sha256
from bank_probe import choose_bank, verify_inputs
from metrics import check_split

REVISION = 'f48bff53b9a6a0cce4539e94b6def95f609eaea4'
CHECKPOINT = '90be5da5e86645c39eceaedfd9f05cc7d35734d9105691da3e69b3e4ec47fb2a'
SOURCE_MANIFEST = '52559fdffd4af9e1a11f0790c1d5724df22697776b0754aaf992635cf30b0227'
VECTOR_MANIFEST = '13dd6b0c5222e2dcd48364700542ab68dfe5df0f2170c297759c6b80767e72f3'


def deny_network(*_args, **_kwargs):
    raise RuntimeError('Network disabled in the CPU speaker probe')


def load_model(source, checkpoint):
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION or subprocess.call(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--', 'fastdiar']):
        raise ValueError('Expected the unchanged pinned FASTDIAR source')
    if sha256(checkpoint) != CHECKPOINT:
        raise ValueError('Checkpoint SHA-256 mismatch')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUDA_VISIBLE_DEVICES='')
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = socket.getaddrinfo = deny_network
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source.resolve()))
    import torch
    from fastdiar.encoder import StreamingEncoder, load_streaming_model
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    started = time.perf_counter()
    model = load_streaming_model(str(checkpoint), device='cpu', dtype=torch.float32)
    if any(p.device.type != 'cpu' for p in list(model.parameters()) + list(model.buffers())):
        raise ValueError('CPU-only invariant failed')
    return torch, model, StreamingEncoder, {
        'source_revision': revision, 'checkpoint_sha256': CHECKPOINT, 'torch': torch.__version__,
        'numpy': np.__version__, 'device': 'cpu', 'threads': 2,
        'parameters': sum(p.numel() for p in model.parameters()), 'checkpoint_bytes': checkpoint.stat().st_size,
        'load_ms': (time.perf_counter() - started) * 1000,
    }


def stream_frames(torch, model, encoder_type, pcm, chunks=None):
    encoder = encoder_type(model, shift_frames=4)
    frame_samples = encoder.frame_samples
    padding = (-len(pcm)) % frame_samples
    data = np.pad(pcm, (0, padding)).astype(np.float32)
    sizes = chunks or [frame_samples * 4] * ((len(data) + frame_samples * 4 - 1) // (frame_samples * 4))
    frames, at = [], 0
    started = time.perf_counter()
    for size in sizes:
        if at >= len(data):
            break
        part = data[at:at + size]; at += len(part)
        frames.extend(encoder.push(torch.from_numpy(part.copy())))
    if at != len(data):
        raise ValueError('Chunk schedule did not consume all audio')
    frames.extend(encoder.flush())
    elapsed = (time.perf_counter() - started) * 1000
    result = torch.stack(frames).numpy()
    if result.shape != (len(data) // frame_samples, 192) or not np.isfinite(result).all():
        raise ValueError('Invalid frame embeddings')
    return result, elapsed, padding


def pool_frames(frames):
    result = {}
    for name, value in [('mean', frames.mean(axis=0)), ('last', frames[-1])]:
        norm = float(np.linalg.norm(value))
        result[name] = value / norm if norm > 1e-12 and np.isfinite(norm) else None
    return result


def smoke(torch, model, encoder_type):
    x = np.arange(16000 * 2 + 17, dtype=np.float32)
    signal = (np.sin(x * 0.07) * (0.1 + 0.05 * np.sin(x * 0.001))).astype(np.float32)
    normal, ms, padding = stream_frames(torch, model, encoder_type, signal)
    fragmented, _, _ = stream_frames(torch, model, encoder_type, signal, [997] * 40)
    jumbo, _, _ = stream_frames(torch, model, encoder_type, signal, [19001, 19001])
    delta = np.concatenate([np.abs(normal - fragmented).ravel(), np.abs(normal - jumbo).ravel()])
    report = {'frames': len(normal), 'padding_samples': padding, 'synthetic_compute_ms': ms,
              'chunk_parity_max_abs': float(delta.max()), 'chunk_parity_mean_abs': float(delta.mean())}
    if delta.max() > 1e-3 or delta.mean() > 1e-4:
        raise ValueError(f'Arbitrary-chunk equivalence failed: {report}')
    return report


def checked_audio(source, record):
    path = (source / record['path']).resolve(strict=True)
    if not path.is_relative_to(source.resolve()) or sha256(path) != record['sha256']:
        raise ValueError('Source audio path/hash mismatch')
    return read_wav(path)


def infer_cohort(torch, model, encoder_type, source, config):
    scenario = next(s for s in config['scenarios'] if s['name'] == 'owner-mixed')
    enrolled = sorted(scenario['enrollment']['owner'])
    check_split(config['records'], enrolled, scenario['queries'])
    if sha256(source / 'manifest.jsonl') != SOURCE_MANIFEST:
        raise ValueError('Registered audio manifest changed')
    lookup = {r['id']: r for r in config['records']}
    if any(lookup[i]['role'] != 'enroll' or lookup[i]['speaker'] != 'owner' for i in enrolled):
        raise ValueError('Only target enrollment sources may build reference profiles')
    references, queries, timings, pads = {}, {}, [], []
    for number, identifier in enumerate(enrolled + scenario['queries'], 1):
        record = lookup[identifier]
        audio = checked_audio(source, record)
        durations = (10.0,) if identifier in enrolled else (1.0, 1.5, 2.6)
        for seconds in durations:
            for alignment in ('prefix', 'onset'):
                pcm, _ = crop(audio, seconds, alignment)
                if not np.any(pcm):
                    pooled = {'mean': None, 'last': None}
                else:
                    frames, elapsed, padding = stream_frames(torch, model, encoder_type, pcm)
                    pooled = pool_frames(frames); timings.append({'ms': elapsed, 'seconds': seconds}); pads.append(padding)
                for pooling, value in pooled.items():
                    if identifier in enrolled:
                        references[(identifier, alignment, pooling)] = value
                    else:
                        queries[(identifier, seconds, alignment, pooling)] = value
        if number % 20 == 0:
            print(json.dumps({'finished_sources': number, 'total_sources': len(enrolled) + len(scenario['queries'])}), flush=True)
    elapsed = [t['ms'] for t in timings]
    by_duration = {}
    for seconds in (1.0, 1.5, 2.6, 10.0):
        times = [t['ms'] for t in timings if t['seconds'] == seconds]
        by_duration[str(seconds)] = {'n': len(times), 'p50_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times, 95))}
    stats = {'inferences': len(timings), 'compute_ms_p50': float(np.median(elapsed)),
             'compute_ms_p95': float(np.percentile(elapsed, 95)), 'by_requested_duration': by_duration,
             'maximum_tail_padding_samples': max(pads)}
    return scenario, lookup, references, queries, stats


def compare(scenario, lookup, references, queries):
    results = []
    for seconds, alignment, pooling in product((1.0, 1.5, 2.6), ('prefix', 'onset'), ('mean', 'last')):
        refs = [{'id': i, 'vector': references[(i, alignment, pooling)]} for i in sorted(scenario['enrollment']['owner'])]
        if any(r['vector'] is None for r in refs):
            raise ValueError('Reference abstention: do not silently alter the bank')
        rows = [{'id': i, 'target': lookup[i]['speaker'] == 'owner', 'role': lookup[i]['role'],
                 'vector': queries[(i, seconds, alignment, pooling)]} for i in scenario['queries']]
        for size, method in ((1, 'nearest'), (7, 'nearest'), (7, 'centroid')):
            results.append({'seconds': seconds, 'alignment': alignment, 'pooling': pooling,
                            'bank_size': size, 'method': method, **choose_bank(refs, rows, size, method)})
    return results


def peak_rss_bytes():
    try:
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if sys.platform == 'darwin' else value * 1024)
    except ImportError:
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'registered', 'fastdiar-source', 'checkpoint', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    if args.out.resolve().is_relative_to(args.source.resolve()) or args.out.resolve().is_relative_to(args.registered.resolve()):
        raise ValueError('Report must not modify either immutable input dataset')
    hashes = verify_inputs(args.registered, VECTOR_MANIFEST)
    config = json.loads((args.registered / 'config.json').read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    torch, model, encoder_type, provenance = load_model(args.fastdiar_source, args.checkpoint)
    parity = smoke(torch, model, encoder_type)
    (args.out / 'smoke.json').write_text(json.dumps({'provenance': provenance, 'smoke': parity}, indent=2))
    scenario, lookup, refs, queries, stats = infer_cohort(torch, model, encoder_type, args.source, config)
    report = {'provenance': provenance, 'input_hashes': hashes, 'smoke': parity, 'compute': stats,
              'script_hashes': {name: sha256(Path(__file__).with_name(name)) for name in ('fastdiar_probe.py', 'bank_probe.py', 'audio.py', 'metrics.py')},
              'results': compare(scenario, lookup, refs, queries), 'peak_process_rss_bytes': peak_rss_bytes(),
              'limits': ['CPU Torch reference, not ONNX/WASM or a shipped voqx profile; peak RSS includes the full Python/Torch process',
                         'Causal encoder only: 960 ms upstream diarization/VAD/clustering delay is not measured here',
                         'No VAD; onset is an offline energy-aligned diagnostic; final frame is explicitly zero-padded up to 80 ms',
                         'Mean/last frame pooling is exploratory, not a qualified speaker-verification head',
                         'Same inherited session-dependent owner and synthetic non-target cohort; not household FAR',
                         'Embeddings stay in RAM; no new latent corpus, enrollment, training or deployed thresholds']}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(args.out / 'report.json'), 'compute': stats}), flush=True)


if __name__ == '__main__':
    main()
