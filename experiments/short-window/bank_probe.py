"""Read-only reference-bank scoring sweep over registered, frozen CPU embeddings.

No model inference, new vectors, training, enrollment or threshold installation. Bank
selection and threshold fitting use calibration only. Test queries never select a bank.
Output is an experiment report, not a new corpus. Inherited session dependence and
synthetic-impostor limitations remain explicit.
"""
import argparse
from itertools import combinations, product
import json
import os
from pathlib import Path
import time

import numpy as np

from metrics import check_split, summarize
from audio import sha256 as digest


def verify_inputs(root, expected_manifest):
    manifest = root / 'manifest.jsonl'
    if digest(manifest) != expected_manifest:
        raise ValueError('Registered manifest hash mismatch')
    entries = {row['path']: row for row in map(json.loads, manifest.read_text().splitlines())}
    required = ('config.json', 'run-2/vectors.jsonl', 'full-enrollment/enrollment-vectors.json')
    hashes = {}
    for relative in required:
        path = root / relative
        if path.stat().st_size != entries[relative]['bytes'] or digest(path) != entries[relative]['sha256']:
            raise ValueError(f'Registered input changed: {relative}')
        hashes[relative] = entries[relative]['sha256']
    return hashes


def vector(value):
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (256,) or not np.isfinite(result).all() or not np.isclose(np.linalg.norm(result), 1, atol=1e-5):
        raise ValueError('Expected a finite unit 256-dimensional WeSpeaker vector')
    return result


def score_rows(query_rows, references, method):
    refs = np.asarray(references)
    if method == 'centroid':
        mean = refs.mean(axis=0)
        norm = np.linalg.norm(mean)
        if norm < 1e-12:
            raise ValueError('Degenerate centroid')
        refs = (mean / norm)[None, :]
    elif method != 'nearest':
        raise ValueError('Unsupported scoring method')
    return [{k: row[k] for k in ('id', 'target', 'role')} | {
        'score': float(np.max(refs @ row['vector'])) if row['vector'] is not None else None,
    } for row in query_rows]


def calibration_key(rows):
    summary = summarize(rows, rows)
    if summary['threshold_calibration_zero_fa'] is None or not summary['positive_n']:
        raise ValueError('Need calibration positives and non-targets')
    # Maximise accepted calibration targets; break ties by ranking, never test outcomes.
    return (summary['test_positive_accept'], summary['conditional_auc'] or 0)


def choose_bank(enrollment, query_rows, size, method):
    calibration = [r for r in query_rows if r['role'] == 'calibration']
    best = None
    for subset in combinations(enrollment, size):
        ids = tuple(row['id'] for row in subset)
        refs = [row['vector'] for row in subset]
        key = calibration_key(score_rows(calibration, refs, method))
        if best is None or key > best[0]:
            best = (key, ids, refs)
    if best is None:
        raise ValueError('No candidate bank at this size')
    rows = score_rows(query_rows, best[2], method)
    cal = [row for row in rows if row['role'] == 'calibration']
    test = [row for row in rows if row['role'] == 'test']
    report = summarize(test, cal)
    report['selected_references'] = best[1]
    report['calibration'] = summarize(cal, cal)
    # An unchanged score threshold illustrates bank growth; not a deployment recommendation.
    report['unchanged_0_60'] = {
        'positive_accept': sum(r['target'] and r['score'] is not None and r['score'] >= 0.6 for r in test),
        'negative_accept': sum(not r['target'] and r['score'] is not None and r['score'] >= 0.6 for r in test),
    }
    return report


def load_vectors(root, config):
    records = {r['id']: r for r in config['records']}
    queries = {}
    for row in map(json.loads, (root / 'run-2/vectors.jsonl').read_text().splitlines()):
        if row['source_sha256'] != records[row['id']]['sha256']:
            raise ValueError('Vector/source hash mismatch')
        key = (row['id'], *row['condition'])
        if key in queries:
            raise ValueError('Duplicate query vector')
        queries[key] = vector(row['vector'])
    references = {}
    for row in json.loads((root / 'full-enrollment/enrollment-vectors.json').read_text()):
        key = (row['id'], row['alignment'], row['mask'])
        if key in references:
            raise ValueError('Duplicate reference vector')
        references[key] = vector(row['vector'])
    return queries, references


def evaluate(config, queries, references):
    scenario = next(s for s in config['scenarios'] if s['name'] == 'owner-mixed')
    ids = sorted(scenario['enrollment']['owner'])
    check_split(config['records'], ids, scenario['queries'])
    records = {r['id']: r for r in config['records']}
    if any(records[i]['role'] != 'enroll' or records[i]['speaker'] != 'owner' for i in ids):
        raise ValueError('A query or non-target entered the reference bank')
    results = []
    for seconds, alignment, mask in product((1.0, 1.5, 2.6), ('prefix', 'onset'), ('uniform', 'valid')):
        enrolled = [{'id': i, 'vector': references[(i, alignment, mask)]} for i in ids]
        if any(row['vector'] is None for row in enrolled):
            raise ValueError('Reference abstention; do not silently change bank size')
        rows = [{'id': i, 'target': records[i]['speaker'] == 'owner', 'role': records[i]['role'],
                 'vector': queries[(i, seconds, alignment, mask)]} for i in scenario['queries']
                if records[i]['role'] in ('calibration', 'test')]
        for method in ('nearest', 'centroid'):
            for size in range(1, len(ids) + 1):
                started = time.perf_counter()
                result = choose_bank(enrolled, rows, size, method)
                results.append({'seconds': seconds, 'alignment': alignment, 'mask': mask,
                                'method': method, 'bank_size': size, **result,
                                'sweep_ms': (time.perf_counter() - started) * 1000})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Registered short-window-ablation/v1 root')
    parser.add_argument('--manifest-sha256', required=True, help='Pinned digest from the registry card')
    parser.add_argument('--out', type=Path, required=True, help='New experiment-report directory, never the dataset root')
    args = parser.parse_args()
    os.umask(0o077)
    root = args.input.resolve(strict=True)
    if args.out.resolve().is_relative_to(root):
        raise ValueError('Immutable dataset root cannot hold experiment outputs')
    hashes = verify_inputs(root, args.manifest_sha256)
    config = json.loads((root / 'config.json').read_text())
    queries, references = load_vectors(root, config)
    started = time.perf_counter()
    results = evaluate(config, queries, references)
    report = {'input_hashes': hashes, 'manifest_sha256': args.manifest_sha256,
              'script_sha256': digest(__file__), 'numpy_version': np.__version__,
              'helper_hashes': {name: digest(Path(__file__).with_name(name)) for name in ('metrics.py', 'audio.py')},
              'selection': 'Exhaust all enrollment-only subsets at each size; calibration-only target acceptance then AUC; stable ID-order tie break',
              'scope': 'CPU scoring replay of cached registered vectors; no new encoder or browser inference',
              'limitations': ['Synthetic non-target voices, not measured household FAR',
                              'Enrollment and short owner queries share a capture session; source-hash disjoint is not session-disjoint',
                              'Dependent duration/channel transforms and repeated calibration searches are exploratory',
                              'No training, new corpus, live registry changes or deployed thresholds'],
              'elapsed_seconds': time.perf_counter() - started, 'results': results}
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'comparisons': len(results), 'elapsed_seconds': report['elapsed_seconds'], 'report': str(args.out / 'report.json')}))


if __name__ == '__main__':
    main()
