"""Read-only native CPU equivalent of the temporary browser reference/query test."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from binding_cohort import MODEL_SHA, NativeStep, pool_frames
from audio import read_wav, sha256


def prepare(path, maximum):
    pcm = read_wav(path)[:maximum]
    if len(pcm) < 24320:
        raise ValueError('Need at least 1.52 seconds of mono PCM16 at 16 kHz.')
    rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)))
    if rms < 0.0001:
        raise ValueError('Audio level is too low for this diagnostic.')
    return np.pad(pcm, (0, (-len(pcm)) % 1280)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--reference', type=Path, action='append', required=True, help='Own-voice reference WAV; repeat up to seven times')
    parser.add_argument('--query', type=Path, required=True, help='Query WAV, first 1.52 seconds')
    args = parser.parse_args()
    try:
        if len(args.reference) > 7 or sha256(args.artifacts / 'encoder.onnx') != MODEL_SHA:
            raise ValueError('Reference limit exceeded or incompatible model.')
        manifest = json.loads((args.artifacts / 'manifest.json').read_text())
        engine = NativeStep(args.artifacts, manifest)
        vectors = [pool_frames(engine.infer(prepare(p, 48640)))['mean'] for p in args.reference]
        query = pool_frames(engine.infer(prepare(args.query, 24320)))['mean']
        if query is None or any(value is None for value in vectors):
            raise ValueError('Insufficient usable embedding data.')
        refs = np.asarray(vectors)
        centroid = refs.mean(axis=0); norm = np.linalg.norm(centroid)
        print(json.dumps({'status': 'Uncalibrated', 'modelSha256': MODEL_SHA, 'references': len(refs),
                          'nearest': float(np.max(refs @ query)),
                          'centroid': float(centroid @ query / norm) if norm > 1e-12 else None,
                          'pooling': 'frame-mean-f64', 'persistence': False}, allow_nan=False))
        return 0
    except Exception as error:
        print(json.dumps({'status': 'Error', 'code': 'SPEAKER_TEST_ERROR', 'message': str(error)}))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
