"""Unlabelled invariance/parity controls, separate from speaker accuracy."""
import argparse
import json
from pathlib import Path

import numpy as np

from audio import RATE, MASK_FRAMES, MODES, crop, read_wav, tensors, unit, sha256
from probe import session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("model", "references", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    runtime = session(args.model, 2)
    rows = [json.loads(line) for line in args.references.read_text().splitlines()]
    result = []
    # First three files only: fixed, label-free selection, not tuned to a result.
    for record in rows[:3]:
        pcm = read_wav(record["path"])
        baseline = runtime.run(None, tensors(pcm, "uniform"))[0].reshape(-1)
        max_error = float(np.max(np.abs(baseline - np.asarray(record["embedding"]))))
        if max_error > 1e-5:
            raise ValueError("Uniform path did not reproduce the original wrapper")
        variants = {}
        for mode in MODES:
            inputs = tensors(pcm, mode)
            vector = unit(runtime.run(None, inputs)[0])
            variants[mode] = {"cos_uniform": float(unit(baseline) @ vector)}
        # Same samples, shifted by exactly two seconds; shift the validity mask too.
        shifted = np.concatenate([np.zeros(2 * RATE, np.float32), pcm])
        uniform = tensors(shifted, "uniform")
        masked = tensors(shifted, "valid")
        points = np.floor(np.arange(MASK_FRAMES) * 160000 / MASK_FRAMES).astype(int)
        masked["weights"][0, points < 2 * RATE] = 0
        v0 = unit(runtime.run(None, tensors(pcm, "valid"))[0])
        result.append({"source_sha256": sha256(record["path"]), "uniform_max_abs_error": max_error,
                       "variants": variants,
                       "shift_uniform_cos": float(unit(baseline) @ unit(runtime.run(None, uniform)[0])),
                       "shift_valid_cos": float(v0 @ unit(runtime.run(None, masked)[0]))})
    with args.out.open("x") as out:
        out.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"cases": len(result), "max_uniform_error": max(r["uniform_max_abs_error"] for r in result)}))


if __name__ == "__main__":
    main()
