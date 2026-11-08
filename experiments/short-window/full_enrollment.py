"""Cross-duration control: full enrollment clips against the cached short queries."""
import argparse
import json
import os
from pathlib import Path

import numpy as np

from audio import MODES, crop, read_wav, sha256
from probe import embed, evaluate, session


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "config", "model", "vectors", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    args.out.mkdir(exist_ok=False, parents=True)
    config = json.loads(args.config.read_text())
    vectors = {}
    for line in args.vectors.read_text().splitlines():
        row = json.loads(line)
        vectors[(row["id"], *row["condition"])] = np.asarray(row["vector"]) if row["vector"] is not None else None
    runtime = session(args.model, 2)
    enrolled = {i for s in config["scenarios"] for i in s["enrollment"]["owner"]}
    details = []
    for record in config["records"]:
        if record["id"] not in enrolled:
            continue
        path = (args.source / record["path"]).resolve()
        if not path.is_relative_to(args.source.resolve()) or sha256(path) != record["sha256"]:
            raise ValueError("Source path/hash mismatch")
        audio = read_wav(path)
        for alignment in ("prefix", "onset"):
            pcm, offset = crop(audio, 10, alignment)
            for mode in MODES:
                vector, elapsed, reason = embed(runtime, pcm, mode)
                details.append({"id": record["id"], "alignment": alignment, "mask": mode,
                                "samples": len(pcm), "offset": offset, "inference_ms": elapsed,
                                "abstention": reason, "vector": vector.tolist() if vector is not None else None})
                for seconds in (1.0, 1.5, 2.6):
                    vectors[(record["id"], seconds, alignment, mode)] = vector
    (args.out / "enrollment-vectors.json").write_text(json.dumps(details, allow_nan=False) + "\n")
    (args.out / "report.json").write_text(json.dumps(evaluate(config, vectors), indent=2, allow_nan=False) + "\n")
    (args.out / "provenance.json").write_text(json.dumps({
        "query_vectors_sha256": sha256(args.vectors), "config_sha256": sha256(args.config),
        "model_sha256": sha256(args.model), "model_weights_sha256": sha256(str(args.model) + ".data"),
        "scripts": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "scope": "Reference crops only changed to full leading window; query vectors reused exactly",
    }, indent=2) + "\n")
    print(json.dumps({"enrollment_vectors": len(details)}))


if __name__ == "__main__":
    main()
