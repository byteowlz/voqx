"""Read-only CPU ablation of speakrs' fixed-window embedding graph; explicit local model only."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort

from audio import MODES, RATE, activity, crop, read_wav, sha256, tensors, unit
from metrics import score_scenario, summarize


def session(model, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    model_session = ort.InferenceSession(str(model), sess_options=options, providers=["CPUExecutionProvider"])
    inputs = {i.name: i.shape for i in model_session.get_inputs()}
    if inputs != {"waveform": [1, 1, 160000], "weights": [1, 589]}:
        raise ValueError(f"Unsupported model inputs: {inputs}")
    return model_session


def embed(model_session, pcm, mode):
    inputs = tensors(pcm, mode)
    if not np.any(activity(pcm)) or not np.any(inputs["weights"]):
        return None, 0.0, "no_activity"
    start = time.perf_counter()
    output = model_session.run(None, inputs)[0].reshape(-1)
    elapsed = (time.perf_counter() - start) * 1000
    # Sparse pooling masks can yield NaN in the exported statistics pooling.
    # Keep these trials as explicit abstentions, never zero-score negatives.
    if not np.isfinite(output).all() or np.linalg.norm(output) < 1e-12:
        return None, elapsed, "invalid_embedding"
    return unit(output), elapsed, None


def infer(config, source, destination, model_session):
    vectors, timings = {}, []
    with (destination / "vectors.jsonl").open("x") as out:
        for record in config["records"]:
            path = (source / record["path"]).resolve()
            if not path.is_relative_to(source.resolve()) or sha256(path) != record["sha256"]:
                raise ValueError("Source path or hash mismatch")
            audio = read_wav(path)
            for seconds in (1.0, 1.5, 2.6):
                for alignment in ("prefix", "onset"):
                    pcm, offset = crop(audio, seconds, alignment)
                    for mode in MODES:
                        vector, elapsed, abstention = embed(model_session, pcm, mode)
                        condition = (seconds, alignment, mode)
                        vectors[(record["id"], *condition)] = vector
                        if vector is not None:
                            timings.append(elapsed)
                        row = {"id": record["id"], "condition": condition, "source_sha256": record["sha256"],
                               "offset_samples": offset, "observed_samples": len(pcm),
                               "activity_proxy_ms": float(activity(pcm).sum() / RATE * 1000),
                               "inference_ms": elapsed, "abstention": abstention,
                               "vector": vector.tolist() if vector is not None else None}
                        out.write(json.dumps(row, allow_nan=False) + "\n")
            out.flush()
            print(json.dumps({"finished_record": record["id"], "conditions": len(vectors)}), flush=True)
    return vectors, timings


def evaluate(config, vectors):
    reports = []
    for scenario in config["scenarios"]:
        if set(scenario["enrollment"]) != {"owner"}:
            continue
        for seconds in (1.0, 1.5, 2.6):
            for alignment in ("prefix", "onset"):
                for mode in MODES:
                    rows = score_scenario(scenario, config["records"], vectors, (seconds, alignment, mode))
                    for method in ("sample", "centroid"):
                        scored = [{**r, "score": r["score"] if method == "sample" else r["centroid"]} for r in rows]
                        reports.append({"scenario": scenario["name"], "seconds": seconds,
                                        "alignment": alignment, "mask": mode, "method": method,
                                        **summarize([r for r in scored if r["role"] == "test"],
                                                    [r for r in scored if r["role"] == "calibration"])})
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2, choices=range(1, 5))
    args = parser.parse_args()
    os.umask(0o077)
    args.out.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_text())
    started = time.perf_counter()
    model_session = session(args.model, args.threads)
    load_ms = (time.perf_counter() - started) * 1000
    vectors, timings = infer(config, args.source, args.out, model_session)
    reports = evaluate(config, vectors)
    provenance = {"model_sha256": sha256(args.model), "external_weights_sha256": sha256(str(args.model) + ".data"),
                  "config_sha256": sha256(args.config), "ort_version": ort.__version__, "threads": args.threads,
                  "load_ms": load_ms, "inference_ms_p50": float(np.median(timings)),
                  "inference_ms_p95": float(np.percentile(timings, 95)), "inference_count": len(timings),
                  "elapsed_seconds": time.perf_counter() - started,
                  "scripts": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
                  "limitations": ["Energy mask is not VAD or speaker identification",
                                  "Synthetic negatives do not measure household false-accept rate",
                                  "Duration crops share parent recordings and are not independent trials",
                                  "Masking does not remove padded context from feature extraction",
                                  "No registry mutations, threshold deployment, training or network inference"]}
    (args.out / "report.json").write_text(json.dumps(reports, indent=2, allow_nan=False) + "\n")
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"records": len(config["records"]), "comparisons": len(reports),
                      "inference_ms_p50": provenance["inference_ms_p50"], "elapsed_seconds": provenance["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
