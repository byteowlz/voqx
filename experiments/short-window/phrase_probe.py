"""Paired phrase-template comparison from existing vectors; no training or new enrollment.

Mapping is private JSON id -> {language, phrase}. Match exact prompted wording,
not an ASR guess. Compare equal-sized single-reference trials; no threshold tuning.
This measures target consistency only, not rejection of other speakers saying the phrase.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from metrics import check_split


def compare(config, vectors, phrases):
    records = config["records"]
    enrollment = [r["id"] for r in records if r["role"] == "enroll" and r["speaker"] == "owner" and r["id"] in phrases]
    queries = [r["id"] for r in records if r["role"] == "test" and r["speaker"] == "owner" and r["id"] in phrases]
    check_split(records, enrollment, queries)
    conditions = sorted({key[1:] for key in vectors})
    output = []
    for condition in conditions:
        paired = []
        for query_id in queries:
            query = vectors.get((query_id, *condition))
            if query is None:
                continue
            key = phrases[query_id]
            # Same language for both template arms. Average single-template scores,
            # not max over different numbers of references.
            same, other = [], []
            for ref_id in enrollment:
                ref = vectors.get((ref_id, *condition))
                if ref is None or phrases[ref_id]["language"] != key["language"]:
                    continue
                destination = same if phrases[ref_id]["phrase"] == key["phrase"] else other
                destination.append(float(query @ ref))
            if same and other:
                paired.append({"id": query_id, "same": float(np.mean(same)), "other": float(np.mean(other))})
        output.append({"condition": condition, "paired_queries": len(paired),
                       "same_median": float(np.median([p["same"] for p in paired])) if paired else None,
                       "other_median": float(np.median([p["other"] for p in paired])) if paired else None,
                       "median_paired_delta": float(np.median([p["same"] - p["other"] for p in paired])) if paired else None,
                       "same_wins": sum(p["same"] > p["other"] for p in paired),
                       "limitations": "Prompt labels not listening-confirmed; target consistency, not speaker discrimination"})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("config", "vectors", "phrases", "out"):
        parser.add_argument("--" + arg, type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    phrases = json.loads(args.phrases.read_text())
    vectors = {}
    for line in args.vectors.read_text().splitlines():
        row = json.loads(line)
        vectors[(row["id"], *row["condition"])] = np.asarray(row["vector"]) if row["vector"] is not None else None
    result = compare(config, vectors, phrases)
    with args.out.open("x") as out:
        out.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"conditions": len(result), "max_pairs": max(r["paired_queries"] for r in result)}))


if __name__ == "__main__":
    main()
