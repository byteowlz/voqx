"""Exploratory metrics. Abstentions stay in denominators; thresholds use calibration only."""
import numpy as np


def auc(positive, negative):
    if not positive or not negative:
        return None
    delta = np.asarray(positive)[:, None] - np.asarray(negative)[None, :]
    return float(np.mean((delta > 0) + 0.5 * (delta == 0)))


def summarize(rows, calibration):
    """Report conditional AUC and calibration-zero-FA operating point (not a guarantee)."""
    positives = [r for r in rows if r["target"]]
    negatives = [r for r in rows if not r["target"]]
    pos_scores = [r["score"] for r in positives if r["score"] is not None]
    neg_scores = [r["score"] for r in negatives if r["score"] is not None]
    cal_neg = [r["score"] for r in calibration if not r["target"] and r["score"] is not None]
    threshold = float(np.nextafter(max(cal_neg), np.inf)) if cal_neg else None
    return {
        "conditional_auc": auc(pos_scores, neg_scores),
        "positive_median": float(np.median(pos_scores)) if pos_scores else None,
        "negative_median": float(np.median(neg_scores)) if neg_scores else None,
        "positive_n": len(positives), "negative_n": len(negatives),
        "positive_abstain": len(positives) - len(pos_scores),
        "negative_abstain": len(negatives) - len(neg_scores),
        "threshold_calibration_zero_fa": threshold,
        "test_positive_accept": sum(s >= threshold for s in pos_scores) if threshold is not None else None,
        "test_negative_accept": sum(s >= threshold for s in neg_scores) if threshold is not None else None,
    }


def check_split(records, enroll_ids, query_ids):
    lookup = {r["id"]: r for r in records}
    if len(lookup) != len(records) or set(enroll_ids) & set(query_ids):
        raise ValueError("IDs must be unique and enrollment/query disjoint")
    def hashes(ids):
        return {lookup[i].get("source_sha256", lookup[i]["sha256"]) for i in ids}
    if hashes(enroll_ids) & hashes(query_ids):
        raise ValueError("A source crosses enrollment and query splits")


def score_scenario(scenario, records, vectors, condition):
    """Same preprocessing for references and queries; no enrollment registry changes."""
    enrolled = scenario["enrollment"].get("owner", [])
    queries = scenario["queries"]
    check_split(records, enrolled, queries)
    refs = [vectors.get((i, *condition)) for i in enrolled]
    refs = [v for v in refs if v is not None]
    lookup = {r["id"]: r for r in records}
    if not refs:
        raise ValueError("No usable enrollment vectors")
    mean = np.mean(refs, axis=0)
    mean /= np.linalg.norm(mean)
    result = []
    for identifier in queries:
        record = lookup[identifier]
        if record["role"] not in ("calibration", "test"):
            continue
        query = vectors.get((identifier, *condition))
        row = {"id": identifier, "target": record["speaker"] == "owner", "role": record["role"]}
        result.append({**row, "score": float(np.max(np.asarray(refs) @ query)) if query is not None else None,
                       "centroid": float(mean @ query) if query is not None else None})
    return result
