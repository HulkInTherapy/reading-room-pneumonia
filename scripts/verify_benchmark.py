"""Recompute the complete benchmark with Python's standard library only."""

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research/m4_m5/results"


def ratio(numerator, denominator):
    return numerator / denominator if denominator else 0.0


def roc_auc(labels, scores):
    """Mann–Whitney rank statistic with average ranks for tied scores."""
    pairs = sorted(zip(scores, labels))
    positive = sum(labels)
    negative = len(labels) - positive
    if not positive or not negative:
        raise ValueError("ROC AUC requires both classes.")
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][0] == pairs[index][0]:
            end += 1
        mean_rank = (index + 1 + end) / 2
        rank_sum += mean_rank * sum(label for _, label in pairs[index:end])
        index = end
    return (rank_sum - positive * (positive + 1) / 2) / (positive * negative)


def metrics(labels, scores):
    matrix = [[0, 0], [0, 0]]
    for label, score in zip(labels, scores):
        matrix[label][int(score >= 0.5)] += 1
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    normal_f1 = ratio(2 * tn, 2 * tn + fp + fn)
    pneumonia_f1 = ratio(2 * tp, 2 * tp + fp + fn)
    return {
        "n": len(labels),
        "accuracy": ratio(tn + tp, len(labels)),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "precision": ratio(tp, tp + fp),
        "normal_f1": normal_f1,
        "pneumonia_f1": pneumonia_f1,
        "macro_f1": (normal_f1 + pneumonia_f1) / 2,
        "auc": roc_auc(labels, scores),
        "confusion_matrix": matrix,
    }


def verify():
    with (RESULTS / "predictions.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 624:
        raise ValueError("Expected 624 test records; found " + str(len(rows)))
    labels = [int(row["label"]) for row in rows]
    if any(label not in [0, 1] for label in labels):
        raise ValueError("Labels must be 0 (normal) or 1 (pneumonia).")
    expected = json.loads((RESULTS / "summary.json").read_text())
    report = {}
    for condition in "ABCD":
        scores = [float(row[condition]) for row in rows]
        if any(not math.isfinite(score) or score < 0 or score > 1 for score in scores):
            raise ValueError("Invalid probability in condition " + condition)
        measured = metrics(labels, scores)
        saved = expected["conditions"][condition]["metrics"]
        for key, value in measured.items():
            if key == "n":
                continue
            if key == "confusion_matrix":
                if value != saved[key]:
                    raise ValueError(
                        condition + " confusion matrix does not match saved results."
                    )
            elif abs(value - saved[key]) > 1e-10:
                raise ValueError(condition + " " + key + " differs from saved results.")
        report[condition] = measured
    return {
        "verified": True,
        "records": len(rows),
        "dependencies": "Python standard library only",
        "conditions": report,
    }


if __name__ == "__main__":
    try:
        output = verify()
        print("PASS: all four conditions match saved metrics on 624 test records.")
        print(json.dumps(output, indent=2))
    except (OSError, ValueError, KeyError) as exc:
        print("Benchmark verification failed: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
