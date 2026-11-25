"""Evaluate the trained intent classifier on the grouped holdout split.

Reports accuracy, macro/weighted F1, per-class precision/recall/F1, the
confusion matrix and a leakage check.  The old script reported only overall
accuracy, which hid both class imbalance and the fact that every test batch
also appeared in train.

Usage::

    python evaluation/evaluate.py
    python evaluation/evaluate.py --split test_holdout
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

SPLIT_DIR = os.path.join("data", "processed")
MODEL_DIR = os.path.join("models", "intent_classifier")
REPORT_PATH = os.path.join("evaluation", "report.json")
MATRIX_PATH = os.path.join("evaluation", "confusion_matrix.csv")


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def check_leakage(split: str, test_df: pd.DataFrame) -> dict:
    train_path = os.path.join(SPLIT_DIR, "train.jsonl")
    if split == "train" or not os.path.exists(train_path):
        return {"checked": False}
    train_df = load_jsonl(train_path)
    train_batches = set(train_df["source_batch"])
    test_batches = set(test_df["source_batch"])
    overlap = sorted(train_batches & test_batches)
    return {
        "checked": True,
        "train_batches": len(train_batches),
        "test_batches": len(test_batches),
        "overlap": overlap,
        "leak": bool(overlap),
    }


def save_confusion_matrix(labels: list[str], matrix, path: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [int(value) for value in row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="test_holdout")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--report-out", default=REPORT_PATH)
    parser.add_argument("--matrix-out", default=MATRIX_PATH)
    args = parser.parse_args()

    test_path = os.path.join(SPLIT_DIR, f"{args.split}.jsonl")
    if not os.path.exists(test_path):
        raise SystemExit(f"Missing split: {test_path}. Run data_pipeline/clean_data.py first.")
    if not os.path.isdir(args.model_dir):
        raise SystemExit(f"Missing model: {args.model_dir}. Run training/train_classifier.py first.")

    print(f"Loading {test_path} ...")
    test_df = load_jsonl(test_path).dropna(subset=["utterance", "intent_code"])
    if test_df.empty:
        raise SystemExit("Test split is empty.")

    print(f"Loading model from {args.model_dir} ...")
    from setfit import SetFitModel

    model = SetFitModel.from_pretrained(args.model_dir)

    leakage = check_leakage(args.split, test_df)
    if leakage.get("leak"):
        print("\nWARNING: train/test batch leakage detected:")
        print("  ", leakage["overlap"])
        print("  Metrics below are inflated. Re-run clean_data.py with grouped splits.\n")

    print(f"Running predictions on {len(test_df)} rows ...")
    y_true = list(test_df["intent_code"])
    y_pred = list(model.predict(list(test_df["utterance"])))

    labels = sorted(set(y_true) | set(y_pred))
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    print(f"\nAccuracy:    {accuracy * 100:.2f}%")
    print(f"Macro F1:    {macro_f1 * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0))

    true_counts = Counter(y_true)
    support = {label: true_counts.get(label, 0) for label in labels}
    print("Support by class:")
    for label in labels:
        print(f"  {label:<32} {support[label]}")

    per_batch: dict[str, dict] = {}
    if "source_batch" in test_df.columns:
        frame = test_df[["source_batch", "intent_code"]].copy()
        frame["prediction"] = y_pred
        for batch, group in frame.groupby("source_batch"):
            correct = int((group["intent_code"] == group["prediction"]).sum())
            per_batch[str(batch)] = {
                "rows": int(len(group)),
                "correct": correct,
                "accuracy": round(correct / len(group), 4),
                "true_intent": str(group["intent_code"].mode().iloc[0]),
                "majority_prediction": str(group["prediction"].mode().iloc[0]),
            }
        worst = sorted(per_batch.items(), key=lambda item: item[1]["accuracy"])[:10]
        print("\nLowest-accuracy held-out batches:")
        for batch, info in worst:
            print(
                f"  {batch:<48} {info['accuracy'] * 100:5.1f}%  "
                f"({info['correct']}/{info['rows']})  true={info['true_intent']}  "
                f"pred={info['majority_prediction']}"
            )

    payload = {
        "split": args.split,
        "model": args.model_dir,
        "rows": len(test_df),
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "labels": labels,
        "support": support,
        "per_class": report,
        "per_batch": per_batch,
        "leakage": leakage,
    }
    os.makedirs(os.path.dirname(args.report_out) or ".", exist_ok=True)
    with open(args.report_out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    save_confusion_matrix(labels, matrix, args.matrix_out)

    print(f"\nWrote {args.report_out}")
    print(f"Wrote {args.matrix_out}")


if __name__ == "__main__":
    main()
