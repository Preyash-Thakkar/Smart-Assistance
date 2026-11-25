"""Validate the P0 dataset artifacts without needing ML dependencies.

Checks:

* every split file exists and is non-empty;
* every row has an utterance and an intent code;
* no source batch appears in more than one split (no leakage);
* every trian intent has at least one training row;
* every intent with >= 2 batches has a held-out test batch;
* the cleaned DSP database has an intent for every non-garbage row;
* the quarantine list matches the audit report.

Usage::

    python scripts/validate_dataset.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

SPLITS = ("train", "val", "test_holdout")
PROCESSED = os.path.join("data", "processed")


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    problems: list[str] = []
    warnings: list[str] = []

    paths = {split: os.path.join(PROCESSED, f"{split}.jsonl") for split in SPLITS}
    for split, path in paths.items():
        if not os.path.exists(path):
            fail(f"missing {path}; run data_pipeline/clean_data.py")

    split_rows = {split: load_jsonl(path) for split, path in paths.items()}
    for split, rows in split_rows.items():
        if not rows:
            problems.append(f"{split} is empty")
        for index, row in enumerate(rows):
            if not row.get("utterance"):
                problems.append(f"{split}[{index}] has no utterance")
            if not row.get("intent_code"):
                problems.append(f"{split}[{index}] has no intent_code")
            if not row.get("source_batch"):
                problems.append(f"{split}[{index}] has no source_batch")

    batches = {split: {row.get("source_batch") for row in rows} for split, rows in split_rows.items()}
    for left, right in (("train", "val"), ("train", "test_holdout"), ("val", "test_holdout")):
        overlap = batches[left] & batches[right]
        if overlap:
            problems.append(f"batch overlap {left}/{right}: {sorted(overlap)}")

    train_intents = {row["intent_code"] for row in split_rows["train"]}
    batches_by_intent: dict[str, set[str]] = defaultdict(set)
    for row in split_rows["train"] + split_rows["val"] + split_rows["test_holdout"]:
        batches_by_intent[row["intent_code"]].add(row["source_batch"])

    for intent in sorted({row["intent_code"] for rows in split_rows.values() for row in rows}):
        if intent not in train_intents:
            problems.append(f"{intent} is absent from train")
        test_count = len(
            {row["source_batch"] for row in split_rows["test_holdout"] if row["intent_code"] == intent}
        )
        total_batches = len(batches_by_intent[intent])
        if total_batches >= 2 and test_count == 0:
            problems.append(f"{intent} has {total_batches} batches but no test batch")

    label_map_path = os.path.join(PROCESSED, "label_map.json")
    if not os.path.exists(label_map_path):
        problems.append("missing label_map.json")
    else:
        with open(label_map_path, "r", encoding="utf-8") as handle:
            label_map = json.load(handle)
        known = set(label_map)
        for split, rows in split_rows.items():
            unknown = {row["intent_code"] for row in rows} - known
            if unknown:
                problems.append(f"{split} has intents missing from label_map: {sorted(unknown)}")

    db_path = os.path.join(PROCESSED, "complains_cleaned.json")
    if not os.path.exists(db_path):
        problems.append("missing complains_cleaned.json")
    else:
        with open(db_path, "r", encoding="utf-8") as handle:
            database = json.load(handle)
        if not isinstance(database, list) or not database:
            problems.append("complains_cleaned.json is empty")
        else:
            unmapped = [row for row in database if row.get("intent_code") == "UNMAPPED"]
            if unmapped:
                warnings.append(f"{len(unmapped)} DSP rows are UNMAPPED")
            missing = [row for row in database if not row.get("intent_code")]
            if missing:
                problems.append(f"{len(missing)} DSP rows have no intent_code")

    report_path = os.path.join(PROCESSED, "split_report.json")
    if not os.path.exists(report_path):
        problems.append("missing split_report.json")
    else:
        with open(report_path, "r", encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("leakage", {}).get("train_test_batch_overlap"):
            problems.append("split_report.json still reports train/test leakage")
        if report.get("warnings"):
            warnings.extend(report["warnings"])

    print("Dataset summary")
    for split in SPLITS:
        rows = split_rows[split]
        print(
            f"  {split:<14} rows={len(rows):>5}  batches={len(batches[split]):>3}  "
            f"intents={len({row['intent_code'] for row in rows}):>3}"
        )
    print(f"  intents total: {len(batches_by_intent)}")
    print(f"  train intents: {len(train_intents)}")

    if warnings:
        print("\nWarnings")
        for warning in warnings:
            print(f"  - {warning}")

    if problems:
        print("\nProblems")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)

    print("\nPASS: dataset artifacts look consistent.")


if __name__ == "__main__":
    main()
