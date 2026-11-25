"""Build clean, leak-free train/val/test splits for the Config Assist model.

P0 changes compared with the previous pipeline:

1. Labels come from the *source complaint row* in
   ``Real Data/test.complains.json``, not from a semantic guess made from one
   utterance in the batch.
2. The split is grouped by source batch/folder, so paraphrases of the same
   complaint can never appear in both train and test.
3. Batches listed in ``data_pipeline/quarantine.txt`` (content/source
   mismatch) are excluded by default.
4. Template-regenerated replacement batches in
   ``data_pipeline/replacement_batches.txt`` are kept in train only, so
   validation/test stay on the original batches.  A small
   ``data_pipeline/sharpening_batches.txt`` list is always kept to fix a known
   boundary confusion.
5. The report records class coverage, leakage checks and quarantine decisions.

Run::

    python data_pipeline/clean_data.py
    python data_pipeline/clean_data.py --split-mode ratio
    python data_pipeline/clean_data.py --include-quarantined
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_batches import build_audit, parse_quarantine  # noqa: E402
from source_metadata import (  # noqa: E402
    load_real_data,
    read_utterance_rows,
    utterance_text,
)
from taxonomy import (  # noqa: E402
    GARBAGE_QUESTIONS,
    INTENT_TO_TOPIC,
    final_intent,
    intent_topic,
    normalize_text,
    question_to_intent,
)

DEFAULT_OUT_DIR = os.path.join("data", "processed")
SPLITS = ("train", "val", "test_holdout")


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_master_dataset(
    root: str,
    include_quarantined: bool = False,
    keep_hard_negatives: bool = False,
    min_naturalness: float = 1.0,
) -> tuple[list[dict], dict]:
    audit = build_audit(root)
    metadata = {item["batch_number"]: item for item in audit["batches"]}
    quarantined = set() if include_quarantined else set(audit["quarantined_batches"])
    quarantine_from_file = parse_quarantine(os.path.join(root, "data_pipeline", "quarantine.txt"))

    master: list[dict] = []
    per_batch: dict[int, dict] = {}
    skipped = Counter()
    seen_text: dict[str, str] = {}
    duplicates: list[dict] = []
    unmapped: list[dict] = []

    for number, item in sorted(metadata.items()):
        batch_id = item["batch_id"]
        if number in quarantined:
            skipped["quarantined_batch"] += 1
            continue

        intent = item["intent_code"]
        if not intent:
            skipped["unmapped_intent"] += 1
            unmapped.append(
                {
                    "batch_number": number,
                    "batch_id": batch_id,
                    "source_row": item["source_row"],
                    "question": item["question"],
                }
            )
            continue

        rows = read_utterance_rows(item["utterance_path"])
        kept = 0
        for row in rows:
            if not keep_hard_negatives and (row.get("is_hard_negative") or row.get("safety_flag")):
                skipped["hard_negative_or_safety"] += 1
                continue
            text = utterance_text(row)
            if not text:
                skipped["empty_text"] += 1
                continue
            naturalness = row.get("naturalness_score_1_5")
            try:
                naturalness_value = float(naturalness) if naturalness is not None else None
            except (TypeError, ValueError):
                naturalness_value = None
            if naturalness_value is not None and naturalness_value < min_naturalness:
                skipped["low_naturalness"] += 1
                continue

            key = normalize_text(text)
            if key in seen_text:
                duplicates.append(
                    {
                        "utterance": text,
                        "kept_batch": seen_text[key],
                        "duplicate_batch": batch_id,
                    }
                )
                skipped["exact_duplicate"] += 1
                continue
            seen_text[key] = batch_id

            slots = row.get("slots") if isinstance(row.get("slots"), dict) else {}
            master.append(
                {
                    "utterance": text,
                    "intent_code": intent,
                    "topic": item["topic"],
                    "topic_id": item["real_topic_id"],
                    "persona": row.get("persona", "unspecified"),
                    "situation": slots.get("situation", "unspecified"),
                    "source_batch": batch_id,
                    "batch_number": number,
                    "source_row": item["source_row"],
                    "source_question": item["question"],
                    "raw_intent": item["intent_raw"],
                    "language": row.get("language", "English"),
                    "naturalness_score_1_5": naturalness_value,
                    "label_confidence_1_5": row.get("label_confidence_1_5"),
                }
            )
            kept += 1

        per_batch[number] = {
            "batch_id": batch_id,
            "intent_code": intent,
            "source_row": item["source_row"],
            "source_question": item["question"],
            "rows_in_file": len(rows),
            "rows_kept": kept,
            "content_source_overlap": item.get("content_source_overlap"),
            "flags": item.get("flags", []),
        }

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": os.path.abspath(root),
        "include_quarantined": include_quarantined,
        "keep_hard_negatives": keep_hard_negatives,
        "min_naturalness": min_naturalness,
        "batches": {
            "total": len(metadata),
            "included": len(per_batch),
            "quarantined": sorted(quarantined),
            "quarantine_file_entries": sorted(quarantine_from_file),
            "unmapped": unmapped,
        },
        "rows": {
            "total": len(master),
            "skipped": dict(sorted(skipped.items())),
        },
        "duplicates": {
            "count": len(duplicates),
            "examples": duplicates[:20],
        },
        "per_batch": per_batch,
    }
    return master, report


# ---------------------------------------------------------------------------
# Grouped split
# ---------------------------------------------------------------------------

def apply_replacement_policy(
    rows: list[dict],
    replacement_numbers: set[int],
    always_use: set[int] | None = None,
    keep_eligible: set[int] | None = None,
) -> tuple[list[dict], dict]:
    """Keep a replacement batch when needed, forced, or explicitly eligible.

    Template-regenerated replacements are useful for filling a missing class,
    but they are stylistically unlike the original LLM-authored batches.  When
    an intent already has original data, a replacement is dropped unless it is
    in ``always_use`` (kept and forced to train, for boundary sharpening) or
    ``keep_eligible`` (kept and allowed to compete for validation/test, used
    to give single-batch intents held-out support).
    """

    always_use = set(always_use or ())
    keep_eligible = set(keep_eligible or ())
    intents_with_original = {
        row["intent_code"]
        for row in rows
        if row["batch_number"] not in replacement_numbers
    }

    kept: list[dict] = []
    dropped_batches: set[int] = set()
    used_batches: set[int] = set()
    dropped_rows = 0
    for row in rows:
        number = row["batch_number"]
        if number in replacement_numbers:
            if (
                row["intent_code"] in intents_with_original
                and number not in always_use
                and number not in keep_eligible
            ):
                dropped_batches.add(number)
                dropped_rows += 1
                continue
            used_batches.add(number)
        kept.append(row)

    info = {
        "listed": sorted(replacement_numbers),
        "always_use": sorted(always_use),
        "keep_eligible": sorted(keep_eligible),
        "used": sorted(used_batches),
        "dropped_redundant": sorted(dropped_batches),
        "rows_kept": len(kept),
        "rows_dropped": dropped_rows,
    }
    return kept, info


def grouped_split(
    rows: list[dict],
    seed: int = 42,
    mode: str = "coverage",
    train_only: set[int] | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    """Split rows by ``source_batch`` so no batch appears in two splits.

    ``coverage`` (default) guarantees that every intent with at least two
    batches has a held-out test batch.  ``ratio`` is a plain group shuffle
    targeting roughly 80/10/10; use it when you care more about the ratio than
    per-intent test coverage.

    ``train_only`` lists batch numbers that must stay in train (for example
    template-regenerated replacement batches).  They are removed from the
    holdout pool *before* shuffling, so the split stays reproducible.
    """

    if mode not in {"coverage", "ratio"}:
        raise ValueError(f"Unknown split mode: {mode}")

    train_only = set(train_only or ())

    batches_by_intent: dict[str, list[str]] = defaultdict(list)
    rows_by_batch: dict[str, list[dict]] = defaultdict(list)
    batch_intent: dict[str, str] = {}
    batch_number: dict[str, int] = {}
    for row in rows:
        batch = row["source_batch"]
        rows_by_batch[batch].append(row)
        batch_intent[batch] = row["intent_code"]
        batch_number[batch] = int(row["batch_number"])
    for batch, intent in batch_intent.items():
        batches_by_intent[intent].append(batch)

    rng = random.Random(seed)
    split_batches: dict[str, set[str]] = {split: set() for split in SPLITS}
    warnings: list[str] = []
    forced_train = {
        batch for batch, number in batch_number.items() if number in train_only
    }
    split_batches["train"].update(forced_train)

    if mode == "coverage":
        for intent in sorted(batches_by_intent):
            batches = sorted(set(batches_by_intent[intent]))
            pool = [batch for batch in batches if batch not in forced_train]
            rng.shuffle(pool)
            count = len(pool)
            if count >= 3:
                # One held-out test batch, one validation batch, rest train.
                split_batches["test_holdout"].add(pool[0])
                split_batches["val"].add(pool[1])
                split_batches["train"].update(pool[2:])
            elif count == 2:
                split_batches["test_holdout"].add(pool[0])
                split_batches["train"].add(pool[1])
            elif count == 1:
                split_batches["train"].add(pool[0])
                warnings.append(
                    f"{intent} has only one non-replacement batch ({pool[0]}); "
                    "it can only appear in train."
                )
            else:
                warnings.append(
                    f"{intent} has no non-replacement batch; it can only appear in train."
                )
    else:
        unique = sorted(rows_by_batch)
        pool = [batch for batch in unique if batch not in forced_train]
        rng.shuffle(pool)
        total = len(pool)
        n_test = max(1, int(round(total * 0.10))) if total else 0
        n_val = max(1, int(round(total * 0.10))) if total else 0
        split_batches["test_holdout"].update(pool[:n_test])
        split_batches["val"].update(pool[n_test : n_test + n_val])
        split_batches["train"].update(pool[n_test + n_val :])

    split_rows: dict[str, list[dict]] = {split: [] for split in SPLITS}
    for batch, batch_rows in rows_by_batch.items():
        for split in SPLITS:
            if batch in split_batches[split]:
                split_rows[split].extend(batch_rows)
                break
        else:
            warnings.append(f"Batch {batch} was not assigned to any split.")

    # Every intent must be learnable.
    train_intents = {row["intent_code"] for row in split_rows["train"]}
    for intent in sorted(batches_by_intent):
        if intent not in train_intents:
            warnings.append(f"{intent} is not present in train.")

    for split in SPLITS:
        if not split_rows[split]:
            warnings.append(f"Split {split} is empty.")

    info = {
        "mode": mode,
        "seed": seed,
        "batches": {split: sorted(split_batches[split]) for split in SPLITS},
        "batch_counts": {split: len(split_batches[split]) for split in SPLITS},
        "row_counts": {split: len(split_rows[split]) for split in SPLITS},
        "train_only_batches": sorted(forced_train),
        "warnings": warnings,
    }
    return split_rows, info


# ---------------------------------------------------------------------------
# DSP database for inference
# ---------------------------------------------------------------------------

def build_cleaned_database(root: str) -> tuple[list[dict], dict]:
    """Attach the corrected intent code to every source complaint row."""

    real = load_real_data(root)
    cleaned: list[dict] = []
    counter: Counter[str] = Counter()
    unmapped: list[dict] = []

    for index, record in enumerate(real.complaints, start=1):
        question = str(record.get("question", "")).strip()
        key = normalize_text(question)
        if key in GARBAGE_QUESTIONS:
            counter["garbage_dropped"] += 1
            continue
        raw_intent = question_to_intent(question)
        if raw_intent is None:
            counter["unmapped"] += 1
            unmapped.append({"row": index, "question": question})
            intent = "UNMAPPED"
        else:
            intent = final_intent(raw_intent)
            counter[f"intent:{intent}"] += 1

        enriched = dict(record)
        enriched["intent_code"] = intent
        enriched["raw_intent"] = raw_intent
        enriched["topic"] = intent_topic(intent) if intent != "UNMAPPED" else ""
        cleaned.append(enriched)

    report = {
        "source_rows": len(real.complaints),
        "cleaned_rows": len(cleaned),
        "counts": dict(sorted(counter.items())),
        "unmapped": unmapped,
    }
    return cleaned, report


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_split_report(
    master: list[dict],
    split_rows: dict[str, list[dict]],
    split_info: dict,
    dataset_report: dict,
    db_report: dict,
    root: str,
) -> dict:
    all_batches = sorted({row["source_batch"] for row in master})
    split_batches = {
        split: {row["source_batch"] for row in split_rows[split]} for split in SPLITS
    }

    leakage = {
        "train_test_batch_overlap": sorted(split_batches["train"] & split_batches["test_holdout"]),
        "train_val_batch_overlap": sorted(split_batches["train"] & split_batches["val"]),
        "val_test_batch_overlap": sorted(split_batches["val"] & split_batches["test_holdout"]),
    }

    intents = sorted({row["intent_code"] for row in master})
    coverage: dict[str, dict] = {}
    for intent in intents:
        coverage[intent] = {}
        for split in SPLITS:
            split_batch_ids = {
                row["source_batch"] for row in split_rows[split] if row["intent_code"] == intent
            }
            row_count = sum(1 for row in split_rows[split] if row["intent_code"] == intent)
            coverage[intent][split] = {
                "rows": row_count,
                "batches": sorted(split_batch_ids),
                "batch_count": len(split_batch_ids),
            }

    missing_in_test = sorted(
        intent for intent in intents if coverage[intent]["test_holdout"]["batch_count"] == 0
    )
    missing_in_val = sorted(
        intent for intent in intents if coverage[intent]["val"]["batch_count"] == 0
    )

    warnings = list(split_info["warnings"])
    if missing_in_test:
        warnings.append(
            "Intents without a held-out test batch: " + ", ".join(missing_in_test)
        )
    if missing_in_val:
        warnings.append("Intents without a validation batch: " + ", ".join(missing_in_val))
    for name, overlap in leakage.items():
        if overlap:
            warnings.append(f"Leakage detected ({name}): {overlap}")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": os.path.abspath(root),
        "taxonomy": {
            "intent_count": len(intents),
            "intents": intents,
            "intent_to_topic": {intent: INTENT_TO_TOPIC.get(intent, "Unknown") for intent in intents},
        },
        "dataset": dataset_report,
        "database": db_report,
        "split": split_info,
        "leakage": leakage,
        "class_coverage": coverage,
        "warnings": warnings,
        "totals": {
            "rows": len(master),
            "batches": len(all_batches),
            "intents": len(intents),
            "rows_by_split": split_info["row_counts"],
            "batches_by_split": split_info["batch_counts"],
        },
    }


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: dict | list) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=("coverage", "ratio"),
        default="coverage",
        help="coverage: guarantee a test batch per intent when possible; ratio: plain 80/10/10 group shuffle",
    )
    parser.add_argument(
        "--include-quarantined",
        action="store_true",
        help="Include batches listed in data_pipeline/quarantine.txt",
    )
    parser.add_argument(
        "--keep-hard-negatives",
        action="store_true",
        help="Keep rows flagged is_hard_negative/safety_flag (default: drop)",
    )
    parser.add_argument(
        "--min-naturalness",
        type=float,
        default=1.0,
        help="Drop rows with naturalness_score_1_5 below this value (default: 1.0 = keep all)",
    )
    args = parser.parse_args()

    os.makedirs(os.path.join(args.root, args.out_dir), exist_ok=True)
    out_dir = os.path.join(args.root, args.out_dir)

    print("[1/5] Auditing batches...")
    master, dataset_report = build_master_dataset(
        args.root,
        include_quarantined=args.include_quarantined,
        keep_hard_negatives=args.keep_hard_negatives,
        min_naturalness=args.min_naturalness,
    )
    if not master:
        raise SystemExit("No rows were produced. Check the audit report and quarantine file.")

    replacement_file = os.path.join(args.root, "data_pipeline", "replacement_batches.txt")
    sharpening_file = os.path.join(args.root, "data_pipeline", "sharpening_batches.txt")
    test_support_file = os.path.join(args.root, "data_pipeline", "test_support_batches.txt")
    replacement_numbers = parse_quarantine(replacement_file)
    sharpening_numbers = parse_quarantine(sharpening_file)
    test_support_numbers = parse_quarantine(test_support_file)
    replacement_numbers |= sharpening_numbers
    replacement_numbers |= test_support_numbers
    replacement_info = {
        "listed": [],
        "always_use": [],
        "keep_eligible": [],
        "used": [],
        "dropped_redundant": [],
        "rows_kept": len(master),
        "rows_dropped": 0,
    }
    if replacement_numbers:
        master, replacement_info = apply_replacement_policy(
            master,
            replacement_numbers,
            always_use=sharpening_numbers,
            keep_eligible=test_support_numbers,
        )
        dataset_report["replacements"] = replacement_info
        dataset_report["rows"]["total"] = len(master)
        dataset_report["rows"]["skipped"]["redundant_replacement"] = replacement_info["rows_dropped"]
        dataset_report["batches"]["included"] = len({row["batch_number"] for row in master})
        used_counts = Counter(row["batch_number"] for row in master)
        for number, item in dataset_report["per_batch"].items():
            item["rows_used"] = used_counts.get(number, 0)
            if number in replacement_numbers:
                item["replacement_used"] = number in replacement_info["used"]
        print(
            "      replacements: "
            f"used={replacement_info['used']} "
            f"dropped_redundant={replacement_info['dropped_redundant']}"
        )

    print(f"      {dataset_report['rows']['total']} rows from {dataset_report['batches']['included']} batches")
    print(f"      skipped: {dataset_report['rows']['skipped']}")

    print("[2/5] Building grouped splits...")
    train_only = set(replacement_info["always_use"])
    if train_only:
        print(f"      train-only replacement batches: {sorted(train_only)}")
    split_rows, split_info = grouped_split(
        master, seed=args.seed, mode=args.split_mode, train_only=train_only
    )
    print(f"      rows: {split_info['row_counts']}")
    print(f"      batches: {split_info['batch_counts']}")

    print("[3/5] Building cleaned DSP database...")
    database, db_report = build_cleaned_database(args.root)
    print(f"      {db_report['cleaned_rows']} rows, unmapped: {len(db_report['unmapped'])}")

    print("[4/5] Writing artifacts...")
    for split in SPLITS:
        write_jsonl(os.path.join(out_dir, f"{split}.jsonl"), split_rows[split])
    write_json(os.path.join(out_dir, "complains_cleaned.json"), database)
    write_json(os.path.join(out_dir, "batch_metadata.json"), dataset_report["per_batch"])
    write_json(
        os.path.join(out_dir, "label_map.json"),
        {intent: index for index, intent in enumerate(sorted({row["intent_code"] for row in master}))},
    )

    report = build_split_report(master, split_rows, split_info, dataset_report, db_report, args.root)
    write_json(os.path.join(out_dir, "split_report.json"), report)

    print("[5/5] Done.")
    print(f"      intents: {report['taxonomy']['intent_count']}")
    print(f"      leakage: {report['leakage']}")
    if report["warnings"]:
        print("      warnings:")
        for warning in report["warnings"]:
            print(f"        - {warning}")
    print(f"\nWrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
