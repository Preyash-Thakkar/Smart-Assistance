"""Optional semantic audit for the processed splits.

**This script no longer assigns training labels.**

The previous version of ``auto_mapper.py`` picked one intent per batch from a
single utterance and overwrote ``intent_code`` for every row in that batch.
That guessed label disagreed with the source complaint row for 39 of 84
batches.  Labels now come from ``clean_data.py`` (source row -> canonical
question -> intent).

This script is kept as a *sanity check*: it compares each utterance with the
intent descriptions and writes ``semantic_intent_guess`` / ``semantic_score``
to a separate audit file.  It never edits ``train.jsonl``, ``val.jsonl`` or
``test_holdout.jsonl``.

Usage::

    pip install sentence-transformers torch
    python data_pipeline/auto_mapper.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from taxonomy import INTENT_DESCRIPTIONS  # noqa: E402

SPLITS = ("train", "val", "test_holdout")
DEFAULT_AUDIT = os.path.join("data", "processed", "semantic_audit.json")
DEFAULT_ROWS = os.path.join("data", "processed", "semantic_audit.jsonl")


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--split-dir", default=os.path.join("data", "processed"))
    parser.add_argument("--audit-out", default=DEFAULT_AUDIT)
    parser.add_argument("--rows-out", default=DEFAULT_ROWS)
    parser.add_argument("--model", default="all-MiniLM-L6-v2")
    args = parser.parse_args()

    try:
        import torch
        from sentence_transformers import SentenceTransformer, util
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers/torch are not installed. "
            "This audit is optional; install them with:\n"
            "  pip install sentence-transformers torch\n"
            f"Original error: {exc}"
        )

    split_dir = os.path.join(args.root, args.split_dir)
    rows: list[dict] = []
    for split in SPLITS:
        for row in load_jsonl(os.path.join(split_dir, f"{split}.jsonl")):
            row["_split"] = split
            rows.append(row)
    if not rows:
        raise SystemExit(f"No rows found under {split_dir}. Run clean_data.py first.")

    print(f"Loading {args.model} ...")
    model = SentenceTransformer(args.model)
    intent_keys = list(INTENT_DESCRIPTIONS)
    intent_embeddings = model.encode(
        [INTENT_DESCRIPTIONS[key] for key in intent_keys], convert_to_tensor=True
    )

    audit_rows: list[dict] = []
    batch_agreement: dict[str, Counter] = defaultdict(Counter)
    intent_confusion: dict[str, Counter] = defaultdict(Counter)

    for index, row in enumerate(rows, start=1):
        if index % 500 == 0:
            print(f"  {index}/{len(rows)}")
        text = row["utterance"]
        embedding = model.encode(text, convert_to_tensor=True)
        scores = util.cos_sim(embedding, intent_embeddings)[0]
        best_index = int(torch.argmax(scores).item())
        guess = intent_keys[best_index]
        score = float(scores[best_index].item())
        label = row["intent_code"]

        batch = row["source_batch"]
        batch_agreement[batch][guess] += 1
        intent_confusion[label][guess] += 1
        audit_rows.append(
            {
                "utterance": text,
                "intent_code": label,
                "semantic_intent_guess": guess,
                "semantic_score": round(score, 4),
                "agrees": guess == label,
                "source_batch": batch,
                "split": row["_split"],
            }
        )

    total = len(audit_rows)
    agree = sum(1 for row in audit_rows if row["agrees"])
    print(f"\nSemantic agreement with source labels: {agree}/{total} ({agree / total * 100:.1f}%)")

    per_batch = {}
    for batch, counter in sorted(batch_agreement.items()):
        majority, count = counter.most_common(1)[0]
        per_batch[batch] = {
            "rows": sum(counter.values()),
            "majority_guess": majority,
            "majority_count": count,
            "guesses": dict(counter),
        }

    report = {
        "model": args.model,
        "rows": total,
        "agreement": {
            "agree": agree,
            "total": total,
            "percent": round(agree / total * 100, 2),
        },
        "per_intent": {
            intent: dict(counter.most_common())
            for intent, counter in sorted(intent_confusion.items())
        },
        "per_batch": per_batch,
    }

    with open(args.audit_out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with open(args.rows_out, "w", encoding="utf-8") as handle:
        for row in audit_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {args.audit_out}")
    print(f"Wrote {args.rows_out}")


if __name__ == "__main__":
    main()
