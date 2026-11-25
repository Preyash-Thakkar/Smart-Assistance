"""Audit generated batches against the source complaint database.

The batch generator ran over several prompt revisions and, for a handful of
batches, wrote content/glossary text that does not belong to the source row
recorded in ``*_source_mapping.json``.  Training on those rows teaches the
model the wrong concept, so we surface them before building the dataset.

Usage::

    python data_pipeline/audit_batches.py
    python data_pipeline/audit_batches.py --json-out data/processed/batch_audit.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source_metadata import (  # noqa: E402
    build_batch_metadata,
    read_utterance_rows,
    utterance_text,
)

AUDIT_PATH = os.path.join("data", "processed", "batch_audit.json")
QUARANTINE_PATH = os.path.join("data_pipeline", "quarantine.txt")

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "can",
    "could", "do", "does", "for", "from", "has", "have", "i", "if", "in",
    "is", "it", "its", "me", "my", "of", "on", "or", "so", "that", "the",
    "their", "them", "then", "there", "these", "they", "this", "through",
    "to", "too", "very", "was", "we", "when", "with", "would", "you", "your",
    "hearing", "aid", "aids", "device", "sound", "sounds", "feel", "seems",
    "like", "much", "more", "some", "all", "get", "gets", "whenever", "while",
    "wear", "wearing", "use", "using", "also", "just", "really", "still",
    "e", "g", "eg", "etc",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    words = _WORD_RE.findall(str(text).lower())
    return {word for word in words if word not in STOPWORDS and len(word) > 2}


def build_idf(token_sets: list[set[str]]) -> dict[str, float]:
    """Inverse document frequency over the source questions.

    Generic words such as "loud" or "noise" appear in many questions; without
    IDF, every loudness batch would "best match" the generic loudness row.
    """

    document_frequency: Counter[str] = Counter()
    for tokens in token_sets:
        for token in tokens:
            document_frequency[token] += 1
    total = max(1, len(token_sets))
    return {
        token: math.log((1 + total) / (1 + count)) + 1.0
        for token, count in document_frequency.items()
    }


def overlap(source_tokens: set[str], content_tokens: set[str], idf: dict[str, float]) -> float:
    if not source_tokens:
        return 0.0
    total = sum(idf.get(token, 1.0) for token in source_tokens)
    if total == 0:
        return 0.0
    shared = sum(idf.get(token, 1.0) for token in source_tokens & content_tokens)
    return shared / total


def parse_quarantine(path: str = QUARANTINE_PATH) -> set[int]:
    """Read one batch number per line; ``#`` starts a comment."""

    if not os.path.exists(path):
        return set()
    batches: set[int] = set()
    with open(path, "r", encoding="utf-8-sig") as handle:
        for raw_line in handle:
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            match = re.search(r"batch\s*(\d+)", line, re.IGNORECASE) or re.search(r"(\d+)", line)
            if match:
                batches.add(int(match.group(1)))
    return batches


def build_audit(root: str = ".") -> dict:
    metadata = build_batch_metadata(root)
    quarantine = parse_quarantine(os.path.join(root, QUARANTINE_PATH))

    source_tokens = {item.batch_number: tokenize(item.question) for item in metadata}
    idf = build_idf(list(source_tokens.values()))
    content_tokens: dict[int, set[str]] = {}
    utterance_counts: dict[int, int] = {}

    for item in metadata:
        rows = read_utterance_rows(item.utterance_path)
        utterance_counts[item.batch_number] = len(rows)
        texts = [utterance_text(row) for row in rows]
        texts = [text for text in texts if text]
        content_tokens[item.batch_number] = tokenize(" ".join(texts) + " " + " ".join(item.glossary_terms))

    records: list[dict] = []
    flag_counter: Counter[str] = Counter()

    for item in metadata:
        number = item.batch_number
        self_score = overlap(source_tokens[number], content_tokens[number], idf)
        glossary_score = overlap(
            source_tokens[number], tokenize(" ".join(item.glossary_terms)), idf
        )

        best_number = number
        best_score = self_score
        for other in metadata:
            if other.batch_number == number:
                continue
            # A one-token question such as "All loud sounds too loud" reduces
            # to {"loud"} and would match almost everything.  Do not use such
            # a row as a content-alignment reference.
            if len(source_tokens[other.batch_number]) < 2:
                continue
            score = overlap(source_tokens[other.batch_number], content_tokens[number], idf)
            if score > best_score:
                best_score = score
                best_number = other.batch_number

        flags = list(item.flags)
        if source_tokens[number] and self_score == 0.0:
            flags.append("CONTENT_NO_SOURCE_KEYWORDS")
        elif source_tokens[number] and self_score < 0.5:
            flags.append("LOW_CONTENT_OVERLAP")
        if number in quarantine:
            flags.append("QUARANTINED")
        if utterance_counts.get(number, 0) == 0:
            flags.append("NO_UTTERANCES")

        for flag in flags:
            flag_counter[flag] += 1

        record = item.to_dict()
        record.update(
            {
                "utterance_count": utterance_counts.get(number, 0),
                "content_source_overlap": round(self_score, 3),
                "glossary_source_overlap": round(glossary_score, 3),
                "best_content_match_batch": best_number,
                "best_content_match_score": round(best_score, 3),
                "quarantined": number in quarantine,
                "flags": flags,
            }
        )
        records.append(record)

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_count": len(records),
        "quarantine_file": QUARANTINE_PATH,
        "quarantined_batches": sorted(quarantine),
        "summary": {
            "flags": dict(sorted(flag_counter.items())),
            "quarantined_count": len(quarantine),
            "batches_without_source_keywords": sorted(
                r["batch_number"] for r in records if "CONTENT_NO_SOURCE_KEYWORDS" in r["flags"]
            ),
            "batches_with_low_content_overlap": sorted(
                r["batch_number"] for r in records if "LOW_CONTENT_OVERLAP" in r["flags"]
            ),
        },
        "batches": records,
    }


def print_report(audit: dict) -> None:
    print(f"Audited {audit['batch_count']} batches")
    print(f"Quarantined: {audit['quarantined_batches']}")
    print(f"Flags: {audit['summary']['flags']}")
    print(
        "Batches with zero source-keyword overlap: "
        f"{audit['summary']['batches_without_source_keywords']}"
    )
    print(
        "Batches with low (<50%) source-keyword overlap: "
        f"{audit['summary']['batches_with_low_content_overlap']}"
    )
    print()
    print(f"{'batch':>5}  {'src':>3}  {'overlap':>7}  {'best':>4}  {'score':>5}  source / content notes")
    for record in audit["batches"]:
        interesting = (
            record["quarantined"]
            or record["content_source_overlap"] == 0
            or "LOW_CONTENT_OVERLAP" in record["flags"]
        )
        if not interesting:
            continue
        print(
            f"{record['batch_number']:>5}  {str(record['source_row']):>3}  "
            f"{record['content_source_overlap']:>7.2f}  "
            f"{record['best_content_match_batch']:>4}  "
            f"{record['best_content_match_score']:>5.2f}  "
            f"{record['question'][:48]!r} | {', '.join(record['flags'])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root")
    parser.add_argument("--json-out", default=AUDIT_PATH, help="Where to write the JSON report")
    args = parser.parse_args()

    audit = build_audit(args.root)
    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(audit, handle, indent=2)
    print_report(audit)
    if args.json_out:
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
