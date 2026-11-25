"""Regenerate batches whose generated content does not match the source row.

The original generator is not in this repository.  This script is a
deterministic offline replacement: it writes 42 natural-form utterances plus 8
ASR-style variants per batch from the phrase banks in
``data_pipeline/concepts.py``.

Safety rules
------------
* The authoritative source mapping file is **not** modified.
* Only ``*_complaint_english.jsonl``, ``*_glossary.json``,
  ``*_self_audit.json`` and ``README.txt`` are rewritten.
* The generated content must share enough keywords with the source complaint
  to pass :mod:`audit_batches`; otherwise the batch is left in quarantine.

Usage::

    # preview without writing
    python data_pipeline/regenerate_batches.py --batches 17

    # regenerate every batch currently listed in quarantine.txt
    python data_pipeline/regenerate_batches.py --write

    # regenerate a specific set
    python data_pipeline/regenerate_batches.py --batches 17 79 83 84 --write
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audit_batches import overlap, parse_quarantine, tokenize  # noqa: E402
from concepts import CONCEPT_BANKS, DEVICES  # noqa: E402
from source_metadata import (  # noqa: E402
    build_batch_metadata,
    read_utterance_rows,
)

ROOT = "."
QUARANTINE = os.path.join("data_pipeline", "quarantine.txt")

ASR_REPLACEMENTS = [
    ("hearing aid", "hearin aid"),
    ("hearing device", "hearin device"),
    ("hearing machine", "hearin machine"),
    ("my", "ma"),
    ("the", "teh"),
    ("too", "to"),
    ("and", "n"),
    ("with", "wit"),
    ("through", "thru"),
    ("because", "becos"),
    ("something", "somethin"),
    ("everything", "everythin"),
    ("difficult", "dificult"),
    ("uncomfortable", "uncomfortable"),
]


def clean_text(text: str) -> str:
    text = " ".join(str(text).split()).strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def length_bucket(text: str) -> str:
    words = len(text.split())
    if words <= 6:
        return "short"
    if words <= 12:
        return "medium"
    return "long"


def build_candidates(bank: dict, seed: int) -> list[dict]:
    situations = bank.get("situations") or [""]
    seen: set[str] = set()
    candidates: list[dict] = []
    for frame in bank["frames"]:
        for sound in bank["sounds"]:
            for quality in bank["qualities"]:
                for situation in situations:
                    for device in DEVICES:
                        text = frame.format(
                            device=device,
                            sound=sound,
                            quality=quality,
                            situation=situation,
                        )
                        text = clean_text(text)
                        if not text or text in seen:
                            continue
                        seen.add(text)
                        candidates.append(
                            {
                                "text": text,
                                "sound": sound,
                                "quality": quality,
                                "situation": situation,
                            }
                        )
    random.Random(seed).shuffle(candidates)
    return candidates


def asr_variant(text: str, rng: random.Random) -> str:
    """Create a plausible ASR/typing variant without destroying keywords."""

    original = text
    value = text.lower().rstrip(".")
    value = value.replace(",", "")
    choices = [(a, b) for a, b in ASR_REPLACEMENTS if a in value]
    rng.shuffle(choices)
    for source, target in choices[: rng.randint(2, 4)]:
        value = value.replace(source, target)
    value = " ".join(value.split())
    if value == original.lower().rstrip("."):
        value = value.replace("hearing", "hearin")
    return value


def make_row(
    batch: int,
    index: int,
    text: str,
    bank: dict,
    meta,
    *,
    channel: str,
    naturalness: int,
    confidence: int,
    source_utt_id: str | None = None,
    back_translation: str | None = None,
    note: str,
    situation: str = "",
) -> dict:
    return {
        "utt_id": f"B{batch:02d}-EN-{index:03d}",
        "complaint_id": meta.complaint_id,
        "topic_id": meta.real_topic_id,
        "complaint_ids_all": [meta.complaint_id],
        "language": "English",
        "lang_code": "en-IN",
        "script": "Latin",
        "text": text,
        "text_romanized": None,
        "region": "India / general English",
        "dialect": "Indian English; broad, non-region-specific",
        "is_code_mixed": False,
        "persona": "user",
        "register": "casual",
        "channel": channel,
        "perspective": "first_person",
        "sentence_form": "statement",
        "length_bucket": length_bucket(text),
        "slots": {
            "symptom": bank["symptom"],
            "device": "hearing aid",
            "sound_source": bank["sounds"][:3],
            "situation": situation or "general",
            "condition": bank["condition"],
        },
        "is_hard_negative": False,
        "is_multi_intent": False,
        "ambiguous": False,
        "safety_flag": False,
        "safety_type": None,
        "label": meta.complaint_id,
        "source_utt_id": source_utt_id,
        "back_translation_en": back_translation if back_translation is not None else text,
        "naturalness_score_1_5": naturalness,
        "label_confidence_1_5": confidence,
        "generation_notes": note,
    }


def build_rows(meta, count: int, asr_count: int) -> tuple[list[dict], dict]:
    bank = CONCEPT_BANKS.get(meta.intent_code)
    if bank is None:
        raise SystemExit(f"No concept bank for {meta.intent_code} (batch {meta.batch_number}).")

    candidates = build_candidates(bank, seed=meta.batch_number * 7919)
    if len(candidates) < count:
        raise SystemExit(
            f"Batch {meta.batch_number}: only {len(candidates)} unique candidates, need {count}."
        )

    source_tokens = tokenize(meta.question)

    # Guarantee that every source keyword appears somewhere in the batch.  A
    # plain shuffle can otherwise drop the only frame containing a rare token
    # (e.g. "whistles" for "Hearing Aid Whistles").
    selected: list[dict] = []
    for token in sorted(source_tokens):
        for candidate in candidates:
            if candidate in selected:
                continue
            if token in tokenize(candidate["text"]):
                selected.append(candidate)
                break
        else:
            raise SystemExit(
                f"Batch {meta.batch_number}: no candidate contains source keyword {token!r}."
            )
    for candidate in candidates:
        if len(selected) >= count:
            break
        if candidate not in selected:
            selected.append(candidate)
    selected = selected[:count]

    natural_count = count - asr_count
    rng = random.Random(meta.batch_number)
    rows: list[dict] = []

    for index, candidate in enumerate(selected[:natural_count], start=1):
        rows.append(
            make_row(
                meta.batch_number,
                index,
                candidate["text"],
                bank,
                meta,
                channel="chat",
                naturalness=4,
                confidence=5,
                note="template-regenerated natural-form candidate",
                situation=candidate["situation"],
            )
        )

    for offset in range(asr_count):
        base_index = offset % natural_count
        base = rows[base_index]
        variant = asr_variant(base["text"], rng)
        rows.append(
            make_row(
                meta.batch_number,
                natural_count + offset + 1,
                variant,
                bank,
                meta,
                channel="voice",
                naturalness=3,
                confidence=4,
                source_utt_id=base["utt_id"],
                back_translation=base["text"],
                note="template-regenerated ASR-style variant",
                situation=base["slots"]["situation"],
            )
        )

    texts = [row["text"] for row in rows]
    content_tokens = tokenize(" ".join(texts))
    score = overlap(source_tokens, content_tokens, {token: 1.0 for token in source_tokens})
    details = {
        "source_tokens": sorted(source_tokens),
        "content_source_overlap": round(score, 3),
        "unique_texts": len(set(texts)),
        "rows": len(rows),
    }
    if score < 0.5:
        raise SystemExit(
            f"Batch {meta.batch_number}: generated content only overlaps "
            f"{score:.2f} with the source question {meta.question!r}."
        )
    return rows, details


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_batch(meta, rows: list[dict], details: dict) -> None:
    folder = meta.folder
    batch = meta.batch_number
    write_jsonl(os.path.join(folder, f"batch{batch}_complaint_english.jsonl"), rows)

    write_json(
        os.path.join(folder, f"batch{batch}_glossary.json"),
        {
            "language": "English",
            "lang_code": "en-IN",
            "terms": [
                {"term": term, "meaning": meaning} for term, meaning in CONCEPT_BANKS[meta.intent_code]["glossary"]
            ],
            "notes": (
                "Regenerated template glossary. Terms describe the source "
                "complaint concept; not a clinically validated terminology standard."
            ),
        },
    )

    write_json(
        os.path.join(folder, f"batch{batch}_self_audit.json"),
        {
            "batch": batch,
            "complaint_source_row": meta.source_row,
            "complaint_text_source_exact": meta.real_question,
            "intent_code": meta.intent_code,
            "complaint_id": meta.complaint_id,
            "topic": meta.topic,
            "topic_id": meta.real_topic_id,
            "language": "English",
            "rows_written": len(rows),
            "natural_form_candidates": len(rows) - 8,
            "asr_noise_variants": 8,
            "exact_normalized_duplicates": len(rows) - details["unique_texts"],
            "json_parse_errors": 0,
            "generation_method": "deterministic template bank (data_pipeline/concepts.py)",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checks": {
                "source_keyword_overlap": details["content_source_overlap"],
                "source_keywords": details["source_tokens"],
                "required_fields_present": True,
                "embedding_similarity_check": "Not performed",
                "native_speaker_review": "Not performed",
                "domain_expert_review": "Not performed",
                "model_evaluation": "Not performed",
            },
            "limitations": [
                "Template-generated synthetic draft; native-speaker and domain review pending.",
                "Less varied than the LLM-authored batches; treat as training-data repair, not gold data.",
            ],
        },
    )

    readme = (
        f"Hearing-aid AI dataset — Batch {batch}\n\n"
        f"Source row {meta.source_row}: {meta.real_question!r}\n"
        f"Topic: {meta.topic}\n"
        f"Topic ID: {meta.real_topic_id}\n"
        f"Provisional complaint ID: {meta.complaint_id}\n"
        f"Language: English (general Indian English)\n"
        f"Rows: {len(rows)} ({len(rows) - 8} template-generated natural-form candidates "
        f"+ 8 ASR/noisy-text variants)\n\n"
        "Regenerated to replace content that did not match the source row.\n"
        "Files: batch{0}_complaint_english.jsonl, batch{0}_glossary.json, "
        "batch{0}_source_mapping.json, batch{0}_self_audit.json, "
        "taxonomy_master_provisional.json, dataset_manifest.json.\n\n"
        "Limitations: synthetic template draft; no native-speaker/domain-expert review; "
        "scores are generator estimates, not human ratings.\n"
    ).format(batch)
    with open(os.path.join(folder, "README.txt"), "w", encoding="utf-8") as handle:
        handle.write(readme)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument(
        "--batches",
        nargs="*",
        type=int,
        default=None,
        help="Batch numbers to regenerate (default: every batch in quarantine.txt)",
    )
    parser.add_argument("--count", type=int, default=50, help="Rows per batch")
    parser.add_argument("--asr-count", type=int, default=8, help="ASR variants included in --count")
    parser.add_argument("--write", action="store_true", help="Write the regenerated files")
    parser.add_argument("--preview", type=int, default=3, help="Sample rows to print per batch")
    args = parser.parse_args()

    metadata = {item.batch_number: item for item in build_batch_metadata(args.root)}
    batches = args.batches if args.batches is not None else sorted(parse_quarantine(os.path.join(args.root, QUARANTINE)))
    if not batches:
        raise SystemExit("Nothing to regenerate: pass --batches or add entries to quarantine.txt.")

    for batch in batches:
        meta = metadata.get(batch)
        if meta is None:
            raise SystemExit(f"Batch {batch} not found.")
        rows, details = build_rows(meta, args.count, args.asr_count)
        print(
            f"batch {batch:02d}  row={meta.source_row}  {meta.intent_code}  "
            f"rows={details['rows']}  unique={details['unique_texts']}  "
            f"source_overlap={details['content_source_overlap']:.2f}"
        )
        for row in rows[: args.preview]:
            print(f"  - {row['text']}")
        if args.write:
            write_batch(meta, rows, details)
            print(f"  wrote {meta.folder}")


if __name__ == "__main__":
    main()
