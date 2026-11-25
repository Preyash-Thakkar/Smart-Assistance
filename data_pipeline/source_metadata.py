"""Robust readers for the source complaint database and the generated batches.

The batch folders were produced over several prompt revisions, so their
metadata files do **not** share a schema.  For example the source row lives in
``source_row_1_indexed``, ``source_row_number_1_indexed``, ``batch`` or
``source_complaint_row`` depending on the batch, and the complaint text lives
in ``source_question``, ``source_question_exact``, ``complaint_text_exact``,
``complaint_text_exact_as_read`` or ``source_complaint_record.question``.

This module hides that mess behind :func:`build_batch_metadata`.  The
authoritative complaint text is always taken from the matching row in
``Real Data/test.complains.json`` when a row number is available.
"""

from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from taxonomy import (
    final_intent,
    intent_topic,
    normalize_text,
    question_to_intent,
)

# ---------------------------------------------------------------------------
# Paths (relative to the repository root)
# ---------------------------------------------------------------------------

REAL_COMPLAINTS = os.path.join("Real Data", "test.complains.json")
REAL_TOPICS = os.path.join("Real Data", "test.topics.json")

MAPPING_GLOB = os.path.join("Config Assist Multiples", "**", "batch*_source_mapping.json")
UTTERANCE_GLOB = os.path.join("Config Assist Multiples", "**", "batch*_complaint_english.json*")
GLOSSARY_GLOB = os.path.join("Config Assist Multiples", "**", "batch*_glossary.json")

_BATCH_RE = re.compile(r"batch(\d+)", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Generic JSON loading
# ---------------------------------------------------------------------------

def load_json_values(path: str) -> list[Any]:
    """Load JSON, JSONL, or concatenated pretty-printed JSON objects.

    ``batch1_complaint_english.json`` is actually 50 pretty-printed objects
    concatenated together, so the usual ``json.load`` / line-by-line readers
    both fail.  ``json.JSONDecoder.raw_decode`` handles all three shapes.
    """

    with open(path, "r", encoding="utf-8-sig") as handle:
        text = handle.read().strip()
    if not text:
        return []

    # Fast path: a single valid document (array or object).
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if parsed is not None:
        return _flatten(parsed)

    decoder = json.JSONDecoder()
    values: list[Any] = []
    index = 0
    length = len(text)
    while index < length:
        while index < length and text[index] in " \t\r\n":
            index += 1
        if index >= length:
            break
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            newline = text.find("\n", index)
            if newline == -1:
                break
            index = newline + 1
            continue
        values.extend(_flatten(value))
        index = end
    return values


def _flatten(value: Any) -> list[Any]:
    if isinstance(value, list):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _load_first(path: str | None) -> dict[str, Any]:
    if not path or not os.path.exists(path):
        return {}
    for value in load_json_values(path):
        if isinstance(value, dict):
            return value
    return {}


# ---------------------------------------------------------------------------
# Source database
# ---------------------------------------------------------------------------

@dataclass
class RealData:
    complaints: list[dict[str, Any]]
    topics: list[dict[str, Any]]
    row_to_complaint: dict[int, dict[str, Any]]
    topic_titles: dict[str, str]

    def complaint_for_row(self, row: int | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return self.row_to_complaint.get(int(row))


def load_real_data(root: str = ".") -> RealData:
    complaints = [v for v in load_json_values(os.path.join(root, REAL_COMPLAINTS)) if isinstance(v, dict)]
    topics = [v for v in load_json_values(os.path.join(root, REAL_TOPICS)) if isinstance(v, dict)]
    topic_titles = {oid(t.get("_id")): str(t.get("title", "")) for t in topics if oid(t.get("_id"))}
    return RealData(
        complaints=complaints,
        topics=topics,
        row_to_complaint={i: record for i, record in enumerate(complaints, start=1)},
        topic_titles=topic_titles,
    )


def oid(value: Any) -> str | None:
    """Return a Mongo ``$oid`` / plain string as a string."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("$oid", "oid", "_id"):
            if key in value:
                return oid(value[key])
    return str(value)


# ---------------------------------------------------------------------------
# Batch discovery
# ---------------------------------------------------------------------------

@dataclass
class BatchFiles:
    batch_number: int
    folder: str
    mapping_path: str
    utterance_path: str
    glossary_path: str | None = None


@dataclass
class BatchMetadata:
    batch_number: int
    batch_id: str
    folder: str
    mapping_path: str
    utterance_path: str
    glossary_path: str | None

    source_row: int | None
    mapping_question: str
    real_question: str
    question: str
    question_source: str
    question_match: bool | None

    real_topic_id: str | None
    mapping_topic_id: str | None
    topic_title: str
    topic_match: bool | None

    intent_raw: str | None
    intent_code: str | None
    topic: str

    glossary_terms: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_number": self.batch_number,
            "batch_id": self.batch_id,
            "folder": self.folder,
            "mapping_path": self.mapping_path,
            "utterance_path": self.utterance_path,
            "glossary_path": self.glossary_path,
            "source_row": self.source_row,
            "mapping_question": self.mapping_question,
            "real_question": self.real_question,
            "question": self.question,
            "question_source": self.question_source,
            "question_match": self.question_match,
            "real_topic_id": self.real_topic_id,
            "mapping_topic_id": self.mapping_topic_id,
            "topic_title": self.topic_title,
            "topic_match": self.topic_match,
            "intent_raw": self.intent_raw,
            "intent_code": self.intent_code,
            "topic": self.topic,
            "glossary_terms": self.glossary_terms,
            "flags": self.flags,
        }


def _batch_number(path: str) -> int | None:
    match = _BATCH_RE.search(os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def _choose(paths: Iterable[str], prefer_suffix: str | None = None) -> str:
    paths = list(paths)
    if prefer_suffix:
        preferred = [p for p in paths if p.endswith(prefer_suffix)]
        if preferred:
            paths = preferred
    # Prefer the shallowest path, then a stable lexical order.
    return sorted(paths, key=lambda p: (p.count(os.sep), len(p), p))[0]


def discover_batches(root: str = ".") -> dict[int, BatchFiles]:
    """Find every batch and pick one mapping / utterance / glossary file."""

    mappings: dict[int, list[str]] = {}
    utterances: dict[int, list[str]] = {}
    glossaries: dict[int, list[str]] = {}

    for path in glob.glob(os.path.join(root, MAPPING_GLOB), recursive=True):
        number = _batch_number(path)
        if number is not None:
            mappings.setdefault(number, []).append(path)
    for path in glob.glob(os.path.join(root, UTTERANCE_GLOB), recursive=True):
        number = _batch_number(path)
        if number is not None:
            utterances.setdefault(number, []).append(path)
    for path in glob.glob(os.path.join(root, GLOSSARY_GLOB), recursive=True):
        number = _batch_number(path)
        if number is not None:
            glossaries.setdefault(number, []).append(path)

    batches: dict[int, BatchFiles] = {}
    for number, mapping_paths in sorted(mappings.items()):
        mapping_path = _choose(mapping_paths)
        folder = os.path.dirname(mapping_path)
        # Prefer an utterance in the same folder; fall back to any copy.
        same_folder = [p for p in utterances.get(number, []) if os.path.dirname(p) == folder]
        utterance_path = _choose(same_folder or utterances.get(number, []), prefer_suffix=".jsonl")
        if not utterance_path:
            continue
        glossary_paths = glossaries.get(number, [])
        same_folder_glossary = [p for p in glossary_paths if os.path.dirname(p) == folder]
        glossary_path = _choose(same_folder_glossary or glossary_paths) if glossary_paths else None
        batches[number] = BatchFiles(
            batch_number=number,
            folder=folder,
            mapping_path=mapping_path,
            utterance_path=utterance_path,
            glossary_path=glossary_path,
        )
    return batches


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

def extract_row_number(mapping: dict[str, Any], fallback: int | None = None) -> int | None:
    for key in (
        "source_row_1_indexed",
        "source_row_number_1_indexed",
        "source_complaint_row",
        "source_row",
        "source_row_1indexed",
        "row",
        "batch",
    ):
        value = mapping.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return fallback


def extract_mapping_question(mapping: dict[str, Any], glossary: dict[str, Any] | None = None) -> str:
    keys = (
        "source_question",
        "source_question_exact",
        "complaint_text_exact",
        "complaint_text_exact_as_read",
        "source_complaint",
        "complaint",
        "question",
    )
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    record = mapping.get("source_complaint_record")
    if isinstance(record, dict):
        value = record.get("question")
        if isinstance(value, str) and value.strip():
            return value.strip()
    if isinstance(glossary, dict):
        for key in ("source_complaint", "complaint", "question"):
            value = glossary.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def extract_topic_id(mapping: dict[str, Any]) -> str | None:
    for key in (
        "source_topic_id_exact",
        "topic_id_exact",
        "source_topic_id",
        "topic_id",
    ):
        value = oid(mapping.get(key))
        if value:
            return value
    record = mapping.get("source_complaint_record")
    if isinstance(record, dict):
        value = oid(record.get("topic_id"))
        if value:
            return value
    return None


def extract_glossary_terms(glossary: dict[str, Any] | None) -> list[str]:
    if not isinstance(glossary, dict):
        return []
    terms = glossary.get("terms")
    out: list[str] = []
    if isinstance(terms, dict):
        for key, value in terms.items():
            out.append(str(key))
            if isinstance(value, str):
                out.append(value)
    elif isinstance(terms, list):
        for item in terms:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("concept", "term"):
                    value = item.get(key)
                    if isinstance(value, str):
                        out.append(value)
                preferred = item.get("preferred_terms")
                if isinstance(preferred, list):
                    out.extend(str(v) for v in preferred)
    return [term.strip() for term in out if term and term.strip()]


def build_batch_metadata(root: str = ".") -> list[BatchMetadata]:
    real = load_real_data(root)
    batches: list[BatchMetadata] = []

    for number, files in sorted(discover_batches(root).items()):
        mapping = _load_first(files.mapping_path)
        glossary = _load_first(files.glossary_path)

        source_row = extract_row_number(mapping, fallback=number)
        real_record = real.complaint_for_row(source_row)
        real_question = str(real_record.get("question", "")).strip() if real_record else ""
        mapping_question = extract_mapping_question(mapping, glossary)
        authoritative_question = real_question or mapping_question
        question_source = "real_row" if real_question else "mapping"

        question_match: bool | None = None
        if mapping_question and real_question:
            question_match = normalize_text(mapping_question) == normalize_text(real_question)

        real_topic_id = oid(real_record.get("topic_id")) if real_record else None
        mapping_topic_id = extract_topic_id(mapping)
        topic_match: bool | None = None
        if real_topic_id and mapping_topic_id:
            topic_match = real_topic_id == mapping_topic_id

        raw_intent = question_to_intent(authoritative_question)
        intent = final_intent(raw_intent) if raw_intent else None

        flags: list[str] = []
        if source_row is None:
            flags.append("MISSING_SOURCE_ROW")
        elif real_record is None:
            flags.append("SOURCE_ROW_OUT_OF_RANGE")
        if not authoritative_question:
            flags.append("MISSING_SOURCE_QUESTION")
        if intent is None:
            flags.append("UNMAPPED_INTENT")
        if question_match is False:
            flags.append("MAPPING_QUESTION_MISMATCH")
        if topic_match is False:
            flags.append("MAPPING_TOPIC_MISMATCH")

        batches.append(
            BatchMetadata(
                batch_number=number,
                batch_id=os.path.basename(files.folder),
                folder=files.folder,
                mapping_path=files.mapping_path,
                utterance_path=files.utterance_path,
                glossary_path=files.glossary_path,
                source_row=source_row,
                mapping_question=mapping_question,
                real_question=real_question,
                question=authoritative_question,
                question_source=question_source,
                question_match=question_match,
                real_topic_id=real_topic_id,
                mapping_topic_id=mapping_topic_id,
                topic_title=real.topic_titles.get(real_topic_id or "", ""),
                topic_match=topic_match,
                intent_raw=raw_intent,
                intent_code=intent,
                topic=intent_topic(intent) if intent else "Unknown",
                glossary_terms=extract_glossary_terms(glossary),
                flags=flags,
            )
        )
    return batches


# ---------------------------------------------------------------------------
# Utterances
# ---------------------------------------------------------------------------

def read_utterance_rows(path: str) -> list[dict[str, Any]]:
    """Return the utterance records from a batch file (any supported schema)."""

    return [value for value in load_json_values(path) if isinstance(value, dict)]


def utterance_text(row: dict[str, Any]) -> str:
    for key in ("text", "text_romanized", "utterance"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""
