HEARING-AID AI COMPLAINT DATASET — PROMPT-BASED REGENERATION, BATCH 1

This is the first execution batch under the uploaded master prompt: one complaint x one language, up to 50 rows.

Cell:
- Complaint: All Soft Sound are too Soft (source complaint row 1)
- Topic: Loudness
- Language: English (Indian English/general)
- Rows: 50 (42 natural-form candidates + 8 linked ASR-noise candidates)

Ground-truth topic mapping:
complaints.topic_id.$oid == topics._id.$oid
topic_id is copied unchanged from the complaint record; topic is read from topics.title.

ID limitation:
The source complaint JSON has no individual complaint_id field. PROV-CMP-0001 is a provisional batch identifier, not an official ID. Replace/map it to the official complaint ID when the taxonomy owner supplies one.

Files:
- batch1_complaint_english.jsonl: 50 JSONL utterances using the prompt schema.
- batch1_glossary.json: first-pass glossary for this language cell.
- batch1_source_mapping.json: exact source complaint and matching topic record.
- taxonomy_master_provisional.json: source topics and the mapped complaint definition.
- batch1_self_audit.json: count, dedup check, and coverage limitations.
- dataset_manifest.json: file list, sizes, and SHA-256 hashes.

All utterances are synthetic and unreviewed. Normalized exact duplicates were checked; embedding-based similarity, native-speaker review, domain review, and model evaluation were not performed. This batch is not production-ready.
