Hearing-aid AI dataset — Batch 3

Scope
- Complaint source row 3: "Loud sounds are distorted"
- Topic: Loudness
- Topic ID: 66add6ac8d19f934bd07c673
- Provisional complaint ID: PROV-CMP-0003
- Language: English (general Indian English)
- Rows: 50 (42 natural-form candidates + 8 ASR/noisy-text variants)

Files
- batch3_complaint_english.jsonl: one JSON object per utterance
- batch3_glossary.json: initial language terminology notes
- batch3_self_audit.json: count and validation notes
- batch3_source_mapping.json: source row and ID mapping
- taxonomy_master_provisional.json: current batch mapping only
- dataset_manifest.json: batch metadata

Important limitations
- Provisional complaint IDs must be reconciled against the canonical source taxonomy before production ingestion.
- Synthetic draft data. No native-speaker review, hearing-care domain review, embedding-based near-duplicate analysis, or model evaluation has been performed.
- Naturalness and label-confidence scores are generator estimates, not human ratings.
