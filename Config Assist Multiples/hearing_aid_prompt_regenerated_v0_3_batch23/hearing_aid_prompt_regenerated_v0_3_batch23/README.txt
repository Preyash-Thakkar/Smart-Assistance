Hearing Aid User Assistant Dataset — Batch 23

Source complaint (row 23, 1-indexed):  Crowd noise is loud
Topic: Noise
Topic ID: 66b060d95c6674686e02d026
Provisional complaint ID: PROV-CMP-0023

Contents
- batch23_complaint_english.jsonl: 50 utterance records (42 natural-form candidates + 8 synthetic ASR/noisy-text candidates)
- batch23_glossary.json: key semantic terms
- batch23_source_mapping.json: traceability to source row and taxonomy
- batch23_self_audit.json: automated checks and limitations
- taxonomy_master_provisional.json: provisional mapping for this batch
- dataset_manifest.json: batch metadata

Important limitations
- Source complaint JSON has no dedicated complaint ID field; PROV-CMP-0023 is provisional and must be reconciled with the canonical taxonomy.
- All utterances and scores are synthetic drafts. Naturalness/confidence scores are estimates, not human ratings.
- Native-speaker/domain review, embedding similarity checks, and model evaluation have not been performed.
- ASR/noisy-text entries are synthetic examples, not outputs from a real speech recognizer.
- Review short utterances for potential overlap with other complaint labels before production.
