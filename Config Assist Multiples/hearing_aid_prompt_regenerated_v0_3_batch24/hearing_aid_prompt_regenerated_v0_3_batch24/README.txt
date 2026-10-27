Hearing Aid User Assistant Dataset — Batch 24

Source complaint (row 24, 1-indexed): Hearing Aid whistles
Topic: Feedback
Topic ID: 66b061095c6674686e02d03e
Provisional complaint ID: PROV-CMP-0024

Contents
- batch24_complaint_english.jsonl: 50 utterance records (42 natural-form candidates + 8 synthetic ASR/noisy-text candidates)
- batch24_glossary.json: key semantic terms
- batch24_source_mapping.json: traceability to source row and taxonomy
- batch24_self_audit.json: automated checks and limitations
- taxonomy_master_provisional.json: provisional mapping for this batch
- dataset_manifest.json: batch metadata

Important limitations
- Source complaint JSON has no dedicated complaint ID field; PROV-CMP-0024 is provisional and must be reconciled with the canonical taxonomy.
- All utterances and scores are synthetic drafts. Naturalness/confidence scores are estimates, not human ratings.
- Native-speaker/domain review, embedding similarity checks, and model evaluation have not been performed.
- ASR/noisy-text entries are synthetic examples, not outputs from a real speech recognizer.
- Review overlap with adjacent feedback/device-fit complaints before production.
