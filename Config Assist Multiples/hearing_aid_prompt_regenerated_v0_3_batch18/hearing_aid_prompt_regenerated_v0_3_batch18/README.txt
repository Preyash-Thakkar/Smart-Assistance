Hearing Aid User Assistant Dataset — Batch 18

Source complaint (row 18, 1-indexed): Sudden noise is uncomfortable
Topic: Noise
Topic ID: 66b060d95c6674686e02d026
Provisional complaint ID: PROV-CMP-0018

Contents
- batch18_complaint_english.jsonl: 50 utterance records (42 natural-form candidates + 8 ASR/noisy-text candidates)
- batch18_glossary.json: key semantic terms
- batch18_source_mapping.json: traceability to source row and taxonomy
- batch18_self_audit.json: automated checks and limitations
- taxonomy_master_provisional.json: provisional mapping for this batch
- dataset_manifest.json: batch metadata

Important limitations
- Source complaint JSON has no dedicated complaint ID field; PROV-CMP-0018 is provisional and must be reconciled with the canonical taxonomy.
- All utterances and scores are synthetic drafts. Naturalness/confidence scores are estimates, not human ratings.
- Native-speaker/domain review, embedding similarity checks, and model evaluation have not been performed.
- ASR/noisy-text entries are synthetic noise examples, not outputs from a real speech recognizer.
- Review short utterances for potential overlap with other complaint labels before production.
