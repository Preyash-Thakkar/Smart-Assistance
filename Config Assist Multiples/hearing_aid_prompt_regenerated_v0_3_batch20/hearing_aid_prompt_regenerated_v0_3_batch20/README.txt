Hearing Aid User Assistant Dataset — Batch 20

Source complaint (row 20, 1-indexed):  Restaurant noise is loud 
Topic: Noise
Topic ID: 66b060d95c6674686e02d026
Provisional complaint ID: PROV-CMP-0020

Contents
- batch20_complaint_english.jsonl: 50 utterance records (42 natural-form candidates + 8 synthetic ASR/noisy-text candidates)
- batch20_glossary.json: key semantic terms
- batch20_source_mapping.json: traceability to source row and taxonomy
- batch20_self_audit.json: automated checks and limitations
- taxonomy_master_provisional.json: provisional mapping for this batch
- dataset_manifest.json: batch metadata

Important limitations
- Source complaint JSON has no dedicated complaint ID field; PROV-CMP-0020 is provisional and must be reconciled with the canonical taxonomy.
- All utterances and scores are synthetic drafts. Naturalness/confidence scores are estimates, not human ratings.
- Native-speaker/domain review, embedding similarity checks, and model evaluation have not been performed.
- ASR/noisy-text entries are synthetic examples, not outputs from a real speech recognizer.
- Review short utterances for potential overlap with other complaint labels before production.
