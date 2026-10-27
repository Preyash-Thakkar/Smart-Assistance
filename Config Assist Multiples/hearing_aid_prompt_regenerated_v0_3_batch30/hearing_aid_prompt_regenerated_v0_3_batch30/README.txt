Hearing Aid User Assistant Dataset — Batch 30

Source complaint (row 30, 1-indexed): Sudden Sounds (e.g., Door Slams) Too Loud
Topic: Loudness
Topic ID: 66add6ac8d19f934bd07c673
Provisional complaint ID: PROV-CMP-0030

Contents
- batch30_complaint_english.jsonl: 50 utterance records (42 natural-form candidates + 8 synthetic ASR/noisy-text candidates)
- batch30_glossary.json: key semantic terms
- batch30_source_mapping.json: traceability to source row and taxonomy
- batch30_self_audit.json: automated checks and limitations
- taxonomy_master_provisional.json: provisional mapping for this batch
- dataset_manifest.json: batch metadata

Important limitations
- Source complaint JSON has no dedicated complaint ID field; PROV-CMP-0030 is provisional and must be reconciled with the canonical taxonomy.
- All utterances and scores are synthetic drafts. Naturalness/confidence scores are estimates, not human ratings.
- Native-speaker/domain review, embedding similarity checks, and model evaluation have not been performed.
- ASR/noisy-text entries are synthetic examples, not outputs from a real speech recognizer.
- Review overlap with general loudness, alarm/crying, and sudden-noise complaint categories before production.
