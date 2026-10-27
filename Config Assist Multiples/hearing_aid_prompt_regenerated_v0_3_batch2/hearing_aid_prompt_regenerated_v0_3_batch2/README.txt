HEARING-AID AI COMPLAINT DATASET — PROMPT-BASED REGENERATION, BATCH 2

Cell:
- Source complaint row: 2
- Source complaint text: All Loud sounds are too loud
- Topic: Loudness
- Language: English (Indian English/general)
- 50 utterances: 42 natural-form candidates + 8 linked ASR-noise candidates

Ground-truth topic mapping:
complaints.topic_id.$oid == topics._id.$oid
The source topic ID is copied unchanged and the topic name comes from topics.title.

ID limitation:
The source complaint JSON has no individual complaint_id field. PROV-CMP-0002 is a provisional local batch ID, not an official source ID.

All utterances are synthetic and unreviewed. Exact normalized duplicates were checked; embedding similarity, native-speaker review, domain review, and model evaluation were not performed.
