# Baseline: grouped-holdout SetFit classifier

Trustworthy baseline after the P0 data repair, batch regeneration, targeted
Noise boundary sharpening and the encoder/iteration tuning round. The model is
evaluated on **31 held-out source batches that never appear in train or
validation**, and **all 31 intents have test support**.

| Metric | Value |
|---|---:|
| Rows evaluated | 1,543 |
| Accuracy | **89.44%** |
| Macro F1 | **89.50%** |
| Weighted F1 | **89.48%** |
| Train / val / test batches | 35 / 11 / 31 |
| Train / val / test rows | 1,742 / 549 / 1,543 |
| Train/test batch overlap | **none** |
| Intents trained / evaluated | 31 / 31 |

Model: `sentence-transformers/all-MiniLM-L12-v2` + SetFit
(`--epochs 1 --head-epochs 16 --iterations 32 --batch-size 16`, seed 42).
Full artifacts: [`evaluation/report.json`](report.json) and
[`evaluation/confusion_matrix.csv`](confusion_matrix.csv).
Tuning logs and per-config reports: [`evaluation/tuning/`](tuning/).

Model weights are **not stored in git** — the 133 MB file exceeds GitHub's
100 MB blob limit. Either fetch the published asset:

```bash
gh release download model-v1 -p model.safetensors -D models/intent_classifier
```

or regenerate them with the training command below. Everything else in
`models/intent_classifier/` (config, label map, head, tokenizer) is tracked.

## Tuning round (validation-selected, test measured once)

Configs were compared on the grouped validation split only; the test split was
scored once, after the winner was fixed. Validation covers 11 intents, so the
test column is the authoritative comparison.

| Configuration | Val accuracy | Val macro F1 | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| MiniLM-L6, it4 (previous baseline) | 87.07% | 90.31% | 84.96% | 85.24% |
| MiniLM-L6, it8 | 89.07% | 91.36% | – | – |
| MiniLM-L6, it16 | 90.71% | 92.98% | – | – |
| MiniLM-L6, it32 | 92.53% | 94.19% | – | – |
| MiniLM-L6, it64 | 92.35% | 94.32% | – | – |
| bge-small-en-v1.5, it32 | 92.35% | 94.60% | – | – |
| **MiniLM-L12, it32 (current)** | **93.99%** | **95.82%** | **89.44%** | **89.50%** |

Notes from the sweep:

* `head_epochs` 8/16/32 and `batch_size` 16/32 were flat within the same encoder.
* `iterations` 16 → 32 was the largest single gain (+1.2 val macro on MiniLM-L6);
  32 → 64 plateaued, so 32 is used.
* `all-mpnet-base-v2` was abandoned: ~3 h for 62% of one training phase on this
  CPU, versus ~25 min for a full MiniLM-L12 run.
* The current model is the exact artifact that produced the test numbers above
  (`models/experiments/minilm12_it32_ep1_h16`, copied to
  `models/intent_classifier`). It is trained on `train.jsonl` only, so the gain
  versus the previous baseline comes from encoder + iterations, not extra data.

## Weak classes

| Class | F1 | Main confusion |
|---|---:|---|
| `NOISE_BACKGROUND_RESTAURANT` | 0.74 | precision-limited; splits across `LOUDNESS_LOUD_TOO_LOUD` and `NOISE_CROWD_TOO_LOUD` |
| `SPEECH_TOO_LOUD_BOOMY` | 0.77 | ↔ `SOUND_TOO_MUCH_BASS` |
| `SOUND_TOO_MUCH_BASS` | 0.78 | → `SPEECH_TOO_LOUD_BOOMY` (17/50) |
| `SPEECH_WHISPER_DISTANT_SOFT` | 0.78 | → `SPEECH_TOO_SOFT` (5), `LOUDNESS_NEARBY_TOO_LOUD` |
| `LOUDNESS_SUDDEN_SOUNDS_LOUD` | 0.79 | ↔ `NOISE_SUDDEN_UNCOMFORTABLE` (7) |
| `NOISE_SUDDEN_UNCOMFORTABLE` | 0.79 | → `LOUDNESS_SUDDEN_SOUNDS_LOUD` (12/50) |
| `SPEECH_TOO_SOFT` | 0.79 | ↔ `SPEECH_WHISPER_DISTANT_SOFT` (8) |

Compared with the previous baseline, 24 of 31 classes improved. The largest
gains were `NOISE_SOUND_UNCOMFORTABLE` 0.81 → 0.96,
`LOUDNESS_LOUD_DISTORTED` 0.84 → 0.97, `NOISE_AMBIENT_NOTICEABLE` 0.74 → 0.85
and `LOUDNESS_NEARBY_TOO_LOUD` 0.85 → 0.95.

Two classes regressed: `SPEECH_WHISPER_DISTANT_SOFT` 0.85 → 0.78 and
`SOUND_TOO_MUCH_BASS` 0.83 → 0.78 (now pulled toward `SPEECH_TOO_LOUD_BOOMY`).
The remaining weak pairs are semantic overlaps, so further gains need either a
domain decision to merge some concepts or targeted data for the
quiet/bass/boominess boundary — not more training iterations.

## Previous baseline (pre-tuning)

The pre-tuning baseline was 84.96% accuracy / 85.24% macro-F1 with the same
splits. Its history is kept here because the repair sequence is still the
justification for the dataset:

| Training data | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| 30 intents, batch 17 quarantined (pre-repair) | 83.58% | 81.08% | reference |
| All 11 regenerated batches added | ≈77% | ≈73% | rejected: template text shifted boundaries |
| Only batch 17 added | 83.23% | 78.17% | restores the missing class |
| + batch 16 (ambient sharpening) | 83.85% | 78.87% | ambient 0.55 → 0.70 |
| + batch 14/17 as test-support (previous baseline) | 84.96% | 85.24% | all 31 intents evaluated |

## Caveats

* The training and test utterances are synthetic paraphrases (50 per source
  complaint), so template similarity makes this an **upper bound** for real
  user speech, not an estimate of production accuracy.
* Batches 14, 16, 17 and 85 are template-generated, not LLM-authored or
  human-reviewed.
* The metric is intent accuracy, not end-to-end DSP-action accuracy.
* Confidence is not calibrated: several correct predictions fall below the
  API's `0.65` clarification threshold.

## Reproduce

```bash
source .venv/bin/activate
python data_pipeline/audit_batches.py
python data_pipeline/clean_data.py
python scripts/validate_dataset.py
python training/train_classifier.py \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --iterations 32 --epochs 1 --head-epochs 16
python evaluation/evaluate.py
```

To re-run the sweep: `python training/tune.py --set all` (writes
`models/experiments/<name>` and `evaluation/tuning/results.json`).
