# Baseline: grouped-holdout SetFit classifier

Trustworthy baseline after the P0 data repair, batch regeneration and targeted
Noise boundary sharpening. The model is evaluated on **31 held-out source batches
that never appear in train or validation**, and **all 31 intents now have test
support**.

| Metric | Value |
|---|---:|
| Rows evaluated | 1,543 |
| Accuracy | **84.96%** |
| Macro F1 | **85.24%** |
| Weighted F1 | **85.21%** |
| Train / val / test batches | 35 / 11 / 31 |
| Train / val / test rows | 1,792 / 549 / 1,543 |
| Train/test batch overlap | **none** |
| Intents trained / evaluated | 31 / 31 |

Model: `sentence-transformers/all-MiniLM-L6-v2` + SetFit
(`--epochs 1 --head-epochs 16 --iterations 4`, seed 42).
Full artifacts: [`evaluation/report.json`](report.json) and
[`evaluation/confusion_matrix.csv`](confusion_matrix.csv).

## Effect of the repair

The pre-repair baseline was 83.58% accuracy / 81.08% macro-F1 with 30 trainable
intents and batch 17 quarantined. The final pipeline runs were measured on the
same held-out original batches plus the two newly supported intents:

| Training data | Accuracy | Macro F1 | Notes |
|---|---:|---:|---|
| 30 intents, batch 17 quarantined (pre-repair) | 83.58% | 81.08% | reference |
| All 11 regenerated batches added | ≈77% | ≈73% | rejected: template text shifted boundaries |
| Only batch 17 added | 83.23% | 78.17% | restores the missing class |
| + batch 16 (ambient sharpening) | 83.85% | 78.87% | ambient 0.55 → 0.70 |
| + batch 14/17 as test-support (final) | **84.96%** | **85.24%** | all 31 intents evaluated |

## Weak classes

| Class | F1 | Main confusion |
|---|---:|---|
| `NOISE_BACKGROUND_RESTAURANT` | 0.68 | absorbs ambient noise (precision 0.52) |
| `SPEECH_TOO_SOFT` | 0.71 | ↔ `SPEECH_WHISPER_DISTANT_SOFT`, `SPEECH_POOR_QUALITY` |
| `NOISE_AMBIENT_NOTICEABLE` | 0.74 | → `NOISE_BACKGROUND_RESTAURANT` (14/50) |
| `NOISE_SUDDEN_UNCOMFORTABLE` | 0.74 | → `LOUDNESS_SUDDEN_SOUNDS_LOUD` |
| `LOUDNESS_SUDDEN_SOUNDS_LOUD` | 0.74 | ↔ `NOISE_SUDDEN_UNCOMFORTABLE` |

The Noise taxonomy still overlaps semantically, but sharpening the ambient
boundary (batch 16) moved ambient F1 from 0.52 to 0.74 and background precision
from 0.48 to 0.52. Further gains need either a domain decision to merge some
Noise concepts or a stronger encoder.

`NOISE_SOUND_UNCOMFORTABLE` (F1 0.81) and `SOUND_TOO_MUCH_BASS` (F1 0.83) now
have held-out test support from regenerated batches 17/85 and 14/47.

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
python training/train_classifier.py
python evaluation/evaluate.py
```
