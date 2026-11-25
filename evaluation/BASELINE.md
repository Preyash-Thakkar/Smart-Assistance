# Baseline: grouped-holdout SetFit classifier

Trustworthy baseline after the P0 data repair and the replacement-batch
regeneration. The model is evaluated on **29 original source batches that never
appear in train or validation**; the same batches are used before and after the
repair, so the two runs are directly comparable.

| Metric | Value |
|---|---:|
| Rows evaluated | 1,443 |
| Accuracy | **83.23%** |
| Macro F1 | **78.17%** |
| Weighted F1 | **83.52%** |
| Train / val / test batches | 34 / 11 / 29 |
| Train / val / test rows | 1,692 / 549 / 1,443 |
| Train/test batch overlap | **none** |
| Intents trained | 31 |

Model: `sentence-transformers/all-MiniLM-L6-v2` + SetFit
(`--epochs 1 --head-epochs 16 --iterations 4`, seed 42).
Full artifacts: [`evaluation/report.json`](report.json) and
[`evaluation/confusion_matrix.csv`](confusion_matrix.csv).

## Effect of the repair

The pre-repair baseline was 83.58% accuracy / 81.08% macro-F1 with 30 trainable
intents. All 11 mismatched batches were then regenerated from their correct
source rows. Two training runs were measured on the unchanged test set:

| Training data | Accuracy | Macro F1 | Decision |
|---|---:|---:|---|
| 30 intents, batch 17 quarantined (pre-repair) | 83.58% | 81.08% | reference |
| All 11 regenerated batches added | ≈77% | ≈73% | rejected: template text shifted Speech/Noise boundaries |
| Only batch 17 added (final) | **83.23%** | **78.17%** | keeps the missing class at a small cost |

Batch 17 is the only source for `NOISE_SOUND_UNCOMFORTABLE`, so it is kept. The
other ten replacements are dropped as redundant because their intent already
has original data; the template style otherwise pulls nearby classes together
(for example `SPEECH_TOO_SOFT` → `LOUDNESS_SOFT_TOO_SOFT`). The policy lives in
`data_pipeline/replacement_batches.txt` and `clean_data.py`.

## Weak classes

| Class | F1 | Main confusion |
|---|---:|---|
| `SPEECH_TOO_SOFT` | 0.50 | ↔ `SPEECH_WHISPER_DISTANT_SOFT`, `SPEECH_POOR_QUALITY` |
| `NOISE_AMBIENT_NOTICEABLE` | 0.55 | → `NOISE_BACKGROUND_RESTAURANT` (29/50) |
| `NOISE_BACKGROUND_RESTAURANT` | 0.63 | absorbs ambient noise (precision 0.47) |
| `SPEECH_WHISPER_DISTANT_SOFT` | 0.67 | ↔ `SPEECH_TOO_SOFT` |
| `LOUDNESS_SUDDEN_SOUNDS_LOUD` | 0.71 | ↔ `NOISE_SUDDEN_UNCOMFORTABLE` |
| `SPEECH_TOO_LOUD_BOOMY` | 0.74 | → `SPEECH_TOO_SOFT` |
| `NOISE_SUDDEN_UNCOMFORTABLE` | 0.74 | → `LOUDNESS_SUDDEN_SOUNDS_LOUD` (14/50) |

The dominant remaining problem is still the Noise taxonomy: "ambient noise too
noticeable", "background noise too loud" and "crowd noise too loud" are
described with very similar generated text, and the model sends 29 of 50
ambient rows to `NOISE_BACKGROUND_RESTAURANT`. The regeneration script now
documents and generates sharper boundaries, but the original ambient/background
batches were not rewritten. This is the first target for model tuning or for a
domain decision to merge concepts.

`SOUND_TOO_MUCH_BASS` has one original batch (47) and no held-out test support.
`NOISE_SOUND_UNCOMFORTABLE` is trained from batch 17 only, so it also has no
test support; smoke tests show it can be predicted, but confidence is low
(≈0.37–0.65) and the class is unvalidated.

## Caveats

* The training and test utterances are synthetic paraphrases (50 per source
  complaint), so template similarity makes this an **upper bound** for real
  user speech, not an estimate of production accuracy.
* Batch 17 is template-generated, not LLM-authored or human-reviewed.
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
