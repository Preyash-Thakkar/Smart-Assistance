# Baseline: grouped-holdout SetFit classifier

First trustworthy baseline after the P0 data repair. The model is evaluated on
**29 source batches that never appear in train or validation**.

| Metric | Value |
|---|---:|
| Rows evaluated | 1,443 |
| Accuracy | **83.58%** |
| Macro F1 | **81.08%** |
| Weighted F1 | **83.83%** |
| Train / val / test batches | 33 / 11 / 29 |
| Train / val / test rows | 1,642 / 549 / 1,443 |
| Train/test batch overlap | **none** |

Model: `sentence-transformers/all-MiniLM-L6-v2` + SetFit
(`--epochs 1 --head-epochs 16 --iterations 4`, seed 42).
Full artifacts: [`evaluation/report.json`](report.json) and
[`evaluation/confusion_matrix.csv`](confusion_matrix.csv).

## What changed relative to the old number

The old pipeline reported accuracy on a random *utterance* split. Every test
row shared a source batch with train, so paraphrases of the same complaint were
on both sides. That number was not comparable to this one.

This baseline is grouped by batch and uses labels derived from
`Real Data/test.complains.json`, not from `auto_mapper.py` guesses. Eleven
batches whose generated content does not match their source row are excluded
(see `data_pipeline/quarantine.txt`).

## Weak classes

| Class | F1 | Main confusion |
|---|---:|---|
| `NOISE_AMBIENT_NOTICEABLE` | 0.52 | → `NOISE_BACKGROUND_RESTAURANT` (29/50) |
| `SPEECH_TOO_SOFT` | 0.53 | ↔ `SPEECH_WHISPER_DISTANT_SOFT`, `SPEECH_POOR_QUALITY` |
| `NOISE_BACKGROUND_RESTAURANT` | 0.64 | absorbs ambient/crowd noise (precision 0.48) |
| `SPEECH_TOO_LOUD_BOOMY` | 0.74 | → `SPEECH_TOO_SOFT` (13/50) |
| `LOUDNESS_SUDDEN_SOUNDS_LOUD` | 0.74 | ↔ `NOISE_SUDDEN_UNCOMFORTABLE` |

The strongest signal is that the Noise taxonomy overlaps semantically:
"ambient noise too noticeable", "background noise too loud" and "crowd noise
too loud" are described with very similar generated text. Before tuning the
encoder, either regenerate those batches with sharper concept boundaries or
make a domain decision to merge some of them.

`SOUND_TOO_MUCH_BASS` has only one usable batch (47), so it is trained on but
cannot appear in the test split (`support = 0`). `NOISE_SOUND_UNCOMFORTABLE`
has **no** usable batch: its only batch (17) contains ambient-noise text and is
quarantined. It remains in the DSP database but the model cannot predict it.

## Caveats

* The training and test utterances are synthetic paraphrases (50 per source
  complaint), so template similarity makes this an **upper bound** for real
  user speech, not an estimate of production accuracy.
* The metric is intent accuracy, not end-to-end DSP-action accuracy.
* Confidence is not calibrated: several correct predictions fall below the
  API's `0.65` clarification threshold.

## Reproduce

```bash
source .venv/bin/activate
python data_pipeline/clean_data.py
python scripts/validate_dataset.py
python training/train_classifier.py
python evaluation/evaluate.py
```
