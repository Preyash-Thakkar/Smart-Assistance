# Smart-Assistance / WeHear Config Assist

Hearing-aid complaint understanding: a user describes a problem in natural
language, the model predicts an acoustic **intent**, and the API returns the
matching DSP configuration payload.

> **Branch note:** the dataset exists on `main`; the runnable pipeline and the
> trained model live on `first-model`.

## Repository layout

```
Real Data/                     MongoDB exports (topics + complaints + DSP solutions)
Config Assist Multiples/       Generated complaint batches (84 folders)
data_pipeline/                 Data audit, label repair and split builder
data/processed/                Generated train/val/test + DSP lookup database
training/                      SetFit trainer (train_classifier.py)
evaluation/                    Grouped-holdout evaluation
inference/                     FastAPI service
models/intent_classifier/      Trained SetFit model
scripts/validate_dataset.py    Dependency-free dataset checks
```

## P0 changes (label + leakage repair)

The previous pipeline had two problems that capped accuracy:

1. **47% of rows were mislabelled.** `clean_data.py` only understood one of
   the five source-mapping schemas used across the batches, so 82 of 84
   batches received no source label. `auto_mapper.py` then guessed one label
   per batch from a single utterance and overwrote every row.
2. **100% test leakage.** The old split shuffled individual utterances, so
   paraphrases from the same complaint appeared in both train and test.

P0 fixes this:

* `data_pipeline/source_metadata.py` reads every mapping schema and takes the
  authoritative complaint from the matching row in
  `Real Data/test.complains.json`.
* `data_pipeline/taxonomy.py` is the single source of truth for
  question → intent.
* `data_pipeline/audit_batches.py` flags content/glossary rows that do not
  match their source row. Confirmed mismatches are listed in
  `data_pipeline/quarantine.txt` and excluded by default.
* `data_pipeline/clean_data.py` builds **grouped** splits by source batch, so
  no batch can appear in two splits.
* `evaluation/evaluate.py` reports accuracy, macro-F1, weighted-F1, per-class
  metrics, a confusion matrix and a leakage check.
* `auto_mapper.py` is now an optional semantic **audit**; it no longer writes
  training labels.

After excluding the mismatched batches, 30 intents have usable training data.
`NOISE_SOUND_UNCOMFORTABLE` has no clean batch yet: its only batch (17) contains
ambient-noise text, so it is quarantined and the class is kept in the DSP
database for when a correct batch is generated.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
# CPU-only machines can install torch from the CPU index first (smaller download):
#   pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

# 1. Audit the generated batches (no ML deps required)
python data_pipeline/audit_batches.py

# 2. Build corrected, grouped splits + DSP database
python data_pipeline/clean_data.py

# 3. Validate without loading a model
python scripts/validate_dataset.py

# 4. Train
python training/train_classifier.py

# 5. Evaluate on unseen batches
python evaluation/evaluate.py

# 6. Serve
python inference/api.py
```

The API is then available at `http://localhost:8000/docs`.

## Baseline

The first grouped-holdout baseline is recorded in
[`evaluation/BASELINE.md`](evaluation/BASELINE.md): **83.58% accuracy / 81.08%
macro-F1** on 1,443 utterances from 29 unseen batches, with no train/test batch
overlap. Read the caveats there before treating it as production accuracy.

## Split modes

`clean_data.py` defaults to `--split-mode coverage`:

* every intent with at least two batches gets a held-out test batch;
* every intent with at least three batches also gets a validation batch;
* a one-batch intent can only appear in train (the report warns about it).

Use `--split-mode ratio` for a plain grouped 80/10/10 shuffle. It may leave
some intents out of test, so check `data/processed/split_report.json`.

## Artifacts

| File | Purpose |
|---|---|
| `data/processed/train.jsonl` | Grouped training rows |
| `data/processed/val.jsonl` | Grouped validation rows |
| `data/processed/test_holdout.jsonl` | Unseen-batch test rows |
| `data/processed/complains_cleaned.json` | Intent → DSP payload lookup used by the API |
| `data/processed/split_report.json` | Counts, class coverage, leakage checks, warnings |
| `data/processed/batch_metadata.json` | Per-batch source row, intent and audit flags |
| `data/processed/batch_audit.json` | Full content/source audit |
| `data/processed/label_map.json` | Intent → class index |
| `evaluation/report.json` | Accuracy, macro-F1, per-class report, leakage |
| `evaluation/confusion_matrix.csv` | Confusion matrix |

## Quarantine

`data_pipeline/quarantine.txt` lists batches whose generated content does not
match the source row (for example batch 17 records ambient-noise text but its
source row is "Sound in noise is uncomfortable", and batch 83 records wind
noise but its source row is "Hearing Aid Whistles"). They are excluded by
default. Re-audit with:

```bash
python data_pipeline/audit_batches.py
```

To include them anyway (not recommended until the content is regenerated):

```bash
python data_pipeline/clean_data.py --include-quarantined
```

## Next steps after P0

1. Regenerate the quarantined batches from the correct source rows.
2. Add real user utterances and active-learning review.
3. Tune the model only after the corrected baseline is measured
   (`--iterations`, `--epochs`, stronger encoder, hierarchical topic→intent).
4. Evaluate end-to-end DSP-action accuracy, not only intent accuracy.
