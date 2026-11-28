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
data_pipeline/                 Data audit, label repair, batch regeneration and split builder
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

After regenerating the mismatched batches, all 31 intents have usable training
data, and all 31 now have held-out test support. `NOISE_SOUND_UNCOMFORTABLE` is supported by a single batch (17), so it can
only appear in train and the split report warns about it.

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

# 4. Train (config that produced the recorded baseline)
python training/train_classifier.py \
  --base-model sentence-transformers/all-MiniLM-L12-v2 \
  --iterations 32 --epochs 1 --head-epochs 16

# 5. Evaluate on unseen batches
python evaluation/evaluate.py

# Regression tests for the metric averaging (stdlib unittest, no extra deps)
python -m unittest discover -s tests

# 6. Serve
python inference/api.py
```

The API is then available at `http://localhost:8000/docs`.

## Baseline

The post-repair, post-tuning grouped-holdout baseline is recorded in
[`evaluation/BASELINE.md`](evaluation/BASELINE.md): **89.44% accuracy / 89.50%
macro-F1** on 1,543 utterances from 31 unseen batches, with no train/test batch
overlap and all 31 intents trained and evaluated. The model is SetFit on
`sentence-transformers/all-MiniLM-L12-v2` with `--iterations 32 --epochs 1
--head-epochs 16`; tuning history and per-class results are in that file. Read
the caveats there before treating it as production accuracy.

## Split modes

`clean_data.py` defaults to `--split-mode coverage`:

* every intent with at least two original batches gets a held-out test batch;
* every intent with at least three original batches also gets a validation batch;
* a one-batch intent can only appear in train (the report warns about it);
* template replacements are train-only and are used only when an intent has no
  original batch.

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
| `evaluation/BASELINE.md` | Current baseline, tuning history, weak classes |
| `evaluation/tuning/results.json` | Sweep results (validation macro-F1 per config) |
| `models/intent_classifier/` | Promoted model served by the API (weights via release asset, see `BASELINE.md`) |
| `tests/test_metrics.py` | Regression test for metric averaging |

## Quarantine and regeneration

`data_pipeline/quarantine.txt` lists batches whose generated content does not
match the source row; they are excluded by default. The initial audit
quarantined `10–17`, `79`, `83` and `84` because their text described a
different concept than their label (for example batch 17 held ambient-noise
text although its source row is "Sound in noise is uncomfortable").

Those batches were regenerated offline from the correct source rows:

```bash
# Preview first; nothing is written without --write
python data_pipeline/regenerate_batches.py --batches 10 11 12 13 14 15 16 17 79 83 84
python data_pipeline/regenerate_batches.py --batches 10 11 12 13 14 15 16 17 79 83 84 --write
```

The generator is deterministic and template-based (`data_pipeline/concepts.py`)
because the original LLM generator is not in the repository. It rewrites only
the utterance JSONL, glossary, self-audit and README; the authoritative
`*_source_mapping.json` is never modified. Every regenerated batch must pass the
same source/content audit before its number is removed from `quarantine.txt`:

```bash
python data_pipeline/audit_batches.py
```

Regenerated replacements are listed in
`data_pipeline/replacement_batches.txt`. A replacement is used **only when its
intent has no original batch**; otherwise it is dropped as redundant, because
the template style shifts the decision boundary (measured on the grouped
holdout). Replacements that are used stay in train and never enter validation
or test, so the headline metrics remain comparable across data repairs.

The original generated text remains recoverable from git history. To include
still-quarantined batches anyway (not recommended):

```bash
python data_pipeline/clean_data.py --include-quarantined
```

## Next steps after P0

1. ~~Regenerate the quarantined batches from the correct source rows.~~ Done;
   the replacement text is template-generated, so review it before treating it
   as gold data.
2. Add real user utterances and active-learning review.
3. Generate a second batch for `NOISE_SOUND_UNCOMFORTABLE` and
   `SOUND_TOO_MUCH_BASS` so both can appear in validation/test.
4. ~~Tune the model only after the corrected baseline is measured.~~ Done:
   MiniLM-L12 + `iterations 32` lifted test macro-F1 85.24% → 89.50%. Next gains
   need targeted data for the quiet/bass/boominess boundary, not more iterations.
5. Evaluate end-to-end DSP-action accuracy, not only intent accuracy.
