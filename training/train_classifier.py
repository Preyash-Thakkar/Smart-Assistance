"""Train the SetFit intent classifier on the corrected, grouped splits.

P0 changes:

* trains on ``intent_code`` from the source complaint row (not a batch-folder
  guess and not the 84 synthetic batch names);
* keeps the label map next to the model;
* exposes the few-shot hyperparameters so accuracy can be tuned without
  editing the file.

Usage::

    python training/train_classifier.py
    python training/train_classifier.py --iterations 8 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import os

import pandas as pd
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments

DEFAULT_TRAIN = os.path.join("data", "processed", "train.jsonl")
DEFAULT_VAL = os.path.join("data", "processed", "val.jsonl")
DEFAULT_OUT = os.path.join("models", "intent_classifier")


def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", default=DEFAULT_TRAIN)
    parser.add_argument("--val", default=DEFAULT_VAL)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Epochs for the sentence-transformer body"
    )
    parser.add_argument(
        "--head-epochs", type=int, default=16, help="Epochs for the SetFit classification head"
    )
    parser.add_argument("--iterations", type=int, default=4, help="SetFit pair-generation iterations")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.train):
        raise SystemExit(f"Missing {args.train}. Run data_pipeline/clean_data.py first.")

    print(f"Loading {args.train} ...")
    train_df = load_jsonl(args.train).dropna(subset=["utterance", "intent_code"])
    val_df = load_jsonl(args.val).dropna(subset=["utterance", "intent_code"]) if os.path.exists(args.val) else pd.DataFrame()

    if train_df.empty:
        raise SystemExit("Training split is empty.")

    labels = sorted(train_df["intent_code"].unique())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}

    train_df["label"] = train_df["intent_code"].map(label2id)
    if not val_df.empty:
        val_df = val_df[val_df["intent_code"].isin(labels)].copy()
        val_df["label"] = val_df["intent_code"].map(label2id)

    print(f"Training rows: {len(train_df)} | validation rows: {len(val_df)} | classes: {len(labels)}")
    print("Class distribution:")
    for label, count in train_df["intent_code"].value_counts().sort_index().items():
        print(f"  {label:<32} {count}")

    train_dataset = Dataset.from_pandas(
        train_df[["utterance", "label"]].rename(columns={"utterance": "text"})
    )
    val_dataset = (
        Dataset.from_pandas(val_df[["utterance", "label"]].rename(columns={"utterance": "text"}))
        if not val_df.empty and val_df["label"].nunique() >= 2
        else None
    )

    print(f"Loading base model: {args.base_model}")
    model = SetFitModel.from_pretrained(args.base_model, labels=labels)

    training_args = TrainingArguments(
        batch_size=args.batch_size,
        num_epochs=(args.epochs, args.head_epochs),
        eval_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="epoch" if val_dataset is not None else "no",
        load_best_model_at_end=val_dataset is not None,
        num_iterations=args.iterations,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    print("Training ...")
    trainer.train()

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    with open(os.path.join(args.output, "label_map.json"), "w", encoding="utf-8") as handle:
        json.dump(id2label, handle, indent=2)
    with open(os.path.join(args.output, "training_meta.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "base_model": args.base_model,
                "epochs": args.epochs,
                "head_epochs": args.head_epochs,
                "iterations": args.iterations,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "classes": labels,
            },
            handle,
            indent=2,
        )

    print(f"\nSaved model and label map to {args.output}")


if __name__ == "__main__":
    main()
