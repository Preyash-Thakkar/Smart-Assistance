import os
import json
import pandas as pd
from setfit import SetFitModel, Trainer, TrainingArguments
from datasets import Dataset

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def main():
    print("Loading auto-mapped datasets...")
    train_path = os.path.join("data", "processed", "train.jsonl")
    val_path = os.path.join("data", "processed", "val.jsonl")

    train_df = load_jsonl(train_path)
    val_df = load_jsonl(val_path)
    
    # We now train on the corrected 22 intent codes, not the 84 batches.
    train_df = train_df.dropna(subset=["utterance", "intent_code"])
    val_df = val_df.dropna(subset=["utterance", "intent_code"])
    
    unique_labels = sorted(train_df["intent_code"].unique())
        
    label2id = {label: i for i, label in enumerate(unique_labels)}
    id2label = {i: label for label, i in label2id.items()}
    
    train_df["label"] = train_df["intent_code"].map(label2id)
    val_df["label"] = val_df["intent_code"].map(label2id)
    
    train_dataset = Dataset.from_pandas(train_df[["utterance", "label"]].rename(columns={"utterance": "text"}))
    val_dataset = Dataset.from_pandas(val_df[["utterance", "label"]].rename(columns={"utterance": "text"}))
    
    print(f"Initializing SetFit model for {len(unique_labels)} exact acoustic intents...")
    model = SetFitModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2",
        labels=unique_labels
    )
    
    args = TrainingArguments(
        batch_size=16,
        num_epochs=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        num_iterations=2  # Prevents pair explosion, keeps training under 20 mins
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    
    print("Starting exact-match training process...")
    trainer.train()
    
    save_path = os.path.join("models", "intent_classifier")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    
    with open(os.path.join(save_path, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(id2label, f, indent=2)
        
    print(f"\nSuccess! Production model saved to '{save_path}'")

if __name__ == "__main__":
    main()