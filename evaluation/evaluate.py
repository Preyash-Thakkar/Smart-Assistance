import os
import json
import pandas as pd
from setfit import SetFitModel
from sklearn.metrics import accuracy_score, classification_report

def load_jsonl(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return pd.DataFrame(data)

def main():
    test_path = "data/processed/test_holdout.jsonl"
    model_path = "models/intent_classifier"
    
    if not os.path.exists(model_path):
        print("Model not found. Ensure training completed successfully.")
        return

    print("Loading test data and model...")
    test_df = load_jsonl(test_path)
    # FIX: We now evaluate against the true intent_code, not the batch folder name
    test_df = test_df.dropna(subset=["utterance", "intent_code"])
    
    model = SetFitModel.from_pretrained(model_path)
    
    print("Running predictions on holdout set...")
    preds = model.predict(list(test_df["utterance"]))
    
    # FIX: Ground truth is intent_code
    y_true = list(test_df["intent_code"])
    
    acc = accuracy_score(y_true, preds)
    print(f"\nOverall Accuracy: {acc * 100:.2f}%\n")
    
    print(classification_report(y_true, preds, zero_division=0))

if __name__ == "__main__":
    main()