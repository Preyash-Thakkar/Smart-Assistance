"""FastAPI inference service for the hearing-aid Config Assist model.

Request::

    POST /predict
    {"text": "soft sounds are too quiet", "device_id": 6}

The classifier predicts an ``intent_code``; the service then finds the DSP
payload for that intent and device.  P0 adds deterministic device-aware
selection so a generic record cannot shadow a device-specific record.
"""

from __future__ import annotations

import json
import os

from fastapi import FastAPI
from pydantic import BaseModel
from setfit import SetFitModel

MODEL_PATH = os.environ.get("CONFIG_ASSIST_MODEL", os.path.join("models", "intent_classifier"))
DB_PATH = os.environ.get("CONFIG_ASSIST_DB", os.path.join("data", "processed", "complains_cleaned.json"))
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIG_ASSIST_THRESHOLD", "0.65"))

app = FastAPI(title="WeHear Config Assist AI")

print(f"Loading model from {MODEL_PATH} ...")
model = SetFitModel.from_pretrained(MODEL_PATH)

print(f"Loading DSP database from {DB_PATH} ...")
with open(DB_PATH, "r", encoding="utf-8") as handle:
    database = json.load(handle)


class ComplaintRequest(BaseModel):
    text: str
    device_id: int


def select_solution(intent: str, device_id: int) -> tuple[list | None, str, int]:
    """Pick the most specific DSP record for an intent/device pair."""

    candidates = [doc for doc in database if doc.get("intent_code") == intent]
    exact = [doc for doc in candidates if device_id in (doc.get("devices") or [])]
    generic = [doc for doc in candidates if not doc.get("devices")]

    if exact:
        record = exact[0]
        return record.get("complaint_solution", []), "exact_device_match", len(candidates)
    if generic:
        record = generic[0]
        return record.get("complaint_solution", []), "generic_record", len(candidates)
    return None, "no_match", len(candidates)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "database": DB_PATH,
        "records": len(database),
        "labels": len(getattr(model, "labels", [])),
    }


@app.post("/predict")
def predict_intent(request: ComplaintRequest) -> dict:
    prediction = str(model.predict([request.text])[0])
    probabilities = model.predict_proba([request.text])[0]
    confidence = float(max(probabilities))
    needs_clarification = confidence < CONFIDENCE_THRESHOLD

    solution, selection, candidate_count = select_solution(prediction, request.device_id)

    return {
        "user_input": request.text,
        "device_id": request.device_id,
        "predicted_intent": prediction,
        "confidence_percent": round(confidence * 100, 2),
        "needs_clarification": needs_clarification,
        "selection_reason": selection,
        "candidate_records": candidate_count,
        "dsp_actions": solution if solution is not None else "No hardware mapping found for this device.",
    }


if __name__ == "__main__":
    import uvicorn

    print("\n--- Config Assist API ---")
    print("Open http://localhost:8000/docs to test.")
    uvicorn.run(app, host="0.0.0.0", port=8000)
