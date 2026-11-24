import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
from setfit import SetFitModel

app = FastAPI(title="WeHear Config Assist AI")

MODEL_PATH = "models/intent_classifier"
DB_PATH = "data/processed/complains_cleaned.json"

print("Loading AI Model and Database into memory...")
model = SetFitModel.from_pretrained(MODEL_PATH)

# Load the database so we can map intents to DSP payloads
with open(DB_PATH, "r", encoding="utf-8") as f:
    database = json.load(f)

class ComplaintRequest(BaseModel):
    text: str
    device_id: int  # We now require the device_id to fetch the right DSP array

@app.post("/predict")
def predict_intent(request: ComplaintRequest):
    # 1. AI Predicts the Intent
    prediction = model.predict([request.text])[0]
    
    # 2. Get Confidence Score
    probs = model.predict_proba([request.text])[0]
    confidence = float(max(probs))
    needs_clarification = confidence < 0.65
    
    # 3. Lookup the DSP Action in the Database
    hardware_solution = None
    
    # Search the database for a matching intent AND device ID
    for doc in database:
        if doc.get("intent_code") == prediction:
            # Check if this document applies to the requested device_id
            # Note: Legacy rows might not have a 'devices' array
            devices = doc.get("devices", [])
            if request.device_id in devices or not devices:
                hardware_solution = doc.get("complaint_solution", [])
                break
                
    return {
        "user_input": request.text,
        "predicted_intent": prediction,
        "confidence_percent": round(confidence * 100, 2),
        "needs_clarification": needs_clarification,
        "dsp_actions": hardware_solution if hardware_solution else "No hardware mapping found for this device."
    }

if __name__ == "__main__":
    import uvicorn
    print("\n--- FULL API SERVER RUNNING ---")
    print("Test it in your browser at: http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)