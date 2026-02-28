from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# 1. Update the Schema to match your 7 new features
class LoginRequest(BaseModel):
    attempts_per_min: int
    fail_ratio: float
    unique_accounts: int
    device_change: int
    geo_anomaly: int
    honeypot: int
    typing_speed: float

# Load the model
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "..", "ml_engine", "models", "detector_v1.pkl")
model = joblib.load(MODEL_PATH)

@app.post("/v1/score")
async def get_risk_score(request: LoginRequest):
    # 1. IMMEDIATE BLOCK (Short-Circuit) 
    # If the honeypot is filled, it's a bot. No ML needed.
    if request.honeypot == 1:
        return {
            "risk_score": 1.0,
            "action": "BLOCK",
            "is_anomaly": True,
            "reason": "Honeypot Triggered: Automated Bot Detected"
        }

    # 2. BEHAVIORAL ANALYSIS (ML Path) [cite: 33, 38]
    # If no honeypot, use the Isolation Forest to check for stealthy patterns
    features = np.array([[
        request.attempts_per_min, 
        request.fail_ratio, 
        request.unique_accounts, 
        request.device_change, 
        request.geo_anomaly, 
        request.honeypot, # Keep in vector for model consistency
        request.typing_speed
    ]])
    
    raw_anomaly_score = model.decision_function(features)[0]
    # Sigmoid function to normalize risk [cite: 34, 39]
    risk_score = 1 / (1 + np.exp(raw_anomaly_score * 12)) 
    
    # 3. DYNAMIC RESPONSE LOGIC [cite: 40, 42]
    action = "ALLOW"
    if risk_score > 0.7:
        action = "BLOCK"
    elif risk_score > 0.35:
        action = "MFA"
        
    return {
        "risk_score": round(float(risk_score), 3),
        "action": action,
        "is_anomaly": bool(risk_score > 0.5)
    }