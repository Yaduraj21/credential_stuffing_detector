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
    # --- 1. DETERMINISTIC HARD RULES (Zero Tolerance) ---
    
    # Honeypot check (Highest Priority)
    if request.honeypot == 1:
        return {
            "risk_score": 1.0, "action": "BLOCK", "is_anomaly": True,
            "reason": "Honeypot Triggered: Automated Bot Detected"
        }

    # Inhuman Speed Check: If login takes less than 0.8 seconds, it's a bot.
    # No human can load, type, and click that fast.
    if request.typing_speed < 0.8:
        return {
            "risk_score": 1.0, "action": "BLOCK", "is_anomaly": True,
            "reason": "Inhuman interaction speed detected"
        }

    # High Velocity Check: If they are trying 50+ times a minute.
    if request.attempts_per_min > 50:
         return {
            "risk_score": 0.95, "action": "BLOCK", "is_anomaly": True,
            "reason": "High-frequency attack pattern"
        }

    # --- 2. BEHAVIORAL ANALYSIS (ML Path) ---
    features = np.array([[
        request.attempts_per_min, 
        request.fail_ratio, 
        request.unique_accounts, 
        request.device_change, 
        request.geo_anomaly, 
        request.honeypot, 
        request.typing_speed
    ]])
    
    raw_anomaly_score = model.decision_function(features)[0]
    
    # TUNING: Change 12 to 15 or 18 to make the model MORE SENSITIVE.
    # Higher number = steeper risk curve for minor anomalies.
    risk_score = 1 / (1 + np.exp(raw_anomaly_score * 15)) 
    
    # --- 3. DYNAMIC RESPONSE LOGIC ---
    # More aggressive thresholds for a Hackathon environment
    action = "ALLOW"
    if risk_score > 0.65: # Lowered from 0.7 to be more strict
        action = "BLOCK"
    elif risk_score > 0.30: # Lowered from 0.35 to trigger MFA earlier
        action = "MFA"
        
    return {
        "risk_score": round(float(risk_score), 3),
        "action": action,
        "is_anomaly": bool(risk_score > 0.5),
        "telemetry": {
            "speed_check": "Inhuman" if request.typing_speed < 1.5 else "Human-like",
            "score_stiffness": 15
        }
    }