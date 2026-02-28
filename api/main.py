from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

# 1. Updated Schema to include registration flag
class LoginRequest(BaseModel):
    attempts_per_min: int
    fail_ratio: float
    unique_accounts: int
    device_change: int
    geo_anomaly: int
    honeypot: int
    typing_speed: float
    is_registration: bool  # Set True for signup, False for login

# Load the model
current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "..", "ml_engine", "models", "detector_v1.pkl")
model = joblib.load(MODEL_PATH)

@app.post("/v1/score")
async def get_risk_score(request: LoginRequest):
    # --- 1. REGISTRATION BASELINE LOGIC ---
    # During registration, the current device IS the home device.
    # We force device_change to 0 so the ML doesn't penalize a new user.
    effective_device_change = 0 if request.is_registration else request.device_change
    
    status_msg = "Baseline established during registration." if request.is_registration else "Standard login analysis."

    # --- 2. DETERMINISTIC HARD RULES (Zero Tolerance) ---
    
    # Honeypot check: Bots clicking hidden fields are blocked regardless of phase.
    if request.honeypot == 1:
        return {
            "risk_score": 1.0, "action": "BLOCK", "is_anomaly": True,
            "reason": f"Honeypot Triggered. {status_msg}"
        }

    # Inhuman Speed Check: Instant block if registration/login is too fast for a human.
    if request.typing_speed < 0.8:
        return {
            "risk_score": 1.0, "action": "BLOCK", "is_anomaly": True,
            "reason": f"Inhuman interaction speed. {status_msg}"
        }

    # High Velocity Check
    if request.attempts_per_min > 50:
         return {
            "risk_score": 0.95, "action": "BLOCK", "is_anomaly": True,
            "reason": "High-frequency attack pattern detected."
        }

    # --- 3. BEHAVIORAL ANALYSIS (ML Path) ---
    # We use 'effective_device_change' so the model sees '0' for all registrations.
    features = np.array([[
        request.attempts_per_min, 
        request.fail_ratio, 
        request.unique_accounts, 
        effective_device_change, 
        request.geo_anomaly, 
        request.honeypot, 
        request.typing_speed
    ]])
    
    raw_anomaly_score = model.decision_function(features)[0]
    
    # Sigmoid normalization with stiffness 15
    risk_score = 1 / (1 + np.exp(raw_anomaly_score * 15)) 
    
    # --- 4. CONTEXTUAL OVERRIDE (Safety Net) ---
    # If it's a login on a new device but typing is perfectly human and no fails:
    # downgrade BLOCK to MFA to reduce False Positives.
    is_human = request.typing_speed > 2.5 and request.fail_ratio == 0
    if not request.is_registration and request.device_change == 1 and is_human:
        risk_score = min(risk_score, 0.45)
        status_msg = "New device detected with human behavior. Escalating to MFA."

    # --- 5. DYNAMIC RESPONSE LOGIC ---
    action = "ALLOW"
    if risk_score > 0.65:
        action = "BLOCK"
    elif risk_score > 0.30:
        action = "MFA"
        
    return {
        "risk_score": round(float(risk_score), 3),
        "action": action,
        "is_anomaly": bool(risk_score > 0.5),
        "reason": status_msg,
        "telemetry": {
            "mode": "Registration" if request.is_registration else "Login",
            "device_status": "Trusted/Home" if effective_device_change == 0 else "Unrecognized"
        }
    }

@app.get("/")
async def health():
    return {"status": "Beyond Limits ML Engine Online", "active_model": "Isolation Forest v1.8"}