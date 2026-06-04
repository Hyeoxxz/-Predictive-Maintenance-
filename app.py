import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# 모델 로드
rul_model    = joblib.load("rul_model.pkl")
clf_model    = joblib.load("clf_model.pkl")
scaler       = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")

app = FastAPI(title="설비 예지보전 API")

# 입력 데이터 형식 정의
class SensorInput(BaseModel):
    engine_id:  int
    cycle:      int
    sensor_1:   float
    sensor_2:   float
    sensor_3:   float
    sensor_4:   float
    sensor_5:   float
    sensor_6:   float
    sensor_7:   float
    sensor_8:   float
    sensor_9:   float
    sensor_10:  float
    sensor_11:  float

def build_features(data: SensorInput):
    """센서값 → 모델 입력 벡터 변환"""
    sensor_vals = {f'sensor_{i}': getattr(data, f'sensor_{i}') for i in range(1, 12)}
    row = {}
    for col in feature_cols:
        if col == 'cycle':
            row[col] = data.cycle
        elif '_mean5' in col:
            row[col] = sensor_vals[col.replace('_mean5', '')]
        elif '_std5' in col:
            row[col] = 0.0
        else:
            row[col] = sensor_vals[col]
    return np.array([row[c] for c in feature_cols]).reshape(1, -1)

def get_alert(rul, prob):
    if rul <= 15 or prob >= 0.85:
        return "CRITICAL", f"즉시 점검 필요 — 잔여 수명 약 {rul:.0f}사이클"
    elif rul <= 30 or prob >= 0.5:
        return "WARNING",  f"점검 권고 — 잔여 수명 약 {rul:.0f}사이클"
    else:
        return "NORMAL",   f"정상 운영 중 — 잔여 수명 약 {rul:.0f}사이클"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: SensorInput):
    X = build_features(data)
    X_scaled = scaler.transform(X)

    rul  = float(rul_model.predict(X_scaled)[0])
    prob = float(clf_model.predict_proba(X_scaled)[0][1])
    rul  = max(0.0, round(rul, 1))
    prob = round(prob, 4)

    alert, message = get_alert(rul, prob)

    return {
        "engine_id":          data.engine_id,
        "cycle":              data.cycle,
        "predicted_rul":      rul,
        "failure_probability": prob,
        "alert_level":        alert,
        "message":            message
    }