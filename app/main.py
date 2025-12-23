import os
import time
import logging
import numpy as np
import joblib
import sys
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram, REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger import jsonlogger

from config import settings
from drift_detector import SimpleDriftDetector
from model_loader import AIOpsModel

# Logging
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(message)s')
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(settings.log_level)

# Prometheus metrics
try:
    predictions_total = Counter('aiops_predictions_total', 'Total predictions')
except:
    predictions_total = REGISTRY._names_to_collectors.get('aiops_predictions_total')

try:
    drift_detected_total = Counter('aiops_drift_total', 'Drift detections')
except:
    drift_detected_total = REGISTRY._names_to_collectors.get('aiops_drift_total')

try:
    prediction_duration = Histogram('aiops_duration_seconds', 'Duration')
except:
    prediction_duration = REGISTRY._names_to_collectors.get('aiops_duration_seconds')

# Pydantic models
class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_length=1, max_length=100)
    model_config = ConfigDict(json_schema_extra={"example": {"features": [1.2, 3.4, 5.6, 7.8]}})

class PredictionOutput(BaseModel):
    prediction: float
    drift_detected: bool
    drift_info: Dict[str, Any]
    processing_time: float
    model_config = ConfigDict(protected_namespaces=())

# FastAPI
app = FastAPI(title=settings.app_name, version=settings.version)
try:
    Instrumentator().instrument(app).expose(app)
except:
    pass

model = None
drift_detector = None

@app.on_event("startup")
async def startup_event():
    global model, drift_detector
    logger.info("🚀 Starting AIOps")
    
    try:
        if os.path.exists(settings.model_path):
            model = joblib.load(settings.model_path)
            logger.info(f"✅ Model: {getattr(model, 'version', '1.0')}")
        else:
            logger.warning("⚠️ Model file not found, using mock")
            model = AIOpsModel()
            model.version = "0.0.1-mock"
    except Exception as e:
        logger.error(f"❌ Model load error: {e}, using mock")
        model = AIOpsModel()
        model.version = "0.0.1-mock"
    
    drift_detector = SimpleDriftDetector(threshold=settings.drift_threshold)
    drift_detector.fit(np.random.randn(100, 4) * 2 + 5)
    logger.info("✅ Drift detector ready")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Shutdown")

@app.get("/")
def root():
    return {"service": settings.app_name, "status": "healthy"}

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "drift_detector_ready": drift_detector is not None
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput, request: Request):
    start_time = time.time()
    try:
        X = np.array([input_data.features])
        
        with prediction_duration.time():
            prediction = model.predict(X)[0]
        
        drift_info = drift_detector.detect(X)
        
        if drift_info['drift_detected']:
            drift_detected_total.inc()
            logger.warning(f"🚨 DRIFT! Z={drift_info['max_z_score']:.2f}")
        
        predictions_total.inc()
        processing_time = time.time() - start_time
        
        return PredictionOutput(
            prediction=float(prediction),
            drift_detected=drift_info['drift_detected'],
            drift_info=drift_info,
            processing_time=processing_time
        )
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def stats():
    try:
        pred = predictions_total._value._value
        drift = drift_detected_total._value._value
    except:
        pred, drift = 0, 0
    return {
        "predictions": pred,
        "drifts": drift,
        "model": getattr(model, 'version', 'unknown')
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False, log_config=None)
