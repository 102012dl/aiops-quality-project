#!/bin/bash

###############################################################################
# AIOPS QUALITY PROJECT - SIMPLE SETUP
# Створює базову структуру з FastAPI + всіма компонентами
###############################################################################

set -e

echo "🚀 Creating AIOps Quality Project..."
echo ""

# Створення директорій
echo "📁 Creating directories..."
mkdir -p app model helm/templates argocd grafana prometheus tests scripts

# app/requirements.txt
cat > app/requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
numpy==1.24.3
scikit-learn==1.3.2
prometheus-client==0.19.0
prometheus-fastapi-instrumentator==6.1.0
python-multipart==0.0.6
requests==2.31.0
python-json-logger==2.0.7
joblib==1.3.2
EOF

echo "✅ Created app/requirements.txt"

# app/__init__.py
echo '"""AIOps FastAPI Application"""' > app/__init__.py

# app/config.py
cat > app/config.py << 'EOF'
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "AIOps Inference Service"
    version: str = "1.0.0"
    model_path: str = "model.pkl"
    drift_threshold: float = 3.0
    log_level: str = "INFO"
    gitlab_webhook: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()
EOF

echo "✅ Created app/config.py"

# app/drift_detector.py
cat > app/drift_detector.py << 'EOF'
import numpy as np
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleDriftDetector:
    """Simple Z-score based drift detector"""
    
    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self.reference_mean: Optional[np.ndarray] = None
        self.reference_std: Optional[np.ndarray] = None
        self.drift_count = 0
        
    def fit(self, X: np.ndarray):
        """Fit on reference data"""
        self.reference_mean = np.mean(X, axis=0)
        self.reference_std = np.std(X, axis=0) + 1e-6
        logger.info(f"Drift detector fitted on {len(X)} samples")
        
    def detect(self, X: np.ndarray) -> Dict:
        """Detect drift using Z-score"""
        if self.reference_mean is None:
            return {"drift_detected": False, "reason": "Not fitted"}
        
        z_scores = np.abs((X - self.reference_mean) / self.reference_std)
        max_z_score = np.max(z_scores)
        
        drift_detected = max_z_score > self.threshold
        
        if drift_detected:
            self.drift_count += 1
            logger.warning(f"🚨 DRIFT DETECTED! Z-score: {max_z_score:.2f}")
        
        return {
            "drift_detected": drift_detected,
            "max_z_score": float(max_z_score),
            "threshold": self.threshold,
            "drift_count": self.drift_count
        }
EOF

echo "✅ Created app/drift_detector.py"

# app/main.py - ПОВНИЙ FastAPI
cat > app/main.py << 'EOFMAIN'
import os
import time
import logging
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger import jsonlogger
import requests

from config import settings
from drift_detector import SimpleDriftDetector

# ============================================================================
# LOGGING SETUP
# ============================================================================
logHandler = logging.StreamHandler()
formatter = jsonlogger.JsonFormatter(
    '%(asctime)s %(name)s %(levelname)s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logHandler.setFormatter(formatter)
logger = logging.getLogger()
logger.addHandler(logHandler)
logger.setLevel(settings.log_level)

# ============================================================================
# PROMETHEUS METRICS
# ============================================================================
predictions_total = Counter('predictions_total', 'Total number of predictions')
drift_detected_total = Counter('drift_detected_total', 'Total drift detections')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
model_version = Gauge('model_version_info', 'Model version', ['version'])

# ============================================================================
# PYDANTIC MODELS
# ============================================================================
class PredictionInput(BaseModel):
    features: List[float] = Field(..., min_items=1, max_items=100)
    
    class Config:
        json_schema_extra = {
            "example": {"features": [1.2, 3.4, 5.6, 7.8]}
        }

class PredictionOutput(BaseModel):
    prediction: float
    model_version: str
    drift_detected: bool
    drift_info: Dict[str, Any]
    processing_time: float

# ============================================================================
# FASTAPI APP
# ============================================================================
app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="ML Inference Service with Drift Detection"
)

Instrumentator().instrument(app).expose(app)

# Global state
model = None
drift_detector = None

@app.on_event("startup")
async def startup_event():
    global model, drift_detector
    logger.info("🚀 Starting AIOps Inference Service...")
    
    # Load model
    if os.path.exists(settings.model_path):
        model = joblib.load(settings.model_path)
        logger.info(f"✅ Model loaded from {settings.model_path}")
        model_version.labels(version=getattr(model, 'version', '1.0.0')).set(1)
    else:
        logger.warning(f"⚠️ Model not found, using mock model")
        model = MockModel()
    
    # Init drift detector
    drift_detector = SimpleDriftDetector(threshold=settings.drift_threshold)
    reference_data = np.random.randn(100, 4) * 2 + 5
    drift_detector.fit(reference_data)
    logger.info("✅ Drift detector initialized")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("👋 Shutting down")

@app.get("/")
def root():
    return {
        "service": settings.app_name,
        "version": settings.version,
        "status": "healthy"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "drift_detector_ready": drift_detector is not None
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput, request: Request):
    """Main prediction endpoint with drift detection"""
    start_time = time.time()
    
    try:
        X = np.array([input_data.features])
        
        logger.info({
            "event": "prediction_request",
            "features": input_data.features,
            "client_ip": request.client.host
        })
        
        # Prediction
        with prediction_duration.time():
            prediction = model.predict(X)[0]
        
        # Drift detection
        drift_info = drift_detector.detect(X)
        
        if drift_info['drift_detected']:
            drift_detected_total.inc()
            logger.warning({"event": "drift_detected", "drift_info": drift_info})
            
            if settings.gitlab_webhook:
                trigger_retrain()
        
        predictions_total.inc()
        processing_time = time.time() - start_time
        
        logger.info({
            "event": "prediction_response",
            "prediction": float(prediction),
            "drift_detected": drift_info['drift_detected'],
            "processing_time": processing_time
        })
        
        return PredictionOutput(
            prediction=float(prediction),
            model_version=getattr(model, 'version', '1.0.0'),
            drift_detected=drift_info['drift_detected'],
            drift_info=drift_info,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error({"event": "error", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    return {
        "total_predictions": predictions_total._value._value,
        "drift_detections": drift_detected_total._value._value,
        "model_version": getattr(model, 'version', 'unknown'),
        "drift_threshold": settings.drift_threshold
    }

def trigger_retrain():
    if not settings.gitlab_webhook:
        return
    try:
        requests.post(
            settings.gitlab_webhook,
            json={"ref": "final-project", "variables": {"TRIGGER_REASON": "drift_detected"}}
        )
        logger.info("Retrain pipeline triggered")
    except Exception as e:
        logger.error(f"Failed to trigger retrain: {e}")

class MockModel:
    version = "0.0.1-mock"
    def predict(self, X):
        return np.mean(X, axis=1) * 1.5

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_config=None)
EOFMAIN

echo "✅ Created app/main.py (FastAPI service - 15 балів)"

# model/train.py
cat > model/train.py << 'EOF'
import numpy as np
import joblib
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

class AIOpsModel:
    def __init__(self):
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        print(f"🔧 Training model version {self.version}...")
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        score = self.model.score(X_scaled, y)
        print(f"✅ Model trained! R² score: {score:.4f}")
        return self
        
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
        
    def save(self, path="model.pkl"):
        joblib.dump(self, path)
        print(f"💾 Model saved to {path}")

if __name__ == "__main__":
    print("="*60)
    print("🚀 AIOps Model Training")
    print("="*60)
    
    np.random.seed(42)
    X = np.random.randn(1000, 4) * 2 + 5
    y = X[:, 0] * 2 + X[:, 1] * 0.5 - X[:, 2] + np.random.randn(1000) * 0.1
    
    model = AIOpsModel()
    model.fit(X, y)
    model.save("model.pkl")
    
    import os, shutil
    if os.path.exists("../app"):
        shutil.copy("model.pkl", "../app/model.pkl")
        print("📋 Model copied to app/")
    
    print("="*60)
EOF

cat > model/requirements.txt << 'EOF'
numpy==1.24.3
scikit-learn==1.3.2
joblib==1.3.2
pandas==2.1.3
EOF

echo "✅ Created model/train.py"

# Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

EXPOSE 8000

HEALTHCHECK CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

echo "✅ Created Dockerfile"

# docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  inference:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DRIFT_THRESHOLD=3.0
      - LOG_LEVEL=INFO
    volumes:
      - ./logs:/app/logs
EOF

echo "✅ Created docker-compose.yml"

# .gitignore
cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.Python
venv/
ENV/
*.egg-info/
.DS_Store
*.pkl
*.pt
.env
logs/
EOF

echo "✅ Created .gitignore"

echo ""
echo "════════════════════════════════════════════════════════"
echo "✅ БАЗОВА СТРУКТУРА СТВОРЕНА!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "📁 Створені файли:"
echo "   app/main.py         - FastAPI service"
echo "   app/drift_detector.py - Drift detection"
echo "   app/config.py       - Configuration"
echo "   model/train.py      - Model training"
echo "   Dockerfile          - Docker config"
echo ""
echo "🚀 НАСТУПНІ КРОКИ:"
echo ""
echo "1️⃣ Встановити залежності:"
echo "   cd app"
echo "   pip install -r requirements.txt"
echo ""
echo "2️⃣ Тренувати модель:"
echo "   cd model"
echo "   pip install -r requirements.txt"
echo "   python train.py"
echo ""
echo "3️⃣ Запустити FastAPI:"
echo "   cd app"
echo "   python main.py"
echo ""
echo "4️⃣ Тестувати:"
echo "   curl http://localhost:8000/health"
echo '   curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '"'"'{"features": [1,2,3,4]}'"'"
echo ""
echo "🎯 Готово! FastAPI сервіс створено (15 балів)"
echo "════════════════════════════════════════════════════════"








