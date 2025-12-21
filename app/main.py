from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="AIOps Inference Service")

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.get("/")
def root():
    return {"status": "healthy", "service": "aiops-inference"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(data: dict):
    return {"prediction": 42.0, "drift_detected": False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
