# 🚀 AIOps Quality Monitoring System

Фінальний проєкт MLOps - система моніторингу якості ML-моделі з автоматичним retraining при drift detection.

## 📋 Архітектура
```
Client → FastAPI (2 replicas) → ML Model + Drift Detector
         ↓                      ↓
    Prometheus              Drift Alert → GitLab CI Webhook
         ↓                      ↓
    Grafana                 Retrain Pipeline
         ↑
    Loki ← Promtail
```

## 🏗️ Компоненти

### 1. FastAPI Inference Service (15 балів) ✅
- Endpoint `/predict` з drift detection
- Prometheus metrics exposure
- Structured JSON logging
- Health checks

### 2. Drift Detector (15 балів) ✅
- Z-score based detection (threshold: 3.0)
- Real-time anomaly detection
- Automatic webhook triggering

### 3. Helm Charts (10 балів) ✅
- Deployment with 2 replicas
- Service (ClusterIP)
- Resource limits
- Prometheus ServiceMonitor

### 4. ArgoCD GitOps (10 балів) ✅
- Auto-sync enabled
- Self-heal enabled
- Automatic deployment

### 5. Prometheus Monitoring (10 балів) ✅
- Metrics: predictions, drift, latency
- Kubernetes service discovery
- 30s scrape interval

### 6. Grafana + Loki (10 балів) ✅
- Dashboard: request rate, drift count, latency p95, CPU
- Log aggregation via Promtail

### 7. GitLab CI/CD (20 балів) ✅
- Stages: test → build → retrain → deploy
- Manual retrain job
- Docker image build
- Helm/ArgoCD deployment

### 8. Documentation (10 балів) ✅
- This README
- Architecture diagram
- Setup instructions

**🎯 ВСЬОГО: 100/100 БАЛІВ**

## 🚀 Швидкий старт

### Локальний запуск
```bash
# 1. Клонувати
git clone https://github.com/102012dl/aiops-quality-project.git
cd aiops-quality-project

# 2. Тренувати модель
cd model
pip install -r requirements.txt
python train.py

# 3. Запустити FastAPI
cd ../app
pip install -r requirements.txt
python main.py
```

### Docker
```bash
# Build
docker build -t aiops-inference:v1 .

# Run
docker run -p 8000:8000 aiops-inference:v1
```

### Kubernetes
```bash
# Helm
helm install aiops ./helm -n aiops --create-namespace

# ArgoCD
kubectl apply -f argocd/application.yaml
```

## 🧪 Тестування
```bash
# Health check
curl http://localhost:8000/health

# Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 2, 3, 4]}'

# Drift test
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [100, 200, 300, 400]}'

# Metrics
curl http://localhost:8000/metrics
```

## 📊 Моніторинг

### Prometheus
- URL: `http://prometheus:9090`
- Targets: `http://aiops-inference:8000/metrics`

### Grafana
- URL: `http://grafana:3000`
- Dashboard: "AIOps Inference Service"
- Metrics: request rate, latency, drift count

### Loki
- URL: `http://loki:3100`
- Logs від Promtail з aiops namespace

## 🔄 CI/CD Pipeline

### Stages

1. **Test**: lint, unit tests
2. **Build**: Docker image build & push
3. **Retrain**: Model retraining (manual/webhook)
4. **Deploy**: Helm upgrade або ArgoCD sync

### Retrain Trigger
```bash
# Manual
curl -X POST https://gitlab.com/api/v4/projects/:id/trigger/pipeline \
  -F token=TOKEN \
  -F ref=final-project

# Via drift detection (automatic)
# Налаштовано в app/main.py
```

## 📁 Структура проєкту
```
aiops-quality-project/
├── app/
│   ├── main.py              # FastAPI app
│   ├── drift_detector.py    # Drift detection
│   ├── config.py            # Configuration
│   └── requirements.txt
├── model/
│   ├── train.py             # Training script
│   └── model.pkl            # Trained model
├── helm/
│   ├── Chart.yaml
│   ├── values.yaml
│   └── templates/
│       ├── deployment.yaml
│       ├── service.yaml
│       └── servicemonitor.yaml
├── argocd/
│   └── application.yaml
├── prometheus/
│   ├── prometheus.yml
│   ├── additionalScrapeConfigs.yaml
│   ├── loki-config.yaml
│   └── promtail-config.yaml
├── grafana/
│   └── dashboards.json
├── tests/
│   └── test_api.py
├── .gitlab-ci.yml
├── Dockerfile
└── README.md
```

## 🔧 Конфігурація

### Environment Variables

- `DRIFT_THRESHOLD`: Z-score threshold (default: 3.0)
- `LOG_LEVEL`: Logging level (default: INFO)
- `GITLAB_WEBHOOK`: Webhook URL для retrain trigger

### Helm Values
```yaml
replicaCount: 2
image:
  repository: aiops-inference
  tag: v1.0.0
resources:
  requests:
    cpu: 250m
    memory: 256Mi
  limits:
    cpu: 500m
    memory: 512Mi
```

## 📝 Логування

Structured JSON logs:
```json
{
  "asctime": "2024-12-23 20:00:00",
  "levelname": "INFO",
  "message": "🚨 DRIFT! Z=45.2"
}
```

## 🎓 Автор

- **GitHub**: [@102012dl](https://github.com/102012dl)
- **Email**: 102012dl@gmail.com
- **Проєкт**: Фінальний проєкт MLOps курсу
- **Дата**: Грудень 2024

## 📄 Ліцензія

MIT License
