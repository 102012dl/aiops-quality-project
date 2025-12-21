# AIOps Quality Monitoring System

Фінальний проєкт з MLOps - система моніторингу якості ML-моделі.

## 🏗️ Архітектура

```
Client → FastAPI → ML Model → Drift Detector → GitLab CI → ArgoCD
```

## 🚀 Швидкий старт

```bash
# Клонування
git clone https://github.com/102012dl/aiops-quality-project.git
cd aiops-quality-project

# Локальний запуск
python -m venv venv
source venv/bin/activate
pip install -r app/requirements.txt
python app/main.py
```

## 📦 Компоненти

- FastAPI inference сервіс
- Drift detection
- Prometheus + Grafana моніторинг
- Loki логування
- ArgoCD GitOps
- GitLab CI retrain pipeline

## 👤 Автор

- **GitHub**: [@102012dl](https://github.com/102012dl)
- **Email**: 102012dl@gmail.com
- **Проєкт**: Фінальний проєкт MLOps курсу

## 📅 Дата створення

2025-12-21

## 📄 Ліцензія

MIT License
