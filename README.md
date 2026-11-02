# Credit Risk MLOps

End-to-end MLOps project that predicts whether a loan applicant is a **good or bad credit risk**, using the German Credit dataset (OpenML `credit-g`).

**Live app:** https://credit-risk-gaurang123-btaffvh5dfd7anb4.centralindia-01.azurewebsites.net/

## Pipeline
| Stage | Tools |
|---|---|
| Planning | Scrum backlog (`docs/backlog.md`) |
| Environment and version control | uv, Git, GitHub (feature branches + PRs) |
| Data | ETL + feature store (`src/etl.py`), EDA (`notebooks/eda.ipynb`) |
| Training | scikit-learn pipelines, MLflow tracking, quality gate (`src/train.py`) |
| Serving | Flask API + web UI, gunicorn (`src/app.py`) |
| CI | GitHub Actions: lint, train, test, Docker build |
| Containers | Docker, Kubernetes (`k8s/`) |
| Cloud | Azure Container Registry + App Service |
| Monitoring | Prometheus + Grafana (`docker-compose.yml`) |

**Best model:** Logistic Regression, with accuracy 0.755, F1 0.667 and ROC-AUC 0.807.

## Quick start
```powershell
uv sync
uv run python -m src.etl
uv run python -m src.train
uv run pytest
uv run python -m src.app        # http://localhost:5000
```

## Docker / Kubernetes / Monitoring
```powershell
docker build -t credit-risk-api:v1 .
kubectl apply -f k8s/
docker compose up -d            # API :5000, Prometheus :9090, Grafana :3000
```

## API
- `GET /` web UI
- `GET /health` health check
- `POST /predict` JSON applicant data, returns `risk` and `probability_bad`
- `GET /metrics` Prometheus metrics
