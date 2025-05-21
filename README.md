# Analisis-de-grandes-volumenes-de-datos-AI-MODEL

This repository contains the source code for the LendingClub credit-risk project described in `AGENTS.md`.
The project implements a full MLOps pipeline using PySpark, DVC and MLflow.

## Quickstart

```bash
# Pull the data and start services
$ dvc pull
$ docker compose -f docker/docker-compose.yml up -d

# Run the pipeline
$ dvc repro
```

The FastAPI service will be available at `http://localhost:8000/predict`.
