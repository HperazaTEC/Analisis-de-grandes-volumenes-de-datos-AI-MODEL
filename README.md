# Analisis-de-grandes-volumenes-de-datos-AI-MODEL

This repository contains the source code for the LendingClub credit-risk project described in `AGENTS.md`.
The project implements a full MLOps pipeline using PySpark, DVC and MLflow.
Raw data is automatically downloaded from Kaggle using the credentials
defined in a `.env` file.
Exploratory notebooks used during development are kept under the `notebooks/`
directory.

## Quickstart

```bash
# Pull the data and start services

$ cp .env.example .env   # edit with your Kaggle credentials

$ dvc pull
$ docker compose -f docker/docker-compose.yml up -d

# Run the pipeline
$ dvc repro
```

The FastAPI service will be available at `http://localhost:8000/predict`.
