# Analisis-de-grandes-volumenes-de-datos-AI-MODEL

This repository contains the source code for the LendingClub credit-risk
project described in `AGENTS.md`.  It implements a full MLOps pipeline using
**PySpark**, **DVC** and **MLflow**.  Raw data is downloaded from Kaggle using
the credentials stored in a `.env` file.  Exploratory notebooks used during
development live in the `notebooks/` directory.

## Quickstart

```bash
# 1. Create a virtual environment and install dependencies
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt

# 2. Configure Kaggle credentials
$ cp .env.example .env   # edit this file with your API key and dataset

# 3. (Optional) fetch cached data and start the local services
$ dvc pull             # downloads data from the configured remote
$ docker compose -f docker/docker-compose.yml up -d

# 4. Execute the pipeline end-to-end
$ dvc repro

# 5. Run the unit tests
$ pytest -q
```

The FastAPI service will be available at `http://localhost:8000/predict`.
