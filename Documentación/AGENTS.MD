# AGENTS Model – LendingClub Credit-Risk Project

## Objetivo

Crear un sistema de *machine learning* de punta a punta que prediga la probabilidad de incumplimiento y segmente perfiles de prestatarios usando los datos históricos de Lending Club (2007-2020 Q3). El proyecto utiliza **PySpark** para procesamiento distribuido y **MLflow** para trazabilidad de experimentos, todo dentro de una arquitectura **dockerizada** basada en principios de **MLOps**. (BALANCEA LAS CLASES)

> El conjunto original contiene **≈ 2.9 M** registros y **142** variables, con \~26 % de valores nulos y outliers relevantes. fileciteturn0file0

## Fuentes de datos

| Fuente                            | Descripción                                                                                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/raw/Loan_status_2007-2020Q3.gzip`       | Dataset Lending Club completo descargado desde Kaggle.                                                                                                                            |
| `data/processed/sample_M.parquet` | Muestra estratificada construida con 68 estratos `grade × loan_status`, siguiendo la técnica documentada en el informe de muestreo. fileciteturn0file1 |

## Arquitectura general

```
DataOps ─┐
          ├──► PySpark ETL → Feature Store → MLflow Tracking ▲
GitHub Actions ─┘                                │
                                             Docker Registry
                                                  │
                                             Serving API (FastAPI)
```

### Descripción de agentes

1. **Data-Fetch Agent (`src/agents/fetch.py`)**
   Descarga datos brutos desde Kaggle y los guarda localmente en `data/raw/`.
2. **Data-Preparation Agent (`src/agents/prep.py`)**
   • Limpieza de nulos, winsorización de outliers y *type casting*.
   • Genera `data/processed/M.parquet`.
3. **Sampling-Split Agent (`src/agents/split.py`)**
   Estratifica por `grade` y `loan_status`, reserva 80-20 train/test con semilla reproducible.
4. **Supervised-Learning Agent (`src/agents/train_sup.py`)**
   Entrena RandomForest, GBT y MLP sobre la etiqueta binaria `default_flag`.
5. **Unsupervised-Learning Agent (`src/agents/train_unsup.py`)**
   Ejecuta K-Means y GaussianMixture para clústeres de riesgo.
6. **Experiment-Tracking Agent (MLflow)**
   Autolog de métricas, artefactos y parámetros; registra el mejor modelo en la *Model Registry*.
7. **Pipeline-Orchestrator Agent (Makefile)**
   Ejecuta en secuencia `fetch → prep → split → train_sup → train_unsup → evaluate → register`.
8. **CI/CD Agent (GitHub Actions)**
   Ejecuta el pipeline completo y las pruebas unitarias en cada *push*; publica imágenes en Docker Hub.
9. **Deployment Agent (`docker/`)**
   Contiene `Dockerfile` y `docker-compose.yml`; expone un endpoint `/predict` vía FastAPI.

## Estructura de carpetas

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── agents/
│   ├── pipelines/
│   └── utils/
├── models/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── mlruns/
└── AGENTS.md
```

## Cumplimiento de *Actividad 3*

| Sección rúbrica          | Agente responsable                        |
| ------------------------ | ----------------------------------------- |
| Introducción teórica     | Contenido inicial de *AGENTS.md*          |
| Selección de los datos   | Data-Versioning & Preparation Agents      |
| Preparación de los datos | Data-Preparation Agent                    |
| Entrenamiento / prueba   | Sampling-Split Agent                      |
| Modelos sup/unsup        | Supervised & Unsupervised Agents + MLflow |

## Servicios Docker (fragmento)

```yaml
version: "3.9"
services:
  spark-master:
    image: bitnami/spark:latest
    environment:
      - SPARK_MODE=master
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    ports: ["5000:5000"]
  credit-risk-app:
    build: ./docker
    depends_on: [spark-master, mlflow]
    ports: ["8000:8000"]
```

## Pasos de reproducción

1. `docker compose up -d`
2. Ejecutar el pipeline con `make pipeline` o correr manualmente los agentes en el orden indicado.
3. Visitar **MLflow UI → [http://localhost:5000](http://localhost:5000)**.

---

*Equipo 58 – MNA | Entrega: Domingo 25 de mayo 23:59*
