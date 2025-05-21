# AGENTS Model – LendingClub Credit-Risk Project

## Objetivo

Crear un sistema de *machine learning* de punta a punta que prediga la probabilidad de incumplimiento y segmente perfiles de prestatarios usando los datos históricos de Lending Club (2007-2020 Q3). El proyecto utiliza **PySpark** para procesamiento distribuido, **MLflow** para trazabilidad de experimentos y **DVC** para control de versiones de datos/modelos, todo dentro de una arquitectura **dockerizada** basada en principios de **MLOps**. (BALANCEA LAS CLASES)

> El conjunto original contiene **≈ 2.9 M** registros y **142** variables, con \~26 % de valores nulos y outliers relevantes. fileciteturn0file0

## Fuentes de datos

| Fuente                            | Descripción                                                                                                                                               |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data/raw/lending_club.csv`       | Dataset Lending Club completo.                                                                                                                            |
| `data/processed/sample_M.parquet` | Muestra estratificada construida con 68 estratos `grade × loan_status`, siguiendo la técnica documentada en el informe de muestreo. fileciteturn0file1 |

## Arquitectura general

```
DVC → DataOps ─┐
               ├──► PySpark ETL → Feature Store → MLflow Tracking ▲
GitHub Actions ─┘                                │
                                             Docker Registry
                                                  │
                                             Serving API (FastAPI)
```

### Descripción de agentes

1. **Data-Versioning Agent (`dvc.yaml` – stage `fetch`)**
   Descarga datos brutos y los almacena en un remoto S3/MinIO.
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
7. **Pipeline-Orchestrator Agent (DVC)**
   Define el DAG `fetch → prep → split → train_sup → train_unsup → evaluate → register`.
8. **CI/CD Agent (GitHub Actions)**
   Corre `dvc repro` en cada *push*, ejecuta pruebas unitarias y publica imágenes en Docker Hub.
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
├── dvc.yaml
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
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: admin
      MINIO_ROOT_PASSWORD: admin123
  credit-risk-app:
    build: ./docker
    depends_on: [spark-master, mlflow, minio]
    ports: ["8000:8000"]
```

## Pasos de reproducción

1. `dvc pull`
2. `docker compose up -d`
3. `dvc repro`
4. Visitar **MLflow UI → [http://localhost:5000](http://localhost:5000)**.

---

*Equipo 58 – MNA | Entrega: Domingo 25 de mayo 23:59*
