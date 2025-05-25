# Analisis de Grandes Volúmenes de Datos - Modelo de Riesgo Crediticio

Este proyecto implementa un pipeline completo de **MLOps** para predecir el incumplimiento de préstamos de *LendingClub* y segmentar a los solicitantes en grupos de riesgo. El flujo se ejecuta con **PySpark** para el procesamiento distribuido, **MLflow** para el seguimiento de experimentos y **DVC** para el versionado de datos y modelos.


## Objetivo del proyecto

- Construir un sistema reproducible que descargue los datos históricos de LendingClub (2007‑2020 Q3).
- Limpiar, balancear y transformar la información para entrenar modelos supervisados y no supervisados.
- Registrar experimentos y almacenar el mejor modelo en un *Model Registry*.
- Exponer un servicio REST mediante FastAPI para realizar predicciones.

## Secciones del pipeline

1. **fetch** – Descarga el conjunto original desde Kaggle utilizando las credenciales definidas en `.env`.
2. **prep** – Limpieza de valores nulos y atípicos, conversión de tipos y generación de `data/processed/M.parquet`.
3. **split** – Genera una partición estratificada 80/20 (train/test) con semilla fija para asegurar la reproducibilidad.
4. **train_sup** – Entrena modelos supervisados (RandomForest, GBT y MLP) con balanceo de clases.
5. **train_unsup** – Ejecuta algoritmos de clúster (K-Means y GaussianMixture) para segmentar perfiles de riesgo.
6. **evaluate** – Calcula métricas de desempeño y guarda los resultados en MLflow.
7. **register** – Registra en el *Model Registry* el modelo con mejor desempeño para su despliegue.

El archivo `dvc.yaml` orquesta estos pasos en la secuencia:
`fetch → prep → split → train_sup → train_unsup → evaluate → register`.

## Guía rápida

```bash

# 1. Crear un entorno virtual e instalar dependencias

$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt

# 2. Configurar las credenciales de Kaggle
$ cp .env.example .env  # edite este archivo con su API key y dataset

# 3. (Opcional) descargar los datos almacenados y arrancar los servicios locales
$ dvc pull
$ docker compose -f docker/docker-compose.yml up -d

# 4. Ejecutar el pipeline completo
$ dvc repro

# 5. Lanzar las pruebas unitarias
$ pytest -q
```

La API FastAPI quedará disponible en `http://localhost:8000/predict`.

## Estructura principal del repositorio

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
├── dvc.yaml
└── AGENTS.md


Consulte `AGENTS.md` para una descripción detallada de cada agente y de la arquitectura general.
