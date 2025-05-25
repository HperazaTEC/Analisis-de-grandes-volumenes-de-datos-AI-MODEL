# Analisis de Grandes Volúmenes de Datos - Modelo de Riesgo Crediticio


Este repositorio contiene el código fuente del proyecto de riesgo crediticio con datos de LendingClub descrito en `AGENTS.md`. El objetivo es construir un pipeline MLOps completo que procese millones de registros en PySpark, entrene modelos supervisados y no supervisados y registre todos los experimentos en MLflow.


## Descripción del dataset

* **Cobertura temporal:** enero 2007 a marzo 2020.
* **Observaciones:** ~2.9 millones de préstamos personales.
* **Columnas originales:** 141 atributos heterogéneos.
* **Tamaño:** ~1.7&nbsp;GB comprimido en Kaggle.
* **Atributo objetivo típico:** `loan_status` (Fully Paid, Charged Off, Default...).
* **Valores faltantes:** más del 25&nbsp;% global (algunas columnas superan el 40&nbsp;%).

### Bloques lógicos de variables más frecuentes

| Bloque | Ejemplos | Comentarios |
| ------ | -------- | ----------- |
| **1. Datos del préstamo** | `loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `sub_grade` | Definen montos, plazos y pricing. |
| **2. Demográficos y empleo** | `emp_title`, `emp_length`, `home_ownership`, `annual_inc` | Texto libre y categorías; `annual_inc` tiene *outliers*. |
| **3. Historial crediticio** | `fico_range_low/high`, `dti`, `open_acc`, `revol_util`, `earliest_cr_line` | Mezcla de numéricas y fechas. `earliest_cr_line` se transforma a año. |
| **4. Comportamiento de pago** | `total_pymnt`, `total_rec_prncp`, `recoveries`, `last_pymnt_d` | Solo conocidas después de la originación → riesgo de *data leakage*. |
| **5. Hardship / collections** | `hardship_flag`, `hardship_type`, `collection_recovery_fee` | Muy escasas (> 70&nbsp;% nulos). |
| **6. Co‑applicant** | `sec_app_fico_range_low/high`, `annual_inc_joint` | Solo existen cuando hay solicitante secundario. |
| **7. Metadatos y fechas** | `issue_d`, `last_credit_pull_d`, `next_pymnt_d` | Útiles para particiones por fecha. |

### Retos de calidad de datos

- `dti` llega a 999 → marcador de error.
- Alta cardinalidad en `emp_title` (~60&nbsp;k valores distintos).
- *Column drift*: antes de 2012 faltan muchos campos nuevos.
- Clase desbalanceada (~20&nbsp;% `Charged Off` vs 80&nbsp;% `Fully Paid`).
- Fechas almacenadas como texto (`issue_d`, `earliest_cr_line`).

## Pipeline MLOps propuesto

1. **Ingesta**: descarga desde Kaggle mediante `src/agents/fetch.py` y guarda en `data/raw/`.
2. **Validación**: se definen esquemas con Great Expectations y la CI falla si el contrato se rompe.
3. **Feature engineering**: imputaciones, codificación de variables categóricas y escalado de numéricas. Se generan atributos derivados como `loan_to_income` o `credit_age`.
4. **Partición temporal**: se filtran préstamos previos a 2012 y se dividen en train (2007‑2017), validación (2018) y test (2019‑Q1 2020) para evitar fugas.
5. **Manejo del desbalance**: estratificación y cálculo de pesos de clase (ver `src/utils/balancing.py`).
6. **Entrenamiento**: modelos base de regresión logística y árboles (RandomForest, GBT), además de redes neuronales sencillas. Todo se registra con MLflow.
7. **Registro y CI/CD**: el mejor modelo se registra en el MLflow Model Registry. GitHub Actions ejecuta las pruebas con PyTest.
8. **Despliegue**: la API FastAPI contenida en `docker/` se expone en `http://localhost:8000/predict`.
9. **Monitorización**: Prometheus y Grafana para métricas de servicio; EvidentlyAI para *data drift*.


Este proyecto implementa un pipeline completo de **MLOps** para predecir el incumplimiento de préstamos de *LendingClub* y segmentar a los solicitantes en grupos de riesgo. El flujo se ejecuta con **PySpark** para el procesamiento distribuido y **MLflow** para el seguimiento de experimentos.


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

Estos pasos deben ejecutarse en el siguiente orden:
`fetch → prep → split → train_sup → train_unsup → evaluate → register`.


Al finalizar, la interfaz de MLflow estará disponible en `http://localhost:5000` y la API de predicción en `http://localhost:8000/predict`.

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
└── AGENTS.md
```

Consulte `AGENTS.md` para una descripción detallada de cada agente y de la arquitectura general.

## Descripción de scripts y funciones

- **`src/agents/fetch.py`**
  - `main()` descarga el dataset desde Kaggle utilizando las variables definidas en `.env` y lo guarda en `data/raw/`.
- **`src/agents/prep.py`**
  - `winsorize(df, cols, lower, upper)` recorta los extremos de columnas numéricas.
  - `main()` limpia nulos, castea tipos, crea `default_flag` y genera `M.parquet` junto con `sample_M.parquet`.
- **`src/agents/split.py`**
  - `stratified_split(df, strat_cols, test_frac, seed)` produce divisiones de entrenamiento y prueba estratificadas.
  - `main()` escribe `train.parquet` y `test.parquet` en `data/processed/`.
- **`src/agents/train_sup.py`**
  - `main()` entrena modelos supervisados (RandomForest, GBT y MLP) aplicando pesos de clase y registra los experimentos en MLflow.
- **`src/agents/train_unsup.py`**
  - `main()` ajusta algoritmos de clústeres (KMeans y GaussianMixture) y guarda los modelos resultantes.
- **`src/agents/evaluate.py`**
  - `main()` carga el mejor modelo supervisado y calcula el AUC sobre el conjunto de prueba.
- **`src/agents/register.py`**
  - `main()` registra en el *Model Registry* la ejecución con mayor métrica obtenida.
- **`src/api.py`**
  - `load_model()` se ejecuta al iniciar la API y recupera el modelo en producción desde MLflow.
  - `predict()` expone el endpoint `/predict` para generar inferencias.
- **`src/utils/balancing.py`**
  - `compute_class_weights()` calcula los pesos para balancear la clase binaria.
  - `add_weight_column()` agrega al `DataFrame` una columna de pesos.
- **`src/utils/spark.py`**
  - `get_spark()` crea la sesión de Spark reutilizada por todos los agentes.

## Guía rápida de uso

1. Clonar este repositorio y crear un entorno virtual de Python.
2. Instalar las dependencias con `pip install -r requirements.txt`.
3. Copiar `\.env.example` a `\.env` y completar las credenciales de Kaggle.
4. Levantar los servicios locales con `docker compose up -d`.
5. Ejecutar secuencialmente:

   `python -m src.agents.fetch`,
   `python -m src.agents.prep`,
   `python -m src.agents.split`,
   `python -m src.agents.train_sup`,
   `python -m src.agents.train_unsup`,
   `python -m src.agents.evaluate`,
   `python -m src.agents.register`.
6. Acceder a la interfaz de MLflow en `http://localhost:5000` y a la API de predicción en `http://localhost:8000/predict`.
7. Para verificar el código ejecutar `pytest`.


7. Acceder a la interfaz de MLflow en `http://localhost:5000` y a la API de predicción en `http://localhost:8000/predict`.
8. Para verificar el código ejecutar `pytest`.


