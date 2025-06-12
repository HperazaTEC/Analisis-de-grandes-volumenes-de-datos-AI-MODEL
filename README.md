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

## Instalación

Este proyecto requiere **Python 3.10+** y las dependencias listadas en
`requirements.txt`. Para ejecutar las pruebas unitarias es necesario
contar con `pyspark`. Puede instalarse manualmente mediante
`pip install pyspark` o bien utilizando el archivo
`requirements-dev.txt` incluido para facilitar la configuración local.


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

## Actividad 4

El notebook de métricas se encuentra en [`notebooks/Metricas_A01795125.ipynb`](notebooks/Metricas_A01795125.ipynb). Para ejecutarlo con las dependencias del proyecto:

1. Instale el entorno local con `pip install -r requirements.txt`.
2. Inicie Jupyter con `jupyter notebook` o `jupyter lab` desde la raíz del repositorio.
3. Abra el archivo `Metricas_A01795125.ipynb` y ejecute las celdas en orden.
El archivo `requirements.txt` ya incluye `pyspark`, por lo que no es necesario
instalarlo de manera independiente. Dicho paquete es necesario tanto para
ejecutar las pruebas unitarias como para correr los notebooks de la **Actividad
4**. De forma opcional se provee `requirements-dev.txt` únicamente como un
archivo de conveniencia para recrear el entorno de desarrollo.
