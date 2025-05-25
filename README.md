# Analisis-de-grandes-volumenes-de-datos-AI-MODEL

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

1. **Ingesta y versionado**: descarga desde Kaggle mediante `src/agents/fetch.py` y guarda en `data/raw/`. El control de versiones se realiza con DVC.
2. **Validación**: se definen esquemas con Great Expectations y la CI falla si el contrato se rompe.
3. **Feature engineering**: imputaciones, codificación de variables categóricas y escalado de numéricas. Se generan atributos derivados como `loan_to_income` o `credit_age`.
4. **Partición temporal**: se filtran préstamos previos a 2012 y se dividen en train (2007‑2017), validación (2018) y test (2019‑Q1 2020) para evitar fugas.
5. **Manejo del desbalance**: estratificación y cálculo de pesos de clase (ver `src/utils/balancing.py`).
6. **Entrenamiento**: modelos base de regresión logística y árboles (RandomForest, GBT), además de redes neuronales sencillas. Todo se registra con MLflow.
7. **Registro y CI/CD**: el mejor modelo se registra en el MLflow Model Registry. GitHub Actions ejecuta las pruebas con PyTest y reproduce el pipeline con DVC.
8. **Despliegue**: la API FastAPI contenida en `docker/` se expone en `http://localhost:8000/predict`.
9. **Monitorización**: Prometheus y Grafana para métricas de servicio; EvidentlyAI para *data drift*.

## Uso rápido

```bash
# 1. Crear entorno virtual e instalar dependencias
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurar credenciales de Kaggle
cp .env.example .env  # editar con tu API key y dataset

# 3. (Opcional) obtener datos cacheados y levantar los servicios
#    Esto inicia Spark, MLflow y MinIO en contenedores Docker
#    y deja MLflow escuchando en http://localhost:5000

dvc pull
docker compose -f docker/docker-compose.yml up -d

# 4. Ejecutar el pipeline completo
#    Descarga la data (si no existe), prepara, divide y entrena modelos

dvc repro

# 5. Correr las pruebas unitarias
pytest -q
```

Al finalizar, la interfaz de MLflow estará disponible en `http://localhost:5000` y la API de predicción en `http://localhost:8000/predict`.
