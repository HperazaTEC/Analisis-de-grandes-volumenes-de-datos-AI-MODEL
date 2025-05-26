# Informe Resumen del Proyecto de Riesgo Crediticio

## Introducción

El presente proyecto implementa un pipeline completo de aprendizaje automatizado para evaluar el riesgo crediticio en los datos históricos de LendingClub. Como se describe en el repositorio, el objetivo es crear un sistema reproducible capaz de manejar millones de registros con PySpark y registrar todos los experimentos mediante MLflow.

## Desarrollo

### Descripción del dataset

El conjunto original comprende aproximadamente 2.9 millones de préstamos con 141 variables, cubriendo el periodo de enero de 2007 a marzo de 2020. La tabla de bloques de variables en `README.md` detalla ejemplos de atributos relacionados con el préstamo, la situación laboral, el historial crediticio y otras categorías relevantes【F:README.md†L10-L33】. Entre los principales retos destacan los altos porcentajes de valores faltantes y la presencia de outliers, así como la desbalanceada distribución del estado de los créditos【F:README.md†L34-L42】.

### Flujo de trabajo

1. **Ingesta y preparación**: `fetch.py` descarga los datos desde Kaggle, mientras que `prep.py` realiza la limpieza y genera el conjunto procesado `M.parquet` con la variable `default_flag`【F:README.md†L122-L135】.
2. **División de datos**: `split.py` produce particiones estratificadas en proporción 80/20 para entrenamiento y prueba, manteniendo la distribución de `grade` y `loan_status`【F:README.md†L174-L183】.
3. **Modelado supervisado**: se entrenan RandomForest, GBT y MLP para predecir `default_flag`. El proceso registra métricas como AUC, precisión y recall en MLflow【F:src/agents/train_sup.py†L160-L213】.
4. **Modelado no supervisado**: se aplican K-Means y GaussianMixture con las características resultantes de la preparación de datos, almacenando los modelos y su inercia en MLflow【F:src/agents/train_unsup.py†L1-L67】.
5. **Evaluación y registro**: el mejor modelo supervisado se evalúa sobre el conjunto de prueba y su AUC se guarda mediante `evaluate.py`, después se registra en el Model Registry con `register.py`【F:src/agents/evaluate.py†L1-L29】【F:src/agents/register.py†L1-L15】.

### Resultados

Los experimentos registrados muestran que los algoritmos supervisados alcanzan métricas de desempeño calculadas en el código de entrenamiento, destacando el AUC como medida principal. Los modelos no supervisados generan agrupamientos que permiten segmentar a los solicitantes en perfiles de riesgo. Todas estas ejecuciones quedan almacenadas en MLflow para su posterior análisis.

## Conclusión

El proyecto establece una arquitectura MLOps para procesar grandes volúmenes de datos crediticios. Mediante un pipeline modular basado en PySpark se gestionan la ingesta, el preprocesamiento y la generación de conjuntos de entrenamiento y prueba. Los modelos supervisados (RandomForest, GBT y MLP) y no supervisados (K-Means y GaussianMixture) proporcionan una aproximación integral al problema, dejando los resultados disponibles en MLflow y facilitando su despliegue mediante una API FastAPI.
