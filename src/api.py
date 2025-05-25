from fastapi import FastAPI
import mlflow.spark
from pyspark.sql import SparkSession

app = FastAPI()
model = None


@app.on_event("startup")
def load_model():
    global model
    try:
        model = mlflow.spark.load_model("models:/credit-risk/Production")
    except Exception:
        model = None


@app.post("/predict")
def predict(features: dict):
    if model is None:
        return {"error": "model not loaded"}
    import pandas as pd
    df = pd.DataFrame([features])
    spark = mlflow.spark.get_active_session() or SparkSession.builder.getOrCreate()
    sdf = spark.createDataFrame(df)
    preds = model.transform(sdf).toPandas()
    return {"prediction": preds['prediction'].iloc[0]}
