"""Train KMeans clustering model with robust preprocessing."""

import json
import os

# Disable dataset autologging noise from mlflow-spark
os.environ["MLFLOW_ENABLE_SPARK_DATASET_AUTOLOGGING"] = "false"

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import (
    OneHotEncoder,
    StandardScaler,
    StringIndexer,
    VectorAssembler,
)
from src.utils.spark import get_spark
from src.utils.metrics import METRICS_DIR
import mlflow
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    fast = os.getenv("FAST_MODE", "false").lower() == "true"
    sample_frac = float(os.getenv("SAMPLE_FRACTION", "0.05"))

    mlflow.spark.autolog()

    spark = get_spark("train_unsup")
    spark.conf.set("spark.sql.debug.maxToStringFields", 50)
    spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)

    df = spark.read.parquet("data/processed/M.parquet")
    if fast:
        df = df.sample(fraction=sample_frac, seed=42)

    num_cols = [c for c, t in df.dtypes if t != "string"]
    cat_cols = [c for c, t in df.dtypes if t == "string"]

    indexers = [
        StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
        for c in cat_cols
    ]
    encoder = OneHotEncoder(
        inputCols=[f"{c}_idx" for c in cat_cols],
        outputCols=[f"{c}_ohe" for c in cat_cols],
        dropLast=False,
    )
    assembler = VectorAssembler(
        inputCols=num_cols + [f"{c}_ohe" for c in cat_cols],
        outputCol="raw_feats",
    )
    scaler = StandardScaler(inputCol="raw_feats", outputCol="features")
    kmeans = KMeans(k=8, seed=42, featuresCol="features", predictionCol="cluster")

    pipeline = Pipeline(stages=indexers + [encoder, assembler, scaler, kmeans])

    with mlflow.start_run(run_name="kmeans"):
        model = pipeline.fit(df)
        mlflow.spark.log_model(
            model,
            "models/unsupervised/kmeans",
            registered_model_name="credit-risk-segmentation",
        )
        inertia = model.stages[-1].summary.trainingCost
        metrics = {"k": 8, "inertia": float(inertia), "rows": df.count(), "fast": fast}

        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        with (METRICS_DIR / "kmeans_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_metrics({"k": 8, "inertia": metrics["inertia"], "rows": metrics["rows"]})

    spark.stop()


if __name__ == "__main__":
    main()
