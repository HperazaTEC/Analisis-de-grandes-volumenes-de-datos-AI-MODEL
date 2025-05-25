"""Train unsupervised clustering models."""
from pyspark.ml.clustering import KMeans, GaussianMixture
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from src.utils.spark import get_spark
import mlflow
from pathlib import Path
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    mlflow.spark.autolog()
    spark = get_spark("train_unsup")
    train = spark.read.parquet("data/processed/train.parquet")
    feature_cols = [c for c in train.columns if c not in {"default_flag", "weight"}]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    scaler = StandardScaler(inputCol="features", outputCol="scaled")
    models = {
        "KMeans": KMeans(featuresCol="scaled", predictionCol="cluster", k=8),
        "GMM": GaussianMixture(featuresCol="scaled", predictionCol="cluster", k=8)
    }
    Path("models/unsupervised").mkdir(parents=True, exist_ok=True)
    for name, algo in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(stages=[assembler, scaler, algo])
            model = pipeline.fit(train)
            mlflow.spark.log_model(model, f"models/unsupervised/{name}", registered_model_name=f"risk-clusters-{name}")


if __name__ == "__main__":
    main()
