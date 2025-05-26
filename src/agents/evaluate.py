"""Evaluate the best supervised model on the test set."""
from pathlib import Path
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from src.utils.spark import get_spark
from src.utils.metrics import dump_metrics
import os


def main() -> None:

    load_dotenv()

    FAST = os.getenv("FAST_MODE", "false").lower() == "true"

    spark = get_spark("evaluate")
    test = spark.read.parquet("data/processed/test.parquet")
    client = MlflowClient()
    model_name = "credit-risk"
    prod = client.get_latest_versions(model_name, stages=["None", "Production"])[0]
    model = mlflow.spark.load_model(prod.source)
    preds = model.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol="default_flag")
    auc = evaluator.evaluate(preds)
    dump_metrics("evaluate", {"auc_test": auc, "fast": FAST})
    mlflow.log_metric("test_auc", auc)


if __name__ == "__main__":
    main()
