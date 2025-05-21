"""Evaluate the best supervised model on the test set."""
from pathlib import Path
import mlflow
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from utils.spark import get_spark


def main() -> None:

    load_dotenv()

    spark = get_spark("evaluate")
    test = spark.read.parquet("data/processed/test.parquet")
    client = MlflowClient()
    model_name = "credit-risk"
    prod = client.get_latest_versions(model_name, stages=["None", "Production"])[0]
    model = mlflow.spark.load_model(prod.source)
    preds = model.transform(test)
    evaluator = BinaryClassificationEvaluator(labelCol="default_flag")
    auc = evaluator.evaluate(preds)
    Path("metrics").mkdir(parents=True, exist_ok=True)
    with open("metrics/evaluation.json", "w") as f:
        f.write('{"auc": %.4f}' % auc)
    mlflow.log_metric("test_auc", auc)


if __name__ == "__main__":
    main()
