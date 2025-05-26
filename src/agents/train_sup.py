"""Train supervised models on default_flag."""
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import FeatureHasher
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from src.utils.spark import get_spark
from src.utils.balancing import add_weight_column
import mlflow
import os
from pathlib import Path
from dotenv import load_dotenv
from py4j.protocol import Py4JJavaError
import logging
import sys
import re



def main() -> int:
    load_dotenv()

    try:
        mlflow.spark.autolog()
    except Exception as e:
        print(f"MLflow autologging not available: {e}")

    seed = 42

    spark = get_spark("train_sup")
    spark.conf.set("spark.python.worker.broadcastTimeout", "600")
    try:
        train = spark.read.parquet("data/processed/train.parquet")
        test = spark.read.parquet("data/processed/test.parquet")
        target = "default_flag"

        train = train.cache()
        test = test.cache()
    
        # Auto-detect available driver memory (GB) and compute batch factor
        mem_str = os.environ.get("SPARK_DRIVER_MEMORY", "6")
        digits = re.findall(r"\d+(?:\.\d+)?", mem_str)
        driver_mem_gb = float(digits[0]) if digits else 6.0
        max_rows = int(1_000_000 * (driver_mem_gb / 6))
        total_rows = train.count()
        if total_rows > max_rows:
            fraction = max_rows / total_rows
            train = train.sample(fraction=fraction, seed=42)
            print(
                f"Downsampled train from {total_rows} to {max_rows} rows (driver_mem={driver_mem_gb}G)"
            )
    
        # Basic preprocessing
        num_cols = [c for c, t in train.dtypes if t != "string" and c != target]
        cat_cols = [c for c, t in train.dtypes if t == "string" and c != target]
        train = train.fillna(0.0, subset=num_cols)
        test = test.fillna(0.0, subset=num_cols)
        hasher = FeatureHasher(
            inputCols=cat_cols + num_cols,
            outputCol="features",
            numFeatures=2 ** 14,
        )
    
        train = add_weight_column(train, target)
        models = {
            "RandomForest": RandomForestClassifier(labelCol=target, featuresCol="features", weightCol="weight"),
            "GBT": GBTClassifier(labelCol=target, featuresCol="features", weightCol="weight"),
            "MLP": MultilayerPerceptronClassifier(labelCol=target, featuresCol="features",
                                                   layers=[len(num_cols) + len(cat_cols), 10, 2])
        }
    
        evaluator = BinaryClassificationEvaluator(labelCol=target)
        Path("models/supervised").mkdir(parents=True, exist_ok=True)
        for name, algo in models.items():
            with mlflow.start_run(run_name=name):
                pipeline = Pipeline(stages=[hasher, algo])
                retry_fraction = 1.0
                while True:
                    subset = train if retry_fraction >= 1.0 else train.sample(fraction=retry_fraction, seed=seed)
                    subset = subset.cache()
                    try:
                        model = pipeline.fit(subset)
                        break
                    except (Py4JJavaError, MemoryError) as e:
                        if "java.lang.OutOfMemoryError" in str(e):
                            if retry_fraction <= 0.01:
                                logging.critical("Model training failed due to OOM")
                                return 1
                            retry_fraction *= 0.5
                            print(
                                f"OOM detected → retrying fit with fraction={retry_fraction}"
                            )
                        else:
                            logging.critical(str(e))
                            return 1

                preds = model.transform(test)
                auc = evaluator.evaluate(preds)
                mlflow.log_metric("auc", auc)
                mlflow.spark.log_model(model, f"models/supervised/{name}", registered_model_name="credit-risk")


        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
