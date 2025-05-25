"""Train supervised models on default_flag."""
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier, MultilayerPerceptronClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from src.utils.spark import get_spark
from src.utils.balancing import add_weight_column
import mlflow
import os
from pathlib import Path
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    os.environ.setdefault("PYSPARK_PIN_THREAD", "false")
    try:
        mlflow.spark.autolog()
    except Exception as e:
        print(f"mlflow spark autologging disabled: {e}")
    spark = get_spark("train_sup")
    train = spark.read.parquet("data/processed/train.parquet")
    test = spark.read.parquet("data/processed/test.parquet")
    target = "default_flag"

    # Basic preprocessing
    cat_cols = [c for c, t in train.dtypes if t == "string" and c != target]
    num_cols = [c for c, t in train.dtypes if t != "string" and c != target]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
    encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh") for c in cat_cols]
    assembler = VectorAssembler(
        inputCols=num_cols + [f"{c}_oh" for c in cat_cols],
        outputCol="features"
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
            pipeline = Pipeline(stages=indexers + encoders + [assembler, algo])
            model = pipeline.fit(train)
            preds = model.transform(test)
            auc = evaluator.evaluate(preds)
            mlflow.log_metric("auc", auc)
            mlflow.spark.log_model(model, f"models/supervised/{name}", registered_model_name="credit-risk")


if __name__ == "__main__":
    main()
