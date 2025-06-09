"""Train supervised models on default_flag."""
import os

# Disable noisy Spark autologging of datasets
os.environ["MLFLOW_ENABLE_SPARK_DATASET_AUTOLOGGING"] = "false"

    from pyspark.ml.classification import (
        RandomForestClassifier,
        GBTClassifier,
        MultilayerPerceptronClassifier,
        LogisticRegression,
    )
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
)
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from src.utils.spark import get_spark
from src.utils.balancing import add_weight_column
from src.utils.metrics import METRICS_DIR
import mlflow
from pathlib import Path
from dotenv import load_dotenv
from py4j.protocol import Py4JJavaError
import logging
import sys
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
import seaborn as sns


def eval_metrics(preds, label_col="default_flag"):
    """Compute common classification metrics for a predictions DataFrame."""
    pdf = preds.select(label_col, "prediction", "probability").toPandas()
    y_true = pdf[label_col].astype(int)
    y_pred = pdf["prediction"].astype(int)
    probs = np.array(pdf["probability"].tolist())[:, 1]
    return {
        "auc": float(roc_auc_score(y_true, probs)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, probs)),
    }


def compute_learning_curve(pipeline, train_df, val_df, fractions, seed=42):
    rows = []
    for frac in fractions:
        subset = train_df if frac >= 1.0 else train_df.sample(fraction=frac, seed=seed)
        subset = subset.cache()
        model = pipeline.fit(subset)
        train_metrics = eval_metrics(model.transform(subset))
        val_metrics = eval_metrics(model.transform(val_df))
        rows.append({"fraction": frac, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}})
        subset.unpersist()
    return pd.DataFrame(rows)



def main() -> int:
    load_dotenv()

    try:
        mlflow.spark.autolog()
    except Exception as e:
        print(f"MLflow autologging not available: {e}")

    FAST = os.getenv("FAST_MODE", "false").lower() == "true"
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

        train = add_weight_column(train, target)

        # Split into train/validation
        train_df, val_df = train.randomSplit([0.8, 0.2], seed=seed)

        # Preprocessing pipeline: StringIndexer -> OneHotEncoder -> VectorAssembler
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
            outputCol="features",
        )
        prep_pipeline = Pipeline(stages=indexers + [encoder, assembler])
        prep_model = prep_pipeline.fit(train_df)
        train_pre = prep_model.transform(train_df)

        n_features = train_pre.select("features").first()["features"].size

        models = {
            "LogisticRegression": LogisticRegression(
                labelCol=target,
                featuresCol="features",
                weightCol="weight",
            ),
            "RandomForest": RandomForestClassifier(
                labelCol=target,
                featuresCol="features",
                weightCol="weight",
                seed=seed,
            ),
            "GBT": GBTClassifier(
                labelCol=target,
                featuresCol="features",
                weightCol="weight",
                seed=seed,
            ),
        }
        if not FAST:
            layers = [n_features, 64, 32, 2]
            models["MLP"] = MultilayerPerceptronClassifier(
                labelCol=target,
                featuresCol="features",
                layers=layers,
                weightCol="weight",
                seed=seed,
            )

        results = []
        evaluator = BinaryClassificationEvaluator(labelCol=target)
        Path("models/supervised").mkdir(parents=True, exist_ok=True)
        for name, algo in models.items():
            with mlflow.start_run(run_name=name) as run:
                pipeline = Pipeline(stages=[prep_model, algo])
                if name == "RandomForest":
                    param_grid = (
                        ParamGridBuilder()
                        .addGrid(algo.maxDepth, [5, 10])
                        .addGrid(algo.numTrees, [20, 50])
                        .build()
                    )
                elif name == "GBT":
                    param_grid = (
                        ParamGridBuilder()
                        .addGrid(algo.maxDepth, [5, 10])
                        .addGrid(algo.maxIter, [20, 50])
                        .addGrid(algo.stepSize, [0.05, 0.1])
                        .build()
                    )
                else:
                    param_grid = []

                cv = (
                    CrossValidator(
                        estimator=pipeline,
                        estimatorParamMaps=param_grid,
                        evaluator=evaluator,
                        numFolds=3,
                    )
                    if param_grid
                    else None
                )

                retry_fraction = 1.0
                while True:
                    subset = (
                        train_df
                        if retry_fraction >= 1.0
                        else train_df.sample(fraction=retry_fraction, seed=seed)
                    )
                    subset = subset.cache()
                    try:
                        if cv:
                            model = cv.fit(subset)
                            best_stage = model.bestModel.stages[-1]
                        else:
                            model = pipeline.fit(subset)
                            best_stage = model.stages[-1]
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

                model.transform(val_df)  # ensure pipeline reused on validation

                train_metrics = eval_metrics(model.transform(subset))
                val_metrics = eval_metrics(model.transform(val_df))
                test_preds = model.transform(test)
                test_metrics = eval_metrics(test_preds)

                metrics = {
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                    **test_metrics,
                    "rows": subset.count(),
                    "fast": FAST,
                }

                best_params = {}
                if name == "RandomForest":
                    best_params["maxDepth"] = best_stage.getOrDefault("maxDepth")
                    best_params["numTrees"] = best_stage.getNumTrees()
                elif name == "GBT":
                    best_params["maxDepth"] = best_stage.getOrDefault("maxDepth")
                    best_params["maxIter"] = best_stage.getMaxIter()
                    best_params["stepSize"] = best_stage.getStepSize()
                if best_params:
                    mlflow.log_params(best_params)

                run_dir = METRICS_DIR / run.info.run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)

                test_pd = test_preds.select(target, "prediction", "probability").toPandas()
                y_true = test_pd[target].astype(int)
                y_pred = test_pd["prediction"].astype(int)
                probs = np.array(test_pd["probability"].tolist())[:, 1]

                cm = confusion_matrix(y_true, y_pred)
                pd.DataFrame(cm).to_csv(run_dir / "cmatrix.csv", index=False)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                plt.tight_layout()
                fig.savefig(run_dir / "cmatrix.png")
                plt.close(fig)

                fpr, tpr, _ = roc_curve(y_true, probs)
                fig, ax = plt.subplots()
                ax.plot(fpr, tpr)
                ax.set_xlabel("FPR")
                ax.set_ylabel("TPR")
                plt.tight_layout()
                fig.savefig(run_dir / "roc.png")
                plt.close(fig)

                prec_c, rec_c, _ = precision_recall_curve(y_true, probs)
                fig, ax = plt.subplots()
                ax.plot(rec_c, prec_c)
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                plt.tight_layout()
                fig.savefig(run_dir / "pr.png")
                plt.close(fig)

                lc_df = compute_learning_curve(
                    pipeline, train_df, val_df, [0.1, 0.3, 0.5, 0.7, 1.0] if not FAST else [1.0], seed=seed
                )
                lc_df.to_csv(run_dir / "learning_curve.csv", index=False)
                fig, ax = plt.subplots()
                ax.plot(lc_df["fraction"], lc_df["train_auc"], label="train")
                ax.plot(lc_df["fraction"], lc_df["val_auc"], label="val")
                ax.set_xlabel("Fraction of training data")
                ax.set_ylabel("AUC")
                ax.legend()
                plt.tight_layout()
                fig.savefig(run_dir / "learning_curve.png")
                plt.close(fig)

                if hasattr(model.stages[-1], "featureImportances"):
                    fi = model.stages[-1].featureImportances.toArray().tolist()

                    with (run_dir / "feature_importance.json").open("w", encoding="utf-8") as f:
                        json.dump(fi, f)
                    mlflow.log_artifact(str(run_dir / "feature_importance.json"))

                mlflow.log_metrics({k: metrics[k] for k in metrics})
                mlflow.log_artifact(str(run_dir / "metrics.json"))
                mlflow.log_artifact(str(run_dir / "cmatrix.csv"))
                mlflow.log_artifact(str(run_dir / "cmatrix.png"))
                mlflow.log_artifact(str(run_dir / "roc.png"))
                mlflow.log_artifact(str(run_dir / "pr.png"))
                mlflow.log_artifact(str(run_dir / "learning_curve.csv"))
                mlflow.log_artifact(str(run_dir / "learning_curve.png"))

                mlflow.spark.log_model(
                    model.bestModel if cv else model,
                    f"models/supervised/{name}",
                    registered_model_name="credit-risk",
                )

                results.append({
                    "model": name,
                    **{k: metrics[k] for k in ["auc", "accuracy", "precision", "recall", "f1", "pr_auc"]},
                })

        summary_df = pd.DataFrame(results)
        print(summary_df)
        return 0
    finally:
        spark.stop()


if __name__ == "__main__":
    sys.exit(main())
