"""Register the best run in the MLflow Model Registry."""
import mlflow
from mlflow.tracking import MlflowClient


def main() -> None:
    client = MlflowClient()
    runs = mlflow.search_runs(order_by=["metrics.auc DESC"], max_results=1)
    if not runs.empty:
        run_id = runs.loc[0, "run_id"]
        client.create_model_version(name="credit-risk", source=f"models:/credit-risk/{run_id}", run_id=run_id)


if __name__ == "__main__":
    main()
