import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.utils.spark import get_spark
from src.utils.crossval import stratified_kfolds


@pytest.fixture(scope="session")
def spark_session():
    spark = get_spark("test-crossval")
    yield spark
    try:
        spark.stop()
    except Exception:
        pass


def _make_df(spark):
    rows = []
    for i in range(200):
        rows.append(("A", "Good", i))
        rows.append(("A", "Bad", 200 + i))
        rows.append(("B", "Good", 400 + i))
        rows.append(("B", "Bad", 600 + i))
    return spark.createDataFrame(rows, ["grade", "loan_status", "id"])


def test_stratified_kfolds(spark_session):
    spark = spark_session
    df = _make_df(spark)
    folds = stratified_kfolds(df, ["grade", "loan_status"], k=5, seed=42)

    # Each row should appear exactly once in validation sets
    val_union = folds[0][1]
    for _, val in folds[1:]:
        val_union = val_union.unionByName(val, allowMissingColumns=True)
    assert val_union.count() == df.count()
    assert val_union.distinct().count() == df.count()

    # Check approximate class distribution per fold
    global_ratio = {
        (r["grade"], r["loan_status"]): r["count"] / df.count()
        for r in df.groupBy("grade", "loan_status").count().collect()
    }
    for _, val in folds:
        total = val.count()
        counts = {
            (r["grade"], r["loan_status"]): r["count"]
            for r in val.groupBy("grade", "loan_status").count().collect()
        }
        for key, expected in global_ratio.items():
            ratio = counts[key] / total
            assert abs(ratio - expected) < 0.1
