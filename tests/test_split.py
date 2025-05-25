import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.utils.spark import get_spark
from src.agents.split import stratified_split


def test_stratified_split_counts():
    spark = get_spark("test-split")
    rows = []
    for i in range(10):
        rows.append(("A", "Good", i))
        rows.append(("B", "Bad", i))
    df = spark.createDataFrame(rows, ["grade", "loan_status", "id"])
    train, test = stratified_split(df, ["grade", "loan_status"], test_frac=0.2, seed=42)

    assert train.count() + test.count() == df.count()
    train_counts = { (r["grade"], r["loan_status"]): r["count"]
                     for r in train.groupBy("grade", "loan_status").count().collect() }
    test_counts = { (r["grade"], r["loan_status"]): r["count"]
                    for r in test.groupBy("grade", "loan_status").count().collect() }

    assert train_counts[("A", "Good")] == 8
    assert test_counts[("A", "Good")] == 2
    assert train_counts[("B", "Bad")] == 8
    assert test_counts[("B", "Bad")] == 2
