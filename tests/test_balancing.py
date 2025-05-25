import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.utils.spark import get_spark
from src.utils.balancing import compute_class_weights, add_weight_column


def test_compute_class_weights():
    spark = get_spark("test-balance")
    df = spark.createDataFrame([(0,), (0,), (0,), (1,)], ["label"])
    weights = compute_class_weights(df, "label")
    assert pytest.approx(weights[0]) == 4 / (2 * 3)
    assert pytest.approx(weights[1]) == 4 / (2 * 1)


def test_add_weight_column():
    spark = get_spark("test-balance")
    df = spark.createDataFrame([(0,), (0,), (0,), (1,)], ["label"])
    df_w = add_weight_column(df, "label")
    rows = {r["label"]: r["weight"] for r in df_w.groupBy("label", "weight").count().collect()}
    assert pytest.approx(rows[0]) == 4 / (2 * 3)
    assert pytest.approx(rows[1]) == 4 / (2 * 1)
