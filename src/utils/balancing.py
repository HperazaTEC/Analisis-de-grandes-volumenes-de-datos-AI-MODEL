from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


def compute_class_weights(df: DataFrame, target: str) -> dict:
    """Compute weights to balance binary classes."""
    counts = df.groupBy(target).count().collect()
    total = sum(row['count'] for row in counts)
    weights = {row[target]: total / (2.0 * row['count']) for row in counts}
    return weights


def add_weight_column(df: DataFrame, target: str, weight_col: str = "weight") -> DataFrame:
    weights = compute_class_weights(df, target)
    mapping_expr = F.create_map([F.lit(x) for item in weights.items() for x in item])
    return df.withColumn(weight_col, mapping_expr.getItem(F.col(target)).cast(DoubleType()))
