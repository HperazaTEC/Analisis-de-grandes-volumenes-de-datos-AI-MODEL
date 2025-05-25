"""Stratified train/test split by grade and loan_status."""
from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from functools import reduce
from src.utils.spark import get_spark
from pathlib import Path


def stratified_split(df: DataFrame, strat_cols: list, test_frac: float, seed: int):
    df = df.withColumn("stratum", F.concat_ws("_", *[F.col(c) for c in strat_cols]))
    strata = [r[0] for r in df.select("stratum").distinct().collect()]
    train_parts = []
    test_parts = []
    for s in strata:
        sub = df.filter(F.col("stratum") == s)
        train, test = sub.randomSplit([1-test_frac, test_frac], seed)
        train_parts.append(train)
        test_parts.append(test)
    union = lambda dfs: reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)
    train_df = union(train_parts).drop("stratum")
    test_df = union(test_parts).drop("stratum")
    return train_df, test_df


def main() -> None:
    spark = get_spark("split")
    df = spark.read.parquet("data/processed/M.parquet")
    train, test = stratified_split(df, ["grade", "loan_status"], test_frac=0.2, seed=42)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    train.write.mode("overwrite").parquet("data/processed/train.parquet")
    test.write.mode("overwrite").parquet("data/processed/test.parquet")


if __name__ == "__main__":
    main()
