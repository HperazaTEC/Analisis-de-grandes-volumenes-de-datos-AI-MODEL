from pyspark.sql import DataFrame, functions as F
from functools import reduce


def stratified_kfolds(df: DataFrame, strat_cols: list, k: int, seed: int):
    """Return k stratified train/val splits using randomSplit."""
    df = df.withColumn("stratum", F.concat_ws("_", *[F.col(c) for c in strat_cols]))
    strata = [r[0] for r in df.select("stratum").distinct().collect()]
    splits = {}
    for s in strata:
        sub = df.filter(F.col("stratum") == s)
        splits[s] = sub.randomSplit([1.0] * k, seed)

    def _union(dfs):
        return reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), dfs)

    folds = []
    for i in range(k):
        val_parts = [_split[i] for _split in splits.values()]
        train_parts = [_split[j] for _split in splits.values() for j in range(k) if j != i]
        train_df = _union(train_parts).drop("stratum")
        val_df = _union(val_parts).drop("stratum")
        folds.append((train_df, val_df))
    return folds
