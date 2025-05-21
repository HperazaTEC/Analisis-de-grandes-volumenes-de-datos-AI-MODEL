"""Clean raw LendingClub data and write processed Parquet."""
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from utils.spark import get_spark
from pathlib import Path


def winsorize(df, cols, lower=0.01, upper=0.99):
    for c in cols:
        q = df.approxQuantile(c, [lower, upper], 0.05)
        df = df.withColumn(c, F.when(F.col(c) < q[0], q[0])
                                  .when(F.col(c) > q[1], q[1])
                                  .otherwise(F.col(c)))
    return df


def main() -> None:
    spark = get_spark("prep")
    src = "data/raw/lending_club.csv"
    dest = Path("data/processed/M.parquet")
    dest.parent.mkdir(parents=True, exist_ok=True)
    df = (spark.read.option("header", "true")
                  .option("inferSchema", "true")
                  .csv(src))
    df = df.dropna(subset=["loan_status", "grade"])
    num_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (DoubleType, IntegerType))]
    df = winsorize(df, num_cols)
    df.write.mode("overwrite").parquet(str(dest))


if __name__ == "__main__":
    main()
