
"""Clean raw LendingClub data and create stratified sample."""

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
from utils.spark import get_spark
from pathlib import Path
from dotenv import load_dotenv
import os



def winsorize(df, cols, lower=0.01, upper=0.99):
    for c in cols:
        q = df.approxQuantile(c, [lower, upper], 0.05)
        df = df.withColumn(c, F.when(F.col(c) < q[0], q[0])
                                  .when(F.col(c) > q[1], q[1])
                                  .otherwise(F.col(c)))
    return df


def main() -> None:
    load_dotenv()
    spark = get_spark("prep")
    src = os.environ.get("RAW_DATA", "data/raw/Loan_status_2007-2020Q3.gzip")
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    df = (
        spark.read
             .option("header", "true")
             .option("inferSchema", "true")
             .option("compression", "gzip")
             .csv(src)
    )

    categorical_vars = [
        "term", "grade", "emp_length", "home_ownership",
        "verification_status", "purpose", "loan_status",
    ]

    numerical_vars = [
        "loan_amnt", "int_rate", "installment", "fico_range_low", "fico_range_high",
        "annual_inc", "dti", "open_acc", "total_acc", "revol_bal", "revol_util",
    ]

    df = df.dropna(subset=categorical_vars + numerical_vars)
    percent_cols = ["int_rate", "revol_util"]
    for c in percent_cols:
        df = df.withColumn(c, F.regexp_replace(c, "%", "").cast("double"))
    for c in set(numerical_vars) - set(percent_cols):
        df = df.withColumn(c, F.col(c).cast("double"))

    df = df.withColumn("grade_status", F.concat_ws("_", "grade", "loan_status"))
    df = df.withColumn(
        "default_flag",
        F.when(F.col("loan_status").isin("Charged Off", "Default"), 1).otherwise(0),
    )
    df = winsorize(df, numerical_vars)

    df.write.mode("overwrite").parquet(str(proc_dir / "M.parquet"))

    # 10 % stratified sample for exploratory analysis
    strata = [r[0] for r in df.select("grade_status").distinct().collect()]
    fractions = {s: 0.10 for s in strata}
    sample = df.sampleBy("grade_status", fractions, seed=42)
    sample.write.mode("overwrite").parquet(str(proc_dir / "sample_M.parquet"))




if __name__ == "__main__":
    main()
