
"""Clean raw LendingClub data and create train/test splits."""

from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from src.utils.spark import get_spark
from src.agents.split import stratified_split
from pathlib import Path
from dotenv import load_dotenv
import unicodedata
import os



def winsorize(df, cols, lower=0.01, upper=0.99):
    for c in cols:
        q = df.approxQuantile(c, [lower, upper], 0.05)
        df = df.withColumn(
            c,
            F.when(F.col(c) < q[0], q[0])
            .when(F.col(c) > q[1], q[1])
            .otherwise(F.col(c)),
        )
    return df


def _normalize(text: str) -> str:
    if text is None:
        return text
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text.lower()


def top_k(df, col, k=100):
    cats = (
        df.groupBy(col)
        .count()
        .orderBy(F.desc("count"))
        .limit(k)
        .collect()
    )
    cats = [r[0] for r in cats]
    return df.withColumn(col, F.when(F.col(col).isin(cats), F.col(col)).otherwise("other"))


def main() -> None:
    load_dotenv()
    spark = get_spark("prep")
    src = Path(os.environ.get("RAW_DATA", "data/raw/Loan_status_2007-2020Q3.gzip"))
    proc_dir = Path("data/processed")
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw_dir = src.parent
    parts = sorted(raw_dir.glob("loan_data_2007_2020Q*.csv"))
    if len(parts) >= 3:
        df = (
            spark.read.option("header", "true").option("inferSchema", "true").csv([str(p) for p in parts])
        )
    else:
        df = (
            spark.read.option("header", "true").option("inferSchema", "true").option("compression", "gzip").csv(str(src))
        )
    if "_c0" in df.columns:
        df = df.drop("_c0")

    categorical_vars = [
        "term", "grade", "emp_length", "home_ownership",
        "verification_status", "purpose", "loan_status",
    ]

    numerical_vars = [
        "loan_amnt", "int_rate", "installment", "fico_range_low", "fico_range_high",
        "annual_inc", "dti", "open_acc", "total_acc", "revol_bal", "revol_util",
    ]

    percent_cols = ["int_rate", "revol_util"]
    for c in percent_cols:
        df = df.withColumn(c, F.regexp_replace(c, "%", "").cast("double"))
    for c in set(numerical_vars) - set(percent_cols):
        df = df.withColumn(c, F.col(c).cast("double"))

    num_cols = [c for c in df.columns if c in numerical_vars]
    cat_cols = [c for c in df.columns if c in categorical_vars]
    medians = {c: df.approxQuantile(c, [0.5], 0.05)[0] for c in num_cols}
    df = df.fillna(medians)
    df = df.fillna("missing", subset=cat_cols)

    df = df.withColumn("grade_status", F.concat_ws("_", "grade", "loan_status"))
    df = df.withColumn(
        "default_flag",
        F.when(F.col("loan_status").isin("Charged Off", "Default"), 1).otherwise(0),
    )

    df = winsorize(df, ["annual_inc", "dti", "loan_amnt", "int_rate", "revol_util"])

    df = df.withColumn("loan_to_income", F.col("loan_amnt") / (F.col("annual_inc") + F.lit(1)))
    issue_year = F.year(F.to_date(F.concat(F.lit("01-"), F.col("issue_d")), "dd-MMM-yyyy"))
    earliest_year = F.year(F.to_date(F.concat(F.lit("01-"), F.col("earliest_cr_line")), "dd-MMM-yyyy"))
    df = df.withColumn("credit_age", issue_year - earliest_year)

    norm_udf = F.udf(_normalize, StringType())
    df = df.withColumn("emp_title", norm_udf(F.col("emp_title")))

    for c in cat_cols:
        df = top_k(df, c, 100)

    train, test = stratified_split(df, ["grade", "loan_status"], test_frac=0.2, seed=42)

    train.write.mode("overwrite").parquet(str(proc_dir / "train.parquet"))
    test.write.mode("overwrite").parquet(str(proc_dir / "test.parquet"))




if __name__ == "__main__":
    main()
