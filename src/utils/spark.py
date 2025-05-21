from pyspark.sql import SparkSession


def get_spark(app_name: str = "credit-risk") -> SparkSession:
    """Create or retrieve a Spark session with common config."""
    return (
        SparkSession.builder
        .appName(app_name)
        .getOrCreate()
    )
