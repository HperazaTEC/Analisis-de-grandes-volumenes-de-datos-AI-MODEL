from pyspark.sql import SparkSession


def get_spark(app_name: str = "credit-risk") -> SparkSession:
    """Create or retrieve a Spark session with common config."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")

        .getOrCreate()
    )
