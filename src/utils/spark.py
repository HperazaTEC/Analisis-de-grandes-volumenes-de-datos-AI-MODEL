from pyspark.sql import SparkSession
import os
import re


def get_spark(app_name: str = "credit-risk") -> SparkSession:
    """Create or retrieve a Spark session with common config."""
    os.environ.setdefault("PYSPARK_PIN_THREAD", "false")
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    mem_str = os.environ.get("SPARK_DRIVER_MEMORY", "10g")
    digits = re.findall(r"\d+(?:\.\d+)?", mem_str)
    driver_mem_gb = float(digits[0]) if digits else 10.0


    builder = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", mem_str)
        .config("spark.executor.memory", os.environ.get("SPARK_EXECUTOR_MEMORY", mem_str))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.driver.maxResultSize", os.environ.get("SPARK_DRIVER_MAXRESULTSIZE", "3g"))
    )

    if driver_mem_gb <= 8:
        builder = builder.config("spark.sql.shuffle.partitions", "200")

    return builder.getOrCreate()
