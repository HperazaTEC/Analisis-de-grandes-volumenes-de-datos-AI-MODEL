from pyspark.sql import SparkSession
import os, re

def get_spark(app_name: str = "credit-risk") -> SparkSession:
    os.environ.setdefault("PYSPARK_PIN_THREAD", "false")
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    master_url = os.getenv("SPARK_MASTER_URL", "local[*]")   # ← aquí la magia

    mem_str = os.environ.get("SPARK_DRIVER_MEMORY", "10g")
    digits  = re.findall(r"\d+(?:\.\d+)?", mem_str)
    driver_mem_gb = float(digits[0]) if digits else 10.0

    builder = (
        SparkSession.builder
        .master(master_url)                     # ← agregado
        .appName(app_name)
        .config("spark.driver.memory", mem_str)
        .config("spark.executor.memory", os.getenv("SPARK_EXECUTOR_MEMORY", mem_str))
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        .config("spark.driver.maxResultSize", os.getenv("SPARK_DRIVER_MAXRESULTSIZE", "3g"))
    )

    if driver_mem_gb <= 8:
        builder = builder.config("spark.sql.shuffle.partitions", "200")

    return builder.getOrCreate()