from pyspark.sql import SparkSession


def get_spark(app_name: str = "credit-risk") -> SparkSession:
    """Create or retrieve a Spark session with common config."""
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.kryoserializer.buffer.max", "512m")
        # Limit shuffle partitions to keep memory usage low and avoid OOM
        .config("spark.sql.shuffle.partitions", "4")
        # Allocate more driver/executor memory to handle moderate datasets
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
        .config("spark.driver.maxResultSize", "1g")

        .getOrCreate()
    )
