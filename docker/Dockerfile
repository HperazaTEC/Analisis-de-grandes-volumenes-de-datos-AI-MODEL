FROM python:3.11-slim

# ───── Dependencias de sistema ─────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        make \
        procps \
        wget && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYTHONPATH=/app
ENV MLFLOW_SPARK_VERSION=3.5.1
RUN mkdir -p /opt/spark/jars && \
    wget -q https://repo1.maven.org/maven2/org/mlflow/mlflow-spark/${MLFLOW_SPARK_VERSION}/mlflow-spark-${MLFLOW_SPARK_VERSION}.jar \
    -O /opt/spark/jars/mlflow-spark.jar

# ───── Aplicación ─────
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
