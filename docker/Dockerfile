# ------------------------------------------------------------------------------
# Base image
# ------------------------------------------------------------------------------
FROM python:3.11-slim

# ------------------------------------------------------------------------------
# Build-time args (se pueden sobreescribir desde docker-compose)
# ------------------------------------------------------------------------------
ARG MLFLOW_SPARK_VERSION=2.22.1
ARG SCALA_SUFFIX=2.13             # 2.12 si tu Spark usa Scala 2.12

# ------------------------------------------------------------------------------
# Sistema y Java
# ------------------------------------------------------------------------------
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        openjdk-17-jdk-headless \
        curl \
        make \
        procps \
        ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"
ENV PYTHONPATH=/app

# ------------------------------------------------------------------------------
# JAR mlflow-spark (curl -fSL aborta si recibe 4xx/5xx)
# ------------------------------------------------------------------------------
RUN set -eu; \
    mkdir -p /opt/spark/jars; \
    curl -fSL \
      "https://repo1.maven.org/maven2/org/mlflow/mlflow-spark_${SCALA_SUFFIX}/${MLFLOW_SPARK_VERSION}/mlflow-spark_${SCALA_SUFFIX}-${MLFLOW_SPARK_VERSION}.jar" \
      -o "/opt/spark/jars/mlflow-spark_${SCALA_SUFFIX}-${MLFLOW_SPARK_VERSION}.jar"

# ------------------------------------------------------------------------------
# Python deps
# ------------------------------------------------------------------------------
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# App code
# ------------------------------------------------------------------------------
COPY . .

# ------------------------------------------------------------------------------
# Entrypoint
# ------------------------------------------------------------------------------
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
