FROM python:3.11-slim

WORKDIR /app

# Install Java for PySpark and basic utilities
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
