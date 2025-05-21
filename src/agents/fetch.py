"""Download raw data from an S3/MinIO bucket."""
from pathlib import Path
import os
import boto3


def main() -> None:
    bucket = os.environ.get("DATA_BUCKET", "lending-data")
    key = os.environ.get("DATA_KEY", "lending_club.csv")
    dest = Path("data/raw/lending_club.csv")
    dest.parent.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("S3_ENDPOINT_URL")
    s3 = boto3.client("s3", endpoint_url=endpoint)
    s3.download_file(bucket, key, str(dest))


if __name__ == "__main__":
    main()
