
"""Download raw LendingClub data from Kaggle with basic validation."""

from pathlib import Path
import os
import zipfile

from dotenv import load_dotenv
from kaggle import api


def main() -> None:
    load_dotenv()

    dataset = os.environ.get("KAGGLE_DATASET")
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    file_name = os.environ.get("KAGGLE_FILE", "Loan_status_2007-2020Q3.gzip")

    if not dataset or not username or not key:
        raise EnvironmentError(
            "Kaggle credentials or dataset not configured.\n"
            "Please set KAGGLE_USERNAME, KAGGLE_KEY and KAGGLE_DATASET in the .env file."
        )
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / file_name

    if out_path.exists():
        return

    api.authenticate()
    api.dataset_download_file(dataset, file_name, path=str(raw_dir), force=True)
    zip_path = raw_dir / f"{file_name}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(path=raw_dir)
        zip_path.unlink()



if __name__ == "__main__":
    main()
