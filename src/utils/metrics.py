import os
import json
import datetime as dt
from pathlib import Path


METRICS_DIR = Path(os.getenv("METRICS_DIR", "metrics"))
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def dump_metrics(step: str, payload: dict) -> None:
    ts = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    with (METRICS_DIR / f"{step}_{ts}.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

