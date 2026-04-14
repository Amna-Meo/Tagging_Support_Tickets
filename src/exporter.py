import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from .database import get_db


def export_corrections(
    output_path: str = "./data/corrections_export.jsonl",
    min_records: int = 0,
) -> dict:
    db = get_db()
    corrections = db.export_all()

    if len(corrections) < min_records:
        return {
            "status": "skipped",
            "reason": f"Only {len(corrections)} corrections, need {min_records}",
            "count": len(corrections),
        }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    with open(output_path, "w") as f:
        for corr in corrections:
            if corr.get("label"):
                record = {
                    "text": corr["text"],
                    "label": corr["label"],
                }
                f.write(json.dumps(record) + "\n")
                exported += 1

    return {
        "status": "success",
        "count": exported,
        "path": str(output_path),
        "timestamp": datetime.utcnow().isoformat(),
    }


def get_export_stats() -> dict:
    db = get_db()
    total = db.get_correction_count()
    return {
        "total_corrections": total,
        "min_for_export": 10,
        "ready_for_export": total >= 10,
    }
