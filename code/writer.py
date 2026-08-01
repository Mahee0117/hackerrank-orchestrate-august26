"""
writer.py
---------
Writes the final output.csv with predictions for all messages.
"""

import csv
from pathlib import Path


OUTPUT_COLUMNS = [
    "message_id",
    "action",
    "message_type",
    "reason",
    "confidence",
    "evidence_message_ids",
]

# Allowed values — used for validation before writing
VALID_ACTIONS       = {"notify", "digest", "mute"}
VALID_MESSAGE_TYPES = {
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
}


def _validate(row: dict) -> dict:
    """
    Ensure every field has a valid value.
    Falls back to safe defaults if the LLM returned something unexpected.
    """
    if row.get("action") not in VALID_ACTIONS:
        row["action"] = "digest"

    if row.get("message_type") not in VALID_MESSAGE_TYPES:
        row["message_type"] = "unknown"

    try:
        conf = float(row.get("confidence", 0.0))
        conf = max(0.0, min(1.0, conf))        # clamp to [0, 1]
    except (TypeError, ValueError):
        conf = 0.0
    row["confidence"] = round(conf, 2)

    if not row.get("reason", "").strip():
        row["reason"] = "No reason provided."

    if not row.get("evidence_message_ids", "").strip():
        row["evidence_message_ids"] = "none"

    return row


def write_output(results: list[dict], output_path: Path) -> None:
    """
    Write predictions to output.csv.

    Parameters
    ----------
    results     : list of dicts, each with the 6 required fields
    output_path : Path to write the CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()

        for row in results:
            row = _validate(row)
            writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})

    print(f"\n✅ Output written → {output_path}  ({len(results)} rows)")
