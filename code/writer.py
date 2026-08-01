"""
writer.py
---------
Writes the final output.csv with predictions for all messages.

Two write modes
---------------
append_row(row, path)
    Writes ONE row immediately after it is produced.
    Creates the file with header if it does not exist yet.
    If the file exists, the row is appended without re-writing the header.
    Use this inside the processing loop for crash-safe incremental output.

write_output(results, path)
    Writes ALL rows at once (overwrites the file).
    Used for batch writes and --limit test runs.
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


def _read_existing_ids(output_path: Path) -> set:
    """
    Return the set of message_ids already recorded in output_path.
    Returns an empty set if the file does not exist or has no rows.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    with open(output_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {str(r["message_id"]) for r in reader if "message_id" in r}


def append_row(row: dict, output_path: Path) -> None:
    """
    Append ONE validated row to output.csv immediately.

    If the file does not exist, it is created and the header is written first.
    If the file exists, the row is appended without re-writing the header.

    This function is called once per message inside the processing loop
    so that every completed prediction is persisted immediately.
    A KeyboardInterrupt or crash after this call loses at most ONE row.

    Raises
    ------
    ValueError
        If the message_id of *row* already exists in output_path.  This is a
        hard guard that fires even if the caller forgot to check done_ids.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row = _validate(row)

    # ── Hard duplicate guard ───────────────────────────────────────────────
    message_id = str(row.get("message_id", ""))
    existing_ids = _read_existing_ids(output_path)
    if message_id in existing_ids:
        raise ValueError(
            f"append_row: duplicate write blocked — message_id {message_id!r} "
            f"already exists in {output_path}"
        )
    # ──────────────────────────────────────────────────────────────────────

    file_exists = output_path.exists() and output_path.stat().st_size > 0

    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in OUTPUT_COLUMNS})


def write_output(results: list[dict], output_path: Path) -> None:
    """
    Write ALL predictions at once, overwriting any existing file.

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
