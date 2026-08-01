"""
main.py
-------
Entry point for the WhatsApp Notification Router.

Pipeline per message:
    media_processor  → extract image/voice content
    context_builder  → join all supporting datasets
    prompts          → build structured LLM prompt
    ollama_client    → call Gemma4
    writer           → append row to output.csv immediately

Resume support
--------------
If output.csv already exists and contains completed predictions,
those message_ids are loaded at startup and skipped in the loop.
Re-running after an interruption continues from where it left off.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from loader import load_data
from media_processor import process_media
from context_builder import build_context
from prompts import build_prompt
from ollama_client import ask_llm
from writer import append_row, OUTPUT_COLUMNS


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "dataset" / "output.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_fallback(message_id: str, evidence_ids: str, error: str) -> dict:
    """Return a safe fallback row when the LLM fails to parse."""
    return {
        "message_id"          : message_id,
        "action"              : "digest",
        "message_type"        : "unknown",
        "reason"              : f"Fallback: LLM parse error. {error[:120]}",
        "confidence"          : 0.0,
        "evidence_message_ids": evidence_ids,
    }


def load_done_ids(output_path: Path) -> set[str]:
    """
    Read output.csv and return the set of message_ids already processed.

    If the file does not exist or is empty, returns an empty set.
    This enables resume: any message_id in this set will be skipped.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(output_path, usecols=["message_id"])
        done = set(df["message_id"].astype(str).tolist())
        return done
    except Exception as e:
        print(f"⚠  Could not read existing output.csv ({e}). Starting fresh.")
        return set()


def verify_output(output_path: Path, all_message_ids: list[str]) -> bool:
    """
    Verify output.csv after completion.

    Checks:
      1. File exists and is readable.
      2. Columns match the required schema exactly.
      3. Row count == len(all_message_ids).
      4. Every message_id appears exactly once.
      5. No blank action or message_type fields.

    Returns True if all checks pass, False otherwise.
    """
    print("\n── Verification ──────────────────────────────────────────────────")

    if not output_path.exists():
        print("  ✗  output.csv does not exist.")
        return False

    try:
        df = pd.read_csv(output_path)
    except Exception as e:
        print(f"  ✗  Could not read output.csv: {e}")
        return False

    passed = True

    # 1. Schema check
    expected_cols = OUTPUT_COLUMNS
    actual_cols   = list(df.columns)
    if actual_cols != expected_cols:
        print(f"  ✗  Column mismatch.\n     Expected: {expected_cols}\n     Got     : {actual_cols}")
        passed = False
    else:
        print(f"  ✓  Columns correct: {actual_cols}")

    # 2. Row count
    expected_rows = len(all_message_ids)
    actual_rows   = len(df)
    if actual_rows != expected_rows:
        print(f"  ✗  Row count: expected {expected_rows}, got {actual_rows}")
        passed = False
    else:
        print(f"  ✓  Row count: {actual_rows} predictions")

    # 3. Line count (header + rows)
    with open(output_path, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f)
    expected_lines = expected_rows + 1
    if line_count != expected_lines:
        print(f"  ✗  Line count: expected {expected_lines}, got {line_count}")
        passed = False
    else:
        print(f"  ✓  Line count: {line_count} ({expected_rows} predictions + 1 header)")

    # 4. message_id coverage
    output_ids  = set(df["message_id"].astype(str).tolist())
    expected_set = set(str(m) for m in all_message_ids)
    missing  = expected_set - output_ids
    extra    = output_ids - expected_set
    dupes    = df[df.duplicated(subset=["message_id"], keep=False)]

    if missing:
        print(f"  ✗  Missing message_ids ({len(missing)}): {sorted(missing)[:10]}")
        passed = False
    else:
        print(f"  ✓  All {len(expected_set)} message_ids present")

    if extra:
        print(f"  ✗  Unexpected message_ids ({len(extra)}): {sorted(extra)[:10]}")
        passed = False

    if not dupes.empty:
        print(f"  ✗  Duplicate message_ids: {dupes['message_id'].tolist()}")
        passed = False
    else:
        print("  ✓  No duplicate message_ids")

    # 5. No blank action / message_type
    blank_action = df["action"].isna().sum() + (df["action"] == "").sum()
    blank_type   = df["message_type"].isna().sum() + (df["message_type"] == "").sum()
    if blank_action:
        print(f"  ✗  {blank_action} rows with blank 'action'")
        passed = False
    else:
        print("  ✓  No blank 'action' fields")

    if blank_type:
        print(f"  ✗  {blank_type} rows with blank 'message_type'")
        passed = False
    else:
        print("  ✓  No blank 'message_type' fields")

    print("──────────────────────────────────────────────────────────────────")
    if passed:
        print("✅  All checks passed. output.csv is complete and valid.")
    else:
        print("❌  Some checks failed. Review the output above.")

    return passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main(limit: int | None = None) -> None:

    # ── Load all datasets ─────────────────────────────────────────────────────
    print("Loading datasets…")
    data = load_data()

    # Add media lookup tables without modifying loader.py
    dataset_dir = PROJECT_ROOT / "dataset"
    for key, filename in [
        ("images",      "images.csv"),
        ("voice_notes", "voice_notes.csv"),
    ]:
        csv_path = dataset_dir / filename
        data[key] = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        if data[key].empty:
            print(f"  ⚠  {filename} not found — media lookup disabled for {key}")

    # ── Resume: find already-processed message_ids ────────────────────────────
    done_ids = load_done_ids(OUTPUT_PATH)
    if done_ids:
        print(f"Resume mode: {len(done_ids)} message(s) already in output.csv — skipping.")

    # ── Prepare message list ──────────────────────────────────────────────────
    messages = data["messages"]
    if limit:
        messages = messages.head(limit)

    all_ids = messages["message_id"].astype(str).tolist()
    total   = len(messages)
    errors  = 0
    skipped = 0

    print(f"Processing {total} messages…\n")

    # ── Main processing loop ──────────────────────────────────────────────────
    for idx, (_, msg) in enumerate(messages.iterrows(), start=1):
        message_id = str(msg["message_id"])
        media_type = str(msg.get("media_type", "")).strip()

        # ── Resume: skip already-done messages ────────────────────────────────
        if message_id in done_ids:
            print(f"  –  [{idx}/{total}] {message_id} → skipped (already processed)")
            skipped += 1
            continue

        # ── Step 1: Extract media content (image / voice) ─────────────────────
        media_text = ""
        if media_type in ("image", "voice"):
            try:
                media_text = process_media(msg, data)
            except Exception as e:
                media_text = f"[Media processing error: {e}]"
                print(f"  ⚠  [{idx}/{total}] {message_id} — media error: {e}")

        # ── Step 2: Build context, evidence IDs, safety signals, hint ─────────
        try:
            context, evidence_ids, safety_signals, confidence_hint = build_context(
                msg, data, media_text=media_text
            )
        except Exception as e:
            context, evidence_ids, safety_signals, confidence_hint = "", "none", {}, 0.70
            print(f"  ⚠  [{idx}/{total}] {message_id} — context error: {e}")

        # ── Step 3: Build prompt ───────────────────────────────────────────────
        prompt = build_prompt(msg, context, safety_signals, confidence_hint)

        # ── Step 4: Call LLM ───────────────────────────────────────────────────
        try:
            llm_result = ask_llm(prompt)
        except Exception as e:
            errors += 1
            row = make_fallback(message_id, evidence_ids, str(e))
            append_row(row, OUTPUT_PATH)          # ← persist immediately
            print(f"  ✗  [{idx}/{total}] {message_id} → FALLBACK  ({e})")
            continue

        # ── Step 5: Build and persist result row ──────────────────────────────
        row = {
            "message_id"          : message_id,
            "action"              : llm_result.get("action", "digest"),
            "message_type"        : llm_result.get("message_type", "unknown"),
            "reason"              : llm_result.get("reason", ""),
            "confidence"          : llm_result.get("confidence", 0.0),
            "evidence_message_ids": evidence_ids,
        }

        append_row(row, OUTPUT_PATH)              # ← persist immediately

        action = row["action"]
        mtype  = row["message_type"]
        conf   = row["confidence"]
        print(f"  ✓  [{idx}/{total}] {message_id} → {action} / {mtype}  (conf={conf})")

    # ── Summary ───────────────────────────────────────────────────────────────
    processed = total - skipped
    print(f"\n✅  Done. {processed} processed, {skipped} skipped, {errors} fallback(s).")
    print(f"    Output → {OUTPUT_PATH}")

    # ── Verification ─────────────────────────────────────────────────────────
    verify_output(OUTPUT_PATH, all_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhatsApp Notification Router")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N messages (for quick testing).",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing output.csv and reprocess all messages from scratch.",
    )
    args = parser.parse_args()

    if args.fresh and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        print(f"🗑  Deleted existing {OUTPUT_PATH} — starting fresh.")

    main(limit=args.limit)