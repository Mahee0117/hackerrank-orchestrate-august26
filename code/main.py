"""
main.py
-------
Entry point for the WhatsApp Notification Router.

Processes every row in messages.csv, calls the LLM with rich context,
and writes predictions to dataset/output.csv.
"""

import argparse
from pathlib import Path

from loader import load_data
from context_builder import build_context
from prompts import build_prompt
from ollama_client import ask_llm
from writer import write_output


# Resolve dataset root and output path from this file's location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "dataset" / "output.csv"


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


def main(limit: int | None = None) -> None:
    print("Loading datasets…")
    data = load_data()

    messages = data["messages"]
    if limit:
        messages = messages.head(limit)

    total   = len(messages)
    results = []
    errors  = 0

    print(f"Processing {total} messages…\n")

    for idx, (_, msg) in enumerate(messages.iterrows(), start=1):
        message_id = str(msg["message_id"])

        # ── Build context and evidence IDs ────────────────────────────────
        try:
            context, evidence_ids = build_context(msg, data)
        except Exception as e:
            context, evidence_ids = "", "none"
            print(f"  ⚠  [{idx}/{total}] {message_id} — context error: {e}")

        # ── Build prompt ──────────────────────────────────────────────────
        prompt = build_prompt(msg, context)

        # ── Call LLM ──────────────────────────────────────────────────────
        try:
            llm_result = ask_llm(prompt)
        except Exception as e:
            errors += 1
            results.append(make_fallback(message_id, evidence_ids, str(e)))
            print(f"  ✗  [{idx}/{total}] {message_id} → FALLBACK  ({e})")
            continue

        # ── Merge result ──────────────────────────────────────────────────
        row = {
            "message_id"          : message_id,
            "action"              : llm_result.get("action", "digest"),
            "message_type"        : llm_result.get("message_type", "unknown"),
            "reason"              : llm_result.get("reason", ""),
            "confidence"          : llm_result.get("confidence", 0.0),
            "evidence_message_ids": evidence_ids,
        }
        results.append(row)

        action = row["action"]
        mtype  = row["message_type"]
        conf   = row["confidence"]
        print(f"  ✓  [{idx}/{total}] {message_id} → {action} / {mtype}  (conf={conf})")

    # ── Write output ──────────────────────────────────────────────────────────
    write_output(results, OUTPUT_PATH)

    if errors:
        print(f"\n⚠  {errors}/{total} messages used fallback due to LLM errors.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhatsApp Notification Router")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N messages (for quick testing).",
    )
    args = parser.parse_args()
    main(limit=args.limit)