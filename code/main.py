"""
main.py
-------
Entry point for the WhatsApp Notification Router.

Pipeline per message:
    media_processor  → extract image/voice content
    context_builder  → join all supporting datasets (cache-accelerated)
    prompts          → build structured LLM prompt
    ollama_client    → call Gemma4
    writer           → append row to output.csv immediately

Performance features
--------------------
- build_cache(): pre-builds O(1) lookup indexes before the loop
- Per-stage timing printed every message
- Messages reordered: text/voice first, image (vision) last
- --workers N: concurrent text processing via ThreadPoolExecutor
- Resume: already-processed messages skipped via done_ids set
- Duplicate guard: done_ids updated immediately after write

Flags
-----
  --limit N    process only first N messages (for testing)
  --fresh      delete output.csv and start from scratch
  --workers N  concurrent workers for text/voice messages (default 1)
"""

import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from loader import load_data
from media_processor import process_media
from context_builder import build_context, build_cache
from prompts import build_prompt
from ollama_client import ask_llm
from writer import append_row, OUTPUT_COLUMNS


# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH  = PROJECT_ROOT / "dataset" / "output.csv"

# Thread-safe write lock (used when --workers > 1)
_write_lock = threading.Lock()


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

    Returns an empty set if the file does not exist or is unreadable.
    This set is the single source of truth for resume and duplicate prevention.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(output_path, usecols=["message_id"])
        return set(df["message_id"].astype(str).tolist())
    except Exception as e:
        print(f"⚠  Could not read existing output.csv ({e}). Starting fresh.")
        return set()


def process_one_message(
    idx:     int,
    total:   int,
    msg:     pd.Series,
    data:    dict,
    cache:   dict,
) -> tuple[dict, dict]:
    """
    Process a single message through the full pipeline.

    Returns (row_dict, timing_dict).
    Thread-safe: reads only from shared immutable data/cache.
    """
    message_id = str(msg["message_id"])
    media_type = str(msg.get("media_type", "")).strip()

    timings: dict = {}

    # ── Step 1: Media extraction ───────────────────────────────────────────
    t0 = time.perf_counter()
    media_text = ""
    if media_type in ("image", "voice"):
        try:
            media_text = process_media(msg, data)
        except Exception as e:
            media_text = f"[Media processing error: {e}]"
    timings["media"] = time.perf_counter() - t0

    # ── Step 2: Context build ──────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        context, evidence_ids, safety_signals, confidence_hint = build_context(
            msg, data, media_text=media_text, cache=cache
        )
    except Exception as e:
        context, evidence_ids, safety_signals, confidence_hint = "", "none", {}, 0.70
        print(f"  ⚠  [{idx}/{total}] {message_id} — context error: {e}")
    timings["ctx"] = time.perf_counter() - t0

    # ── Step 3: Prompt build ───────────────────────────────────────────────
    t0 = time.perf_counter()
    prompt = build_prompt(msg, context, safety_signals, confidence_hint)
    timings["prompt"] = time.perf_counter() - t0

    # ── Step 4: LLM inference ─────────────────────────────────────────────
    t0 = time.perf_counter()
    try:
        llm_result = ask_llm(prompt)
    except Exception as e:
        timings["llm"] = time.perf_counter() - t0
        row = make_fallback(message_id, evidence_ids, str(e))
        timings["total"] = sum(timings.values())
        return row, timings
    timings["llm"] = time.perf_counter() - t0

    # ── Step 5: Build result row ───────────────────────────────────────────
    row = {
        "message_id"          : message_id,
        "action"              : llm_result.get("action",       "digest"),
        "message_type"        : llm_result.get("message_type", "unknown"),
        "reason"              : llm_result.get("reason",       ""),
        "confidence"          : llm_result.get("confidence",   0.0),
        "evidence_message_ids": evidence_ids,
    }
    timings["total"] = sum(timings.values())
    return row, timings


def verify_output(output_path: Path, all_message_ids: list[str]) -> bool:
    """
    Verify output.csv after completion.

    Checks: schema, row count, line count, full ID coverage, no dupes, no blanks.
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

    # 1. Schema
    if list(df.columns) != OUTPUT_COLUMNS:
        print(f"  ✗  Columns: expected {OUTPUT_COLUMNS}, got {list(df.columns)}")
        passed = False
    else:
        print(f"  ✓  Columns: {list(df.columns)}")

    # 2. Row count
    expected = len(all_message_ids)
    if len(df) != expected:
        print(f"  ✗  Row count: expected {expected}, got {len(df)}")
        passed = False
    else:
        print(f"  ✓  Row count: {len(df)} predictions")

    # 3. Line count
    with open(output_path, encoding="utf-8") as f:
        lines = sum(1 for _ in f)
    if lines != expected + 1:
        print(f"  ✗  Line count: expected {expected+1}, got {lines}")
        passed = False
    else:
        print(f"  ✓  Line count: {lines} ({expected} predictions + 1 header)")

    # 4. Coverage + dupes
    output_ids   = set(df["message_id"].astype(str))
    expected_set = set(str(m) for m in all_message_ids)
    missing = expected_set - output_ids
    extra   = output_ids - expected_set
    dupes   = df[df.duplicated(subset=["message_id"], keep=False)]

    if missing: print(f"  ✗  Missing IDs ({len(missing)}): {sorted(missing)[:10]}"); passed = False
    else:       print(f"  ✓  All {len(expected_set)} message_ids present")

    if extra:   print(f"  ✗  Unexpected IDs ({len(extra)}): {sorted(extra)[:10]}"); passed = False

    if not dupes.empty:
        print(f"  ✗  Duplicate IDs: {dupes['message_id'].tolist()}")
        passed = False
    else:
        print("  ✓  No duplicate message_ids")

    # 5. Blank fields
    for col in ("action", "message_type"):
        blanks = df[col].isna().sum() + (df[col] == "").sum()
        if blanks: print(f"  ✗  {blanks} blank '{col}' fields"); passed = False
        else:      print(f"  ✓  No blank '{col}' fields")

    print("──────────────────────────────────────────────────────────────────")
    if passed:
        print("✅  All checks passed. output.csv is complete and valid.")
    else:
        print("❌  Some checks failed. Review output above.")
    return passed


# ── Main ──────────────────────────────────────────────────────────────────────

def main(limit: int | None = None, workers: int = 1) -> None:
    run_start = time.perf_counter()

    # ── Load datasets ─────────────────────────────────────────────────────────
    print("Loading datasets…")
    t0   = time.perf_counter()
    data = load_data()

    dataset_dir = PROJECT_ROOT / "dataset"
    for key, filename in [("images", "images.csv"), ("voice_notes", "voice_notes.csv")]:
        csv_path = dataset_dir / filename
        data[key] = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
        if data[key].empty:
            print(f"  ⚠  {filename} not found — media disabled for {key}")
    print(f"  Datasets loaded in {time.perf_counter()-t0:.2f}s")

    # ── Build cache (O(1) lookups for all profile/history data) ──────────────
    print("Building lookup cache…")
    t0    = time.perf_counter()
    cache = build_cache(data)
    print(f"  Cache built in {time.perf_counter()-t0:.2f}s")

    # ── Resume: load already-processed IDs ────────────────────────────────────
    done_ids = load_done_ids(OUTPUT_PATH)
    if done_ids:
        print(f"Resume mode: {len(done_ids)} message(s) already done — skipping.")

    # ── Message ordering: text/voice first, image last ───────────────────────
    # Image messages need Gemma4 Vision (10–25s each).
    # Processing text messages first ensures most output is written before
    # the slow vision phase begins, maximising resume value if interrupted.
    messages = data["messages"]
    if limit:
        messages = messages.head(limit)

    is_image = messages["media_type"].fillna("") == "image"
    ordered  = pd.concat([messages[~is_image], messages[is_image]]).reset_index(drop=True)
    all_ids  = ordered["message_id"].astype(str).tolist()
    total    = len(ordered)

    pending = [
        (idx + 1, row)
        for idx, (_, row) in enumerate(ordered.iterrows())
        if str(row["message_id"]) not in done_ids
    ]

    n_skip = total - len(pending)
    n_img  = int(is_image.sum())
    n_text = total - n_img

    print(f"\nProcessing {total} messages ({n_text} text/voice, {n_img} image) | "
          f"{n_skip} skipped | workers={workers}\n")

    # Timing accumulators
    total_timings = {"media": 0.0, "ctx": 0.0, "prompt": 0.0, "llm": 0.0, "write": 0.0}
    errors = 0

    # ── Phase 1: Text + voice messages (optionally concurrent) ────────────────
    text_pending  = [(i, r) for i, r in pending if str(r.get("media_type", "")).strip() != "image"]
    image_pending = [(i, r) for i, r in pending if str(r.get("media_type", "")).strip() == "image"]

    def _handle_result(idx: int, row: dict, timings: dict) -> None:
        """Write row to CSV and update done_ids. Called from main thread always."""
        nonlocal errors
        message_id = row["message_id"]
        tw = time.perf_counter()
        with _write_lock:
            # Duplicate guard: only write if not already recorded
            if message_id not in done_ids:
                append_row(row, OUTPUT_PATH)
                done_ids.add(message_id)
        timings["write"] = time.perf_counter() - tw

        for k, v in timings.items():
            total_timings[k] = total_timings.get(k, 0.0) + v

        action = row.get("action",       "?")
        mtype  = row.get("message_type", "?")
        conf   = row.get("confidence",   0.0)
        llm_t  = timings.get("llm", 0.0)
        ctx_t  = timings.get("ctx", 0.0)
        med_t  = timings.get("media", 0.0)
        print(f"  ✓  [{idx}/{total}] {message_id} → {action} / {mtype}  "
              f"(conf={conf} | llm={llm_t:.1f}s ctx={ctx_t:.3f}s media={med_t:.1f}s)")

        if row.get("action") == "digest" and row.get("confidence") == 0.0:
            errors += 1

    if workers > 1 and text_pending:
        # Concurrent text processing.
        # Note: Ollama processes one inference at a time internally.
        # Workers overlap CPU work (context/prompt build) with I/O wait (inference).
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_one_message, idx, total, msg, data, cache): idx
                for idx, msg in text_pending
            }
            for future in as_completed(futures):
                try:
                    row, timings = future.result()
                    _handle_result(futures[future], row, timings)
                except Exception as e:
                    print(f"  ✗  Worker error: {e}")
    else:
        # Sequential text processing (default)
        for idx, msg in text_pending:
            row, timings = process_one_message(idx, total, msg, data, cache)
            _handle_result(idx, row, timings)

    # ── Phase 2: Image messages — always sequential (vision calls are slow) ───
    if image_pending:
        print(f"\n── Starting image phase ({len(image_pending)} messages) ──")
    for idx, msg in image_pending:
        row, timings = process_one_message(idx, total, msg, data, cache)
        _handle_result(idx, row, timings)

    # ── Summary ───────────────────────────────────────────────────────────────
    run_elapsed = time.perf_counter() - run_start
    processed   = len(pending)
    if processed > 0:
        avg = run_elapsed / processed
        print(f"\n{'─'*60}")
        print(f"  Processed : {processed} messages in {run_elapsed:.1f}s")
        print(f"  Skipped   : {n_skip} (already in output.csv)")
        print(f"  Avg/msg   : {avg:.1f}s  (est. full 110: {avg*110/60:.1f} min)")
        print(f"  Stage avg : media={total_timings['media']/processed:.2f}s  "
              f"ctx={total_timings['ctx']/processed:.3f}s  "
              f"llm={total_timings['llm']/processed:.2f}s")
        print(f"  Output    : {OUTPUT_PATH}")

    # ── Verification ──────────────────────────────────────────────────────────
    verify_output(OUTPUT_PATH, all_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhatsApp Notification Router")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only first N messages (for testing)."
    )
    parser.add_argument(
        "--fresh", action="store_true",
        help="Delete existing output.csv and reprocess all messages from scratch."
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help=(
            "Concurrent workers for text/voice messages (default: 1). "
            "Ollama processes one inference at a time; workers overlap CPU "
            "preparation with inference I/O. Safe values: 1-4. "
            "Image messages are always processed sequentially."
        )
    )
    args = parser.parse_args()

    if args.fresh and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        print(f"🗑  Deleted existing {OUTPUT_PATH} — starting fresh.")

    main(limit=args.limit, workers=args.workers)