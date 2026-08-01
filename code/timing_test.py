"""
timing_test.py
--------------
Run the full pipeline on the first N messages and write results to
dataset/test_output.csv (NEVER dataset/output.csv).

Usage:
    python3 code/timing_test.py --limit 4 --workers 3

This is a safe timing probe: it does not accept --fresh and will never
touch the real output.csv.  It always writes to TEST_OUTPUT_PATH.
"""

import argparse
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ── Override output path BEFORE importing main so nothing ever touches
#    dataset/output.csv.
import importlib, types

# Patch sys.path so local imports resolve
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from loader import load_data
from context_builder import build_cache, build_context
from prompts import build_prompt
from ollama_client import ask_llm
from writer import append_row, OUTPUT_COLUMNS

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
TEST_OUT_PATH  = PROJECT_ROOT / "dataset" / "test_output.csv"   # ← isolated path
_write_lock    = threading.Lock()


def make_fallback(message_id, evidence_ids, error):
    return {
        "message_id": message_id, "action": "digest",
        "message_type": "unknown",
        "reason": f"Fallback: {error[:100]}",
        "confidence": 0.0, "evidence_message_ids": evidence_ids,
    }


def process_one(idx, total, msg, data, cache):
    message_id  = str(msg["message_id"])
    media_type  = str(msg.get("media_type", "")).strip()
    media_text  = ""
    timings: dict = {}

    t0 = time.perf_counter()
    if media_type in ("image", "voice"):
        try:
            from media_processor import process_media
            media_text = process_media(msg, data)
        except Exception as e:
            media_text = f"[media error: {e}]"
    timings["media"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        context, evidence_ids, safety_signals, confidence_hint = build_context(
            msg, data, media_text=media_text, cache=cache
        )
    except Exception as e:
        context, evidence_ids, safety_signals, confidence_hint = "", "none", {}, 0.55
    timings["ctx"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    prompt = build_prompt(msg, context, safety_signals, confidence_hint)
    timings["prompt"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    try:
        result = ask_llm(prompt)
    except Exception as e:
        timings["llm"] = time.perf_counter() - t0
        return make_fallback(message_id, evidence_ids, str(e)), timings
    timings["llm"] = time.perf_counter() - t0

    row = {
        "message_id":           message_id,
        "action":               result.get("action",       "digest"),
        "message_type":         result.get("message_type", "unknown"),
        "reason":               result.get("reason",       ""),
        "confidence":           result.get("confidence",   0.0),
        "evidence_message_ids": evidence_ids,
    }
    timings["total"] = sum(timings.values())
    return row, timings


def main():
    parser = argparse.ArgumentParser(description="Timing test — writes to test_output.csv only")
    parser.add_argument("--limit",   type=int, default=4)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    # ── Always start fresh for test output ────────────────────────────────────
    if TEST_OUT_PATH.exists():
        TEST_OUT_PATH.unlink()
        print(f"🗑  Cleared {TEST_OUT_PATH.name}")

    print(f"Loading data…")
    data  = load_data()
    cache = build_cache(data)

    msgs    = data["messages"].head(args.limit)
    total   = len(msgs)
    done_ids: set = set()

    # Reorder: text first, image last (mirrors main.py)
    is_image = msgs["media_type"].fillna("") == "image"
    ordered  = pd.concat([msgs[~is_image], msgs[is_image]]).reset_index(drop=True)
    pending  = list(enumerate(ordered.iterrows(), 1))
    pending  = [(idx, row) for idx, (_, row) in enumerate(ordered.iterrows(), 1)]

    text_pending  = [(i, r) for i, r in pending if str(r.get("media_type","")).strip() != "image"]
    image_pending = [(i, r) for i, r in pending if str(r.get("media_type","")).strip() == "image"]

    wall_start = time.perf_counter()
    total_timings = {"media": 0.0, "ctx": 0.0, "llm": 0.0}

    def handle(idx, row, timings):
        with _write_lock:
            if row["message_id"] not in done_ids:
                append_row(row, TEST_OUT_PATH)
                done_ids.add(row["message_id"])
        for k, v in timings.items():
            total_timings[k] = total_timings.get(k, 0.0) + v
        print(f"  ✓ [{idx}/{total}] {row['message_id']} → {row['action']}/{row['message_type']} "
              f"conf={row['confidence']} llm={timings.get('llm',0):.1f}s")

    print(f"\nRunning {total} messages (workers={args.workers}) → {TEST_OUT_PATH.name}\n")

    if args.workers > 1 and text_pending:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_one, i, total, msg, data, cache): i
                       for i, msg in text_pending}
            for f in as_completed(futures):
                try:
                    row, timings = f.result()
                    handle(futures[f], row, timings)
                except Exception as e:
                    print(f"  ✗ worker error: {e}")
    else:
        for i, msg in text_pending:
            row, timings = process_one(i, total, msg, data, cache)
            handle(i, row, timings)

    for i, msg in image_pending:
        row, timings = process_one(i, total, msg, data, cache)
        handle(i, row, timings)

    wall_elapsed = time.perf_counter() - wall_start
    processed    = len(done_ids)
    avg          = wall_elapsed / processed if processed else 0

    print(f"\n{'─'*55}")
    print(f"  Wall time   : {wall_elapsed:.1f}s  ({wall_elapsed/60:.2f} min)")
    print(f"  Messages    : {processed}")
    print(f"  Avg/message : {avg:.1f}s")
    print(f"  Test output : {TEST_OUT_PATH}  (output.csv untouched)")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
