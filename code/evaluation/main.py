"""
evaluation/main.py
------------------
Evaluate output.csv predictions against the ground-truth labels in
sample_messages.csv.

Usage (from the project root):
    python3 code/evaluation/main.py

Or with explicit paths:
    python3 code/evaluation/main.py \\
        --predictions dataset/output.csv \\
        --ground-truth dataset/sample_messages.csv

Reports
-------
1. Action accuracy       (% exact match)
2. Message-type accuracy (% exact match)
3. Per-disagreement breakdown for action mismatches
4. Confusion matrix for actions
5. Formatted "Self-Evaluation Results" block for README.md
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PRED  = PROJECT_ROOT / "dataset" / "output.csv"
DEFAULT_GT    = PROJECT_ROOT / "dataset" / "sample_messages.csv"

ACTIONS = ["notify", "digest", "mute"]
TYPES   = [
    "personal", "urgent", "event", "payment", "business_update",
    "promotion", "greeting", "forward", "spam", "scam", "unknown",
]


def load(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        print(f"ERROR: {label} not found at {path}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(path)
    df["message_id"] = df["message_id"].astype(str)
    return df


def evaluate(pred_path: Path, gt_path: Path) -> None:
    pred = load(pred_path, "predictions")
    gt   = load(gt_path,   "ground truth")

    # ── Merge on message_id ───────────────────────────────────────────────────
    merged = gt.merge(pred, on="message_id", how="inner", suffixes=("_gt", "_pred"))

    n_gt   = len(gt)
    n_pred = len(pred)
    n_eval = len(merged)

    print(f"\n{'═'*60}")
    print(f"  Evaluation Summary")
    print(f"{'═'*60}")
    print(f"  Ground-truth messages : {n_gt}")
    print(f"  Prediction rows       : {n_pred}")
    print(f"  Matched (evaluated)   : {n_eval}")

    if n_eval == 0:
        print("\n  ✗  No overlapping message_ids — nothing to evaluate.")
        return

    missing_from_pred = set(gt["message_id"]) - set(pred["message_id"])
    if missing_from_pred:
        print(f"  ⚠  {len(missing_from_pred)} GT messages missing from predictions: "
              f"{sorted(missing_from_pred)[:5]}")

    # ── Action accuracy ───────────────────────────────────────────────────────
    action_match   = (merged["action_gt"] == merged["action_pred"])
    action_correct = action_match.sum()
    action_acc     = action_correct / n_eval * 100

    # ── Message-type accuracy ─────────────────────────────────────────────────
    type_match   = (merged["message_type_gt"] == merged["message_type_pred"])
    type_correct = type_match.sum()
    type_acc     = type_correct / n_eval * 100

    print(f"\n  Action accuracy       : {action_correct}/{n_eval}  ({action_acc:.1f}%)")
    print(f"  Message-type accuracy : {type_correct}/{n_eval}  ({type_acc:.1f}%)")

    # ── Confusion matrix for action ───────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Action Confusion Matrix  (rows=ground-truth, cols=predicted)")
    print(f"{'─'*60}")

    cm = pd.crosstab(
        merged["action_gt"].rename("GT \\ Pred"),
        merged["action_pred"],
        dropna=False,
    )
    # Ensure all action columns/rows exist
    for a in ACTIONS:
        if a not in cm.columns: cm[a] = 0
        if a not in cm.index:   cm.loc[a] = 0
    cm = cm.loc[[a for a in ACTIONS if a in cm.index],
                 [a for a in ACTIONS if a in cm.columns]]
    print(cm.to_string())

    # ── Per-disagreement breakdown ────────────────────────────────────────────
    disagreements = merged[~action_match].copy()
    print(f"\n{'─'*60}")
    print(f"  Action Disagreements ({len(disagreements)} total)")
    print(f"{'─'*60}")

    if disagreements.empty:
        print("  ✓  No action disagreements — perfect action accuracy!")
    else:
        for _, row in disagreements.iterrows():
            mid       = row["message_id"]
            gt_act    = row["action_gt"]
            pred_act  = row["action_pred"]
            gt_type   = row["message_type_gt"]
            pred_type = row.get("message_type_pred", "?")
            reason    = str(row.get("reason_pred", row.get("reason", ""))).strip()[:120]
            print(f"\n  [{mid}]")
            print(f"    Ground truth : action={gt_act}, type={gt_type}")
            print(f"    Predicted    : action={pred_act}, type={pred_type}")
            print(f"    Pred reason  : {reason}")

    # ── Message-type disagreements (brief) ────────────────────────────────────
    type_disagree = merged[~type_match & action_match]
    if not type_disagree.empty:
        print(f"\n{'─'*60}")
        print(f"  Type-only Disagreements (action correct, type wrong) — {len(type_disagree)}")
        print(f"{'─'*60}")
        for _, row in type_disagree.iterrows():
            print(f"  [{row['message_id']}]  GT={row['message_type_gt']}  "
                  f"Pred={row['message_type_pred']}")

    # ── README block ──────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  Self-Evaluation Results — paste into README.md")
    print(f"{'═'*60}")

    wrong_actions = len(disagreements)
    wrong_types   = (~type_match).sum()

    # Build mismatch summary (most common wrong pair)
    if not disagreements.empty:
        pairs = (
            disagreements.groupby(["action_gt", "action_pred"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
        )
        mismatch_lines = [
            f"    - GT={r.action_gt} → Pred={r.action_pred}: {r.n}"
            for _, r in pairs.iterrows()
        ]
        mismatch_str = "\n".join(mismatch_lines)
    else:
        mismatch_str = "    - None"

    readme_block = f"""\
## Self-Evaluation Results

- Evaluation set   : {n_eval} messages (sample_messages.csv)
- Action accuracy  : {action_acc:.1f}%  ({action_correct}/{n_eval} correct)
- Type accuracy    : {type_acc:.1f}%  ({type_correct}/{n_eval} correct)
- Action errors    : {wrong_actions}
- Type errors      : {wrong_types}
- Action mismatch breakdown:
{mismatch_str}
"""
    print()
    print(readme_block)
    print(f"{'═'*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate router predictions")
    parser.add_argument("--predictions",  type=Path, default=DEFAULT_PRED)
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GT)
    args = parser.parse_args()
    evaluate(args.predictions, args.ground_truth)


if __name__ == "__main__":
    main()
