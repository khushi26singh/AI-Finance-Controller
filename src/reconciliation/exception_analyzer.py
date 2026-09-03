"""
exception_analyzer.py

Rule-mining script for the AI Finance Controller.

Purpose
-------
Mines a concrete, data-driven answer to a rule-tuning question: when the LLM
looked at an exception originally flagged by Module 2 as
'reference_matched_but_amount_mismatch' and still couldn't confirm a match,
how big was the amount gap it was actually rejecting? Averaging that across
real runs tells you whether the deterministic pre-LLM filter's
MASSIVE_VARIANCE_THRESHOLD_PCT (10%, see apply_advanced_rules() in
llm_resolver.py) is well-calibrated, too loose, or too tight.

Input
-----
Expects the CSV a user downloads from the dashboard's "Needs Human Review"
tab (build_review_table() in src/dashboard/app.py, saved by default as
needs_human_review.csv) -- that is the actual file in this codebase with
'original_exception_reason' and 'ai_reasoning' columns; there is no
'ai match export.csv' produced anywhere in the pipeline. If you're using a
different filename, pass it explicitly with --input.

Interaction with the pre-LLM filter (important)
-------------------------------------------------
apply_advanced_rules() in llm_resolver.py can now resolve an exception
*before* the LLM ever sees it (e.g. "Severe Amount Mismatch: All candidate
gaps exceed the 10% gateway fee threshold."). Rows short-circuited that way
still carry exception_reason == 'reference_matched_but_amount_mismatch' from
Module 2, but their ai_reasoning text is OUR deterministic boilerplate, not
the model's own computed gap -- mining a percentage out of "...exceed the
10% gateway fee threshold." would just measure our own constant back at us,
not a real LLM judgment. This script explicitly excludes rows carrying that
boilerplate so the reported stats reflect genuine LLM reasoning only.

Usage
-----
    python src/reconciliation/exception_analyzer.py
    python src/reconciliation/exception_analyzer.py --input path/to/export.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

TARGET_EXCEPTION_REASON = "reference_matched_but_amount_mismatch"

# The dashboard's actual export filename is checked first; the name given in
# the task spec is also checked as a fallback in case that's what a user
# renamed their download to.
DEFAULT_INPUT_CANDIDATES = [
    Path("data/processed/needs_human_review.csv"),
    Path("data/processed/ai match export.csv"),
    Path("data/processed/ai_match_export.csv"),
]

# Boilerplate strings written by apply_advanced_rules()'s deterministic
# short-circuit (llm_resolver.py) and by mock mode -- never by the LLM
# itself. Rows starting with these are excluded before gap-mining so a
# hardcoded threshold (e.g. "10%") never gets counted as a real AI judgment.
NON_LLM_REASONING_PREFIXES = (
    "Outstanding Payment:",
    "Severe Amount Mismatch:",
    "Data Quality Issue:",
    "Mock mode:",
    "Skipped:",
)

# Matches a (possibly decimal) percentage mentioned in free text, e.g.:
#   "The gap is approximately 44%" -> "44"
#   "roughly a 12.5% variance"      -> "12.5"
PERCENT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")


def resolve_input_path(explicit_path: str | None) -> Path:
    if explicit_path:
        return Path(explicit_path)
    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate
    return DEFAULT_INPUT_CANDIDATES[0]  # fall through to a clear FileNotFoundError below


def load_ai_export(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path}'. Export it from the dashboard's "
            "'Needs Human Review' tab (Download button), or pass --input "
            "with the correct path."
        )
    return pd.read_csv(path)


def is_llm_authored(reasoning_text: str) -> bool:
    """True if this reasoning text plausibly came from the LLM, not our own
    deterministic pre-filter or mock-mode boilerplate."""
    if not isinstance(reasoning_text, str) or not reasoning_text.strip():
        return False
    return not reasoning_text.startswith(NON_LLM_REASONING_PREFIXES)


def extract_percentage_gap(reasoning_text: str) -> int | None:
    """
    Extract the first percentage mentioned in an LLM reasoning string, as an
    integer (per spec -- e.g. "approximately 44%" -> 44; a decimal like
    "12.5%" is truncated to 12).

    Returns None if no percentage is present in the text.
    """
    if not isinstance(reasoning_text, str):
        return None
    match = PERCENT_PATTERN.search(reasoning_text)
    if not match:
        return None
    return int(float(match.group(1)))


def analyze(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to amount-mismatch rows with genuine LLM reasoning, then
    extract the percentage gap each one mentions."""
    for required_col in ("original_exception_reason", "ai_reasoning"):
        if required_col not in df.columns:
            raise ValueError(
                f"Expected column '{required_col}' not found. Columns present: "
                f"{list(df.columns)}. This script expects the dashboard's "
                "Needs Human Review export schema."
            )

    filtered = df[df["original_exception_reason"] == TARGET_EXCEPTION_REASON].copy()
    filtered["is_llm_authored"] = filtered["ai_reasoning"].apply(is_llm_authored)
    filtered["extracted_gap_pct"] = filtered["ai_reasoning"].apply(extract_percentage_gap)
    return filtered


def print_summary_report(filtered: pd.DataFrame) -> None:
    total_rows = len(filtered)
    llm_authored = filtered[filtered["is_llm_authored"]]
    non_llm_count = total_rows - len(llm_authored)
    with_gap = llm_authored.dropna(subset=["extracted_gap_pct"])

    print("=" * 64)
    print("EXCEPTION ANALYZER -- Amount Mismatch Gap Report")
    print("=" * 64)
    print(f"Exception reason filtered: '{TARGET_EXCEPTION_REASON}'")
    print(f"Rows matching that reason:                    {total_rows}")
    print(f"  -- excluded (pre-LLM filter / mock boilerplate): {non_llm_count}")
    print(f"  -- genuine LLM-authored reasoning:               {len(llm_authored)}")
    print(f"  -- of those, with an extractable percentage:     {len(with_gap)}")

    if len(llm_authored) and len(with_gap) < len(llm_authored):
        print(
            f"  (note: {len(llm_authored) - len(with_gap)} LLM-authored row(s) had no "
            "parseable percentage -- excluded from the stats below)"
        )

    print()
    if with_gap.empty:
        print("No LLM-computed percentage gaps available to summarize.")
        print("=" * 64)
        return

    gaps = with_gap["extracted_gap_pct"]
    print(f"Average AI-rejected gap: {gaps.mean():.1f}%")
    print(f"Minimum AI-rejected gap: {gaps.min()}%")
    print(f"Maximum AI-rejected gap: {gaps.max()}%")
    print("=" * 64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine AI-rejected amount-mismatch percentage gaps from a reviewed exceptions export."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to the reviewed-exceptions CSV (defaults to data/processed/needs_human_review.csv).",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.input)
    df = load_ai_export(input_path)
    filtered = analyze(df)
    print(f"Loaded {len(df)} rows from {input_path}\n")
    print_summary_report(filtered)


if __name__ == "__main__":
    main()