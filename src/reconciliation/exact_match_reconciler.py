"""
exact_match_reconciler.py

Module 2 of the AI Finance Controller.

Purpose
-------
Take the two structured sources produced by Module 1 -- internal_invoices.csv
and gateway_payouts.csv -- and perform a deterministic, rule-based exact match
between them BEFORE any fuzzy/AI-based reconciliation is attempted downstream.

Why exact match first?
-----------------------
In any reconciliation pipeline, the majority of transactions are "easy":
they share a common payment reference and the amounts agree exactly. Running
expensive fuzzy matching (string similarity, ML models, LLM calls) on those
rows would be wasteful and would also introduce unnecessary risk of a wrong
match on a case that was already unambiguous. So Module 2's job is to:

    1. Filter out every row that can be matched with 100% certainty using
       simple, explainable rules (join key + exact amount).
    2. Hand back everything else -- unmatched or ambiguous rows -- as
       `unmatched_exceptions`, which is the ONLY input that later fuzzy/AI
       matching modules should ever need to look at.

Join key note
-------------
Module 1's schema does not literally use a column called "payment_id" --
the shared payment identifier is `invoice_id` on the invoice side and
`linked_invoice_id` on the payout side (the field a gateway would populate
with the merchant's invoice/payment reference). This script treats that
pair as the `payment_id` join key referenced in the spec.

Output
------
Two Python lists of dicts:
    - exact_matches:        rows where the payment_id join succeeded AND the
                             amounts agree exactly.
    - unmatched_exceptions: everything else (missing reference on either
                             side, or a reference match with a mismatched
                             amount).

Both lists are also written out to CSV for inspection / handoff to Module 3.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# The "payment_id" join key, as it exists on each side of the reconciliation.
INVOICE_KEY_COL = "invoice_id"
PAYOUT_KEY_COL = "linked_invoice_id"

# The amount columns being compared for an exact match.
INVOICE_AMOUNT_COL = "invoice_amount"
PAYOUT_AMOUNT_COL = "gross_amount"

# Amounts are floats written out of a prior CSV round-trip, so allow a
# tiny epsilon for floating point rounding -- this is still an "exact
# match" in accounting terms (sub-paisa/cent rounding), not a fuzzy one.
AMOUNT_TOLERANCE = 0.01


# --------------------------------------------------------------------------- #
# Step 1: Load
# --------------------------------------------------------------------------- #

def load_datasets(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Module 1's invoice and payout CSVs into DataFrames."""
    invoices = pd.read_csv(data_dir / "internal_invoices.csv")
    payouts = pd.read_csv(data_dir / "gateway_payouts.csv")
    return invoices, payouts


# --------------------------------------------------------------------------- #
# Step 2: Normalize the join key
# --------------------------------------------------------------------------- #

def normalize_key_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Clean up the payment_id join key so trivial formatting noise (stray
    whitespace, inconsistent case, or blank/"N/A" placeholders from a broken
    upstream reference) doesn't silently break an otherwise valid match --
    and, just as importantly, doesn't silently create a FALSE match.

    Blank/"N/A" references are normalized to pandas NA so they never
    accidentally join against each other.
    """
    df = df.copy()
    cleaned = df[column].astype(str).str.strip().str.upper()
    cleaned = cleaned.replace({"": pd.NA, "N/A": pd.NA, "NAN": pd.NA})
    df[column] = cleaned
    return df


# --------------------------------------------------------------------------- #
# Step 3: Deterministic exact match
# --------------------------------------------------------------------------- #

def exact_match_reconcile(
    invoices: pd.DataFrame, payouts: pd.DataFrame
) -> tuple[list[dict], list[dict]]:
    """
    Perform the deterministic exact-match pass.

    The filtering happens in two cheap, explainable stages so the "easy" rows
    are pulled out first and never touched again:

    Stage A - Reference join:
        An outer join on payment_id immediately separates rows into three
        buckets via pandas' merge indicator:
          - "both"        -> a payout exists for this invoice's payment_id
          - "left_only"    -> invoice has no matching payout at all
          - "right_only"   -> payout has no matching invoice at all
        The left_only/right_only rows are unambiguous exceptions -- there is
        nothing left to check, so they go straight to unmatched_exceptions
        without ever looking at amounts.

    Stage B - Amount check (only run on the "both" bucket):
        Of the rows where a reference match exists, only those whose invoice
        amount and payout gross amount agree exactly (within a sub-cent
        rounding tolerance) are kept as exact_matches. Any reference match
        with a mismatched amount is demoted to unmatched_exceptions, since a
        partial/split/incorrect payment needs human or fuzzy-matching
        attention rather than being auto-confirmed.
    """
    invoices = normalize_key_column(invoices, INVOICE_KEY_COL)
    payouts = normalize_key_column(payouts, PAYOUT_KEY_COL)

    # Stage A: outer join on the payment_id key. This one merge call does
    # the heavy lifting of separating "has a counterpart" from "orphaned".
    merged = invoices.merge(
        payouts,
        left_on=INVOICE_KEY_COL,
        right_on=PAYOUT_KEY_COL,
        how="outer",
        indicator=True,
        suffixes=("_invoice", "_payout"),
    )

    # Orphaned rows: no reference join was possible on either side.
    # These are the cheapest exceptions to identify -- filter them out first.
    orphaned_invoice = merged[merged["_merge"] == "left_only"]
    orphaned_payout = merged[merged["_merge"] == "right_only"]

    # Candidate rows: a payment_id reference match exists on both sides.
    candidates = merged[merged["_merge"] == "both"].copy()

    # Stage B: within the candidates, split by exact amount agreement.
    amount_diff = (candidates[INVOICE_AMOUNT_COL] - candidates[PAYOUT_AMOUNT_COL]).abs()
    is_exact = amount_diff <= AMOUNT_TOLERANCE

    exact_df = candidates[is_exact]
    amount_mismatch_df = candidates[~is_exact]

    # Assemble the final exact_matches list -- clean, unambiguous rows only.
    exact_matches = [
        {
            "payment_id": row[INVOICE_KEY_COL],
            "invoice_id": row[INVOICE_KEY_COL],
            "payout_id": row["payout_id"],
            "invoice_amount": row[INVOICE_AMOUNT_COL],
            "gross_amount": row[PAYOUT_AMOUNT_COL],
            "match_reason": "exact_reference_and_amount",
        }
        for _, row in exact_df.iterrows()
    ]

    # Assemble unmatched_exceptions from all three non-clean buckets,
    # each carrying a reason so downstream modules know why it landed here.
    unmatched_exceptions: list[dict] = []

    for _, row in orphaned_invoice.iterrows():
        unmatched_exceptions.append(
            {
                "payment_id": row[INVOICE_KEY_COL],
                "invoice_id": row[INVOICE_KEY_COL],
                "payout_id": None,
                "invoice_amount": row[INVOICE_AMOUNT_COL],
                "gross_amount": None,
                "exception_reason": "no_matching_payout_reference",
            }
        )

    for _, row in orphaned_payout.iterrows():
        unmatched_exceptions.append(
            {
                "payment_id": row[PAYOUT_KEY_COL],
                "invoice_id": None,
                "payout_id": row["payout_id"],
                "invoice_amount": None,
                "gross_amount": row[PAYOUT_AMOUNT_COL],
                "exception_reason": "no_matching_invoice_reference",
            }
        )

    for _, row in amount_mismatch_df.iterrows():
        unmatched_exceptions.append(
            {
                "payment_id": row[INVOICE_KEY_COL],
                "invoice_id": row[INVOICE_KEY_COL],
                "payout_id": row["payout_id"],
                "invoice_amount": row[INVOICE_AMOUNT_COL],
                "gross_amount": row[PAYOUT_AMOUNT_COL],
                "exception_reason": "reference_matched_but_amount_mismatch",
            }
        )

    return exact_matches, unmatched_exceptions


# --------------------------------------------------------------------------- #
# Step 4: Persist results
# --------------------------------------------------------------------------- #

def save_results(
    exact_matches: list[dict], unmatched_exceptions: list[dict], out_dir: Path
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(exact_matches).to_csv(out_dir / "exact_matches.csv", index=False)
    pd.DataFrame(unmatched_exceptions).to_csv(out_dir / "unmatched_exceptions.csv", index=False)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 2: deterministic exact-match reconciliation.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "raw"),
        help="Directory containing Module 1's internal_invoices.csv and gateway_payouts.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "processed"),
        help="Directory to write exact_matches.csv and unmatched_exceptions.csv.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    invoices, payouts = load_datasets(data_dir)
    exact_matches, unmatched_exceptions = exact_match_reconcile(invoices, payouts)
    save_results(exact_matches, unmatched_exceptions, out_dir)

    total = len(invoices) + len(payouts)
    print(f"Loaded {len(invoices)} invoices and {len(payouts)} payouts ({total} rows total).")
    print(f"Exact matches:        {len(exact_matches)}  -> {out_dir / 'exact_matches.csv'}")
    print(f"Unmatched exceptions: {len(unmatched_exceptions)}  -> {out_dir / 'unmatched_exceptions.csv'}")


if __name__ == "__main__":
    main()