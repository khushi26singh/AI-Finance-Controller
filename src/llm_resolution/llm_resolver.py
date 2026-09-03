"""
llm_resolver.py

Module 3 of the AI Finance Controller.

Purpose
-------
Module 2 already separated the "easy" rows (exact_matches) from the hard
ones (unmatched_exceptions). This module's only job is to work through
`unmatched_exceptions` and try to explain each one using semantic reasoning
that simple rule-based joins can't do -- specifically:

    1. Match an invoice/payout to a messy, unstructured bank statement
       narration (e.g. "NEFT-HDFC0000123-RAZORPAY SETTLEMENT-UTR389935567052")
       even when Module 2's exact join couldn't find one.
    2. For cases where a reference match exists but the amount doesn't line
       up, reason about *why* -- is the gap explained by a standard payment
       gateway MDR (Merchant Discount Rate) fee plus tax on that fee, or is
       it something that needs a human to look at?

Design principle: deterministic pre-filtering, LLM for judgment only
----------------------------------------------------------------------
LLM calls are the most expensive and least deterministic step in this
pipeline, so this module never lets the model search blindly. For every
exception we first narrow the field ourselves:

    - If we have a UTR/reference number (from the linked gateway payout),
      we do an exact substring search against bank narrations first --
      this is still deterministic, just delayed from Module 2 because the
      reference lives in a different column than Module 2 checked.
    - Otherwise, we shortlist bank statement rows by amount proximity and
      a payout/invoice date window, and only send that short candidate
      list (typically <= 5 rows) to the LLM.

We also compute a rule-based "fee hypothesis" for every amount gap (does
the gap fall inside the typical MDR-fee-plus-tax range?) BEFORE calling the
LLM, and pass that hypothesis into the prompt as context. The LLM's job is
then to weigh that hypothesis against the semantic evidence in the
narration and candidate list -- not to invent a number from scratch.

Output
------
A single structured JSON report, `resolution_report.json`, with one entry
per exception containing: match status, the matched bank transaction (if
any), a confidence score, the amount-gap diagnosis, and the model's
reasoning -- ready for a human reviewer or an auto-close policy to consume.

Provider
--------
By default this module uses Groq's free API tier (https://console.groq.com),
which gives no-credit-card access to open models like Llama 3.3 70B with
OpenAI-compatible tool calling -- no paid credits required. Google's Gemini
API (--provider gemini) also has a genuinely free tier, and Anthropic's API
is supported via --provider anthropic if you have a paid key for it.
Use --mock to run the full pipeline (shortlisting + fee-hypothesis math)
without making any API calls at all, useful for local testing or CI.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import anthropic
except ImportError:  # pragma: no cover - surfaced clearly at runtime instead
    anthropic = None

try:
    import groq
except ImportError:  # pragma: no cover - surfaced clearly at runtime instead
    groq = None

try:
    from google import genai
    from google.genai import types as genai_types
except ImportError:  # pragma: no cover - surfaced clearly at runtime instead
    genai = None
    genai_types = None

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

DEFAULT_PROVIDER = "gemini"  # free tier, no credit card required
ANTHROPIC_MODEL_NAME = "claude-sonnet-5"
GROQ_MODEL_NAME = "openai/gpt-oss-120b"  # current Groq free-tier model with tool-calling support
# Note: Groq's free-tier model lineup changes periodically (e.g. llama-3.3-70b-versatile
# was retired on 2026-08-16). If this model 404s in the future, check the current list with:
#   curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"
# or see https://console.groq.com/docs/models, and update GROQ_MODEL_NAME accordingly.

GEMINI_MODEL_NAME = "gemini-flash-latest"  # Google's stable alias, always points at their current default Flash model
# Note: pinned Gemini model IDs (e.g. gemini-2.5-flash) can appear in
# models.list() while still being rejected for actual generateContent calls
# on a given key/tier -- catalog listing and invocation access aren't the
# same thing. The "gemini-flash-latest" alias sidesteps this by always
# resolving to whatever Flash model Google currently recommends. The
# resolve_working_model() pre-flight check below still guards against the
# alias itself being unavailable. If you need a specific pinned version
# instead, override with --model <model-id> (see https://ai.google.dev/gemini-api/docs/models).

# Gemini's free tier is tightly rate-limited -- observed as low as 5
# requests/minute *per model* on some accounts. These control how the
# resolution loop paces itself and recovers from 429s instead of crashing.
GEMINI_MIN_SECONDS_BETWEEN_CALLS = 13.0  # ~5 requests/minute with a safety margin
MAX_RATE_LIMIT_RETRIES = 3

MAX_CANDIDATES_PER_EXCEPTION = 5
CANDIDATE_AMOUNT_TOLERANCE_PCT = 0.06   # +/-6% covers gateway fees + tax
CANDIDATE_DATE_WINDOW_DAYS = 15          # payout usually settles within ~10 days

# Typical Indian payment-gateway economics, used only as a deterministic
# sanity check that gets shown to the LLM, not as a substitute for its
# judgment.
TYPICAL_MDR_FEE_PCT_RANGE = (1.5, 3.0)   # gateway fee as % of gross amount
GST_ON_FEE_PCT = 18.0                    # GST charged on top of the fee itself


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #

@dataclass
class FeeHypothesis:
    gap_amount: float
    gap_pct_of_invoice: float
    plausible_mdr_and_tax: bool
    expected_gap_range_pct: tuple[float, float]
    note: str


@dataclass
class ExceptionResolution:
    payment_id: str
    exception_reason: str
    match_status: str                 # "matched" | "no_match_found" | "needs_human_review"
    matched_bank_txn_id: str | None
    confidence: float                 # 0.0 - 1.0
    amount_gap: float | None
    gap_diagnosis: str                # e.g. "mdr_fee_and_tax" | "unexplained" | "n/a"
    reasoning: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Step 1: Load Module 2's output + Module 1's raw context
# --------------------------------------------------------------------------- #

def load_inputs(processed_dir: Path, raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the unmatched exceptions from Module 2, plus the raw Module 1
    tables needed to give the LLM enough context to reason (invoice dates,
    payout UTR numbers, and the messy bank narrations themselves)."""
    exceptions = pd.read_csv(processed_dir / "unmatched_exceptions.csv")
    invoices = pd.read_csv(raw_dir / "internal_invoices.csv")
    payouts = pd.read_csv(raw_dir / "gateway_payouts.csv")
    bank = pd.read_csv(raw_dir / "bank_statements.csv")
    return exceptions, invoices, payouts, bank


def enrich_exceptions(
    exceptions: pd.DataFrame, invoices: pd.DataFrame, payouts: pd.DataFrame
) -> pd.DataFrame:
    """Attach the invoice_date and utr_number context that Module 2's output
    doesn't carry by itself but that we need for shortlisting candidates."""
    enriched = exceptions.merge(
        invoices[["invoice_id", "invoice_date", "customer_name"]],
        on="invoice_id",
        how="left",
    )
    enriched = enriched.merge(
        payouts[["payout_id", "utr_number", "payout_date"]],
        on="payout_id",
        how="left",
    )
    return enriched


# --------------------------------------------------------------------------- #
# Step 2: Deterministic candidate shortlisting (cheap, before any LLM call)
# --------------------------------------------------------------------------- #

def shortlist_candidates(row: pd.Series, bank: pd.DataFrame) -> list[dict]:
    """
    Narrow the entire bank statement down to a handful of plausible rows for
    this specific exception, using only cheap deterministic checks:

      1. If we know the payout's UTR number, an exact substring match against
         the narration is definitive -- return it alone with full confidence.
      2. Otherwise, fall back to amount-proximity + date-window filtering so
         the LLM only ever sees a short list of realistic candidates, never
         the whole ledger.
    """
    bank_credits = bank[bank["txn_type"] == "CR"]

    utr = row.get("utr_number")
    if isinstance(utr, str) and utr and utr.lower() != "nan":
        utr_hits = bank_credits[bank_credits["description"].str.contains(utr, case=False, na=False)]
        if not utr_hits.empty:
            return [
                {**hit, "match_basis": "exact_utr_substring"}
                for hit in utr_hits.to_dict(orient="records")
            ]

    # No UTR to anchor on (or it wasn't found) -> fuzzy shortlist.
    reference_amount = row.get("invoice_amount")
    if pd.isna(reference_amount):
        reference_amount = row.get("gross_amount")
    if pd.isna(reference_amount):
        return []

    anchor_date_str = row.get("invoice_date") or row.get("payout_date")
    anchor_date = pd.to_datetime(anchor_date_str, errors="coerce") if anchor_date_str else None

    lower = reference_amount * (1 - CANDIDATE_AMOUNT_TOLERANCE_PCT)
    upper = reference_amount  # a valid payout is always <= invoice (fees are deducted, never added)
    amount_matches = bank_credits[bank_credits["amount"].between(lower, upper)].copy()

    if anchor_date is not None and not amount_matches.empty:
        amount_matches["txn_date_parsed"] = pd.to_datetime(amount_matches["txn_date"], errors="coerce")
        window_start = anchor_date - pd.Timedelta(days=2)
        window_end = anchor_date + pd.Timedelta(days=CANDIDATE_DATE_WINDOW_DAYS)
        amount_matches = amount_matches[
            amount_matches["txn_date_parsed"].between(window_start, window_end)
        ]

    amount_matches["distance_from_reference"] = (reference_amount - amount_matches["amount"]).abs()
    amount_matches = amount_matches.sort_values("distance_from_reference").head(MAX_CANDIDATES_PER_EXCEPTION)

    return [
        {**hit, "match_basis": "amount_and_date_proximity"}
        for hit in amount_matches.drop(columns=["txn_date_parsed", "distance_from_reference"], errors="ignore").to_dict(
            orient="records"
        )
    ]


# --------------------------------------------------------------------------- #
# Step 3: Deterministic fee/tax hypothesis (cheap math, before any LLM call)
# --------------------------------------------------------------------------- #

def compute_fee_hypothesis(invoice_amount: float | None, candidate_amount: float | None) -> FeeHypothesis | None:
    """
    Check whether the gap between the invoice amount and a candidate credited
    amount falls inside the range a standard gateway MDR fee plus GST on that
    fee would produce. This is pure arithmetic -- it exists to give the LLM a
    grounded, checkable hypothesis instead of asking it to estimate fees from
    nothing.
    """
    if invoice_amount is None or candidate_amount is None or pd.isna(invoice_amount) or pd.isna(candidate_amount):
        return None

    gap_amount = round(invoice_amount - candidate_amount, 2)
    gap_pct = round((gap_amount / invoice_amount) * 100, 3) if invoice_amount else 0.0

    fee_low, fee_high = TYPICAL_MDR_FEE_PCT_RANGE
    expected_low = fee_low * (1 + GST_ON_FEE_PCT / 100)
    expected_high = fee_high * (1 + GST_ON_FEE_PCT / 100)

    plausible = 0 <= gap_pct <= expected_high * 1.15  # small buffer for rounding

    return FeeHypothesis(
        gap_amount=gap_amount,
        gap_pct_of_invoice=gap_pct,
        plausible_mdr_and_tax=plausible,
        expected_gap_range_pct=(round(expected_low, 3), round(expected_high, 3)),
        note=(
            f"Gap is {gap_pct}% of invoice amount; a typical gateway fee "
            f"({fee_low}-{fee_high}%) plus {GST_ON_FEE_PCT}% GST on that fee "
            f"would produce a gap of roughly {round(expected_low, 2)}-{round(expected_high, 2)}%."
        ),
    )


# Any candidate whose gap exceeds this is economically implausible as a fee/tax
# explanation -- real MDR + GST tops out well under this (see compute_fee_hypothesis),
# so a gap this large is not something an LLM needs to "reason" about.
MASSIVE_VARIANCE_THRESHOLD_PCT = 0.10  # 10%


def _no_llm_result(
    match_status: str,
    reasoning: str,
    matched_bank_txn_id: str | None = None,
    confidence: float = 0.0,
    gap_diagnosis: str = "unexplained",
    amount_gap: float | None = None,
) -> dict:
    """Consistent return shape for every apply_advanced_rules() outcome that
    does NOT require an LLM call -- whether that's a rejection (needs_human_review)
    or a deterministic auto-match (matched)."""
    return {
        "needs_llm": False,
        "match_status": match_status,
        "reasoning": reasoning,
        "matched_bank_txn_id": matched_bank_txn_id,
        "confidence": confidence,
        "gap_diagnosis": gap_diagnosis,
        "amount_gap": amount_gap,
        "plausible_candidates": [],
    }


def apply_advanced_rules(
    invoice_amount: float | None, candidate_bank_txns: list[dict] | None
) -> dict:
    """
    Deterministic pre-LLM filter. Runs immediately after candidate shortlisting
    and before any LLM call (mock or live) to cut exceptions that don't need --
    and shouldn't get -- semantic reasoning spent on them.

    Three rules, applied in order:

    1. Zero-Candidate Short-Circuit: no candidates at all means there's nothing
       for the LLM to reason over. Don't call it -- go straight to human review.
    2. Massive Variance Filter: if every remaining candidate's amount gap is
       economically implausible as a fee/tax explanation (> 10%), asking the
       LLM to pick one anyway just risks it rationalizing a bad match. Skip the
       call and surface the gap directly instead.
    3. High-Confidence Deterministic Auto-Match: if exactly one candidate is
       anchored by an EXACT reference match (UTR substring -- the strongest
       evidence a bank credit belongs to this specific invoice/payout, not
       just amount/date coincidence) AND its gap is tight enough to be an
       obvious gateway fee (inside the real MDR+GST range, not just under the
       generous 10% ceiling), there is no genuine semantic ambiguity left for
       an LLM to adjudicate. Auto-confirm it deterministically -- same
       "deterministic first" principle as Module 2's exact match, just applied
       one layer deeper, where the reference match is exact but the amount
       needed a fee explanation to close the loop.

    Returns a dict:
        needs_llm             -- False if a rule fired and reconciliation is
                                  already decided; True if the LLM should run.
        match_status           -- set only when needs_llm is False. Either
                                  "needs_human_review" (rules 1/2) or
                                  "matched" (rule 3).
        reasoning               -- set only when needs_llm is False.
        matched_bank_txn_id    -- set only when match_status == "matched".
        confidence              -- set only when match_status == "matched".
        gap_diagnosis           -- set only when match_status == "matched".
        amount_gap              -- set only when match_status == "matched".
        plausible_candidates   -- candidates worth sending to the LLM (empty
                                  list when needs_llm is False).
    """
    # Rule 1: Zero-Candidate Short-Circuit.
    if not candidate_bank_txns:
        return _no_llm_result(
            "needs_human_review",
            "Outstanding Payment: No candidate bank transactions found.",
        )

    # Can't compute a percentage gap without a valid, non-zero invoice amount.
    # Fail open to the LLM rather than divide by zero or silently drop the row.
    if invoice_amount is None or pd.isna(invoice_amount) or invoice_amount == 0:
        return {
            "needs_llm": True,
            "match_status": None,
            "reasoning": None,
            "plausible_candidates": candidate_bank_txns,
        }

    # Rule 2: Massive Variance Filter.
    gaps_pct = []
    for txn in candidate_bank_txns:
        bank_amount = txn.get("amount")
        if bank_amount is None:
            continue
        gaps_pct.append(abs(invoice_amount - bank_amount) / invoice_amount)

    if not gaps_pct:
        # Candidates exist but none carried a usable amount -- a data quality
        # gap, not a "massive variance" verdict. Don't mislabel it as one;
        # route to review honestly instead of guessing.
        return _no_llm_result(
            "needs_human_review",
            "Data Quality Issue: candidate bank transactions had no usable amount field.",
        )

    if all(gap > MASSIVE_VARIANCE_THRESHOLD_PCT for gap in gaps_pct):
        return _no_llm_result(
            "needs_human_review",
            "Severe Amount Mismatch: All candidate gaps exceed the "
            f"{MASSIVE_VARIANCE_THRESHOLD_PCT:.0%} gateway fee threshold.",
        )

    # Rule 3: High-Confidence Deterministic Auto-Match.
    utr_anchored = [
        txn for txn in candidate_bank_txns if txn.get("match_basis") == "exact_utr_substring"
    ]
    if len(utr_anchored) == 1:
        txn = utr_anchored[0]
        hyp = compute_fee_hypothesis(invoice_amount, txn.get("amount"))
        if hyp is not None and hyp.plausible_mdr_and_tax:
            return _no_llm_result(
                "matched",
                (
                    f"Rule-Resolved: bank transaction {txn['txn_id']} is linked by an exact "
                    f"payment-reference (UTR) match, and the {hyp.gap_pct_of_invoice:.1f}% gap "
                    "falls within the expected gateway fee + GST range -- no semantic ambiguity "
                    "for the LLM to adjudicate, so it was never called."
                ),
                matched_bank_txn_id=txn["txn_id"],
                confidence=0.95,
                gap_diagnosis="mdr_fee_and_tax",
                amount_gap=hyp.gap_amount,
            )

    return {
        "needs_llm": True,
        "match_status": None,
        "reasoning": None,
        "plausible_candidates": candidate_bank_txns,
    }


# --------------------------------------------------------------------------- #
# Step 4: LLM semantic reasoning (only runs on the shortlisted candidates)
# --------------------------------------------------------------------------- #

RESOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "match_status": {
            "type": "string",
            "enum": ["matched", "no_match_found", "needs_human_review"],
            "description": "Overall outcome of the reasoning.",
        },
        "matched_bank_txn_id": {
            "type": ["string", "null"],
            "description": "txn_id of the bank statement line that best matches, or null.",
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this resolution, from 0.0 to 1.0.",
        },
        "gap_diagnosis": {
            "type": "string",
            "enum": ["mdr_fee_and_tax", "partial_payment", "unrelated_transaction", "unexplained", "n/a"],
            "description": "Best explanation for any amount gap between the invoice and the matched transaction.",
        },
        "reasoning": {
            "type": "string",
            "description": "A concise (2-3 sentence) explanation of why this conclusion was reached, referencing the specific narration text or amount evidence used.",
        },
    },
    "required": ["match_status", "matched_bank_txn_id", "confidence", "gap_diagnosis", "reasoning"],
}

TOOL_NAME = "emit_resolution"
TOOL_DESCRIPTION = "Report the resolution of a single unmatched finance exception."

# Anthropic's tool format (Messages API).
ANTHROPIC_TOOL = {
    "name": TOOL_NAME,
    "description": TOOL_DESCRIPTION,
    "input_schema": RESOLUTION_SCHEMA,
}

# Groq's tool format is OpenAI-compatible (chat.completions API).
GROQ_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": RESOLUTION_SCHEMA,
    },
}

# Gemini's function-declaration schema is a subset of OpenAPI 3.0 and uses
# uppercase type names plus "nullable" instead of JSON Schema's type-array
# union (["string", "null"]) used above for Anthropic/Groq.
GEMINI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "match_status": {
            "type": "STRING",
            "enum": ["matched", "no_match_found", "needs_human_review"],
            "description": "Overall outcome of the reasoning.",
        },
        "matched_bank_txn_id": {
            "type": "STRING",
            "nullable": True,
            "description": "txn_id of the bank statement line that best matches, or null.",
        },
        "confidence": {
            "type": "NUMBER",
            "description": "Confidence in this resolution, from 0.0 to 1.0.",
        },
        "gap_diagnosis": {
            "type": "STRING",
            "enum": ["mdr_fee_and_tax", "partial_payment", "unrelated_transaction", "unexplained", "n/a"],
            "description": "Best explanation for any amount gap between the invoice and the matched transaction.",
        },
        "reasoning": {
            "type": "STRING",
            "description": "A concise (2-3 sentence) explanation of why this conclusion was reached, referencing the specific narration text or amount evidence used.",
        },
    },
    "required": ["match_status", "matched_bank_txn_id", "confidence", "gap_diagnosis", "reasoning"],
}


def build_prompt(row: pd.Series, candidates: list[dict], fee_hypotheses: dict[str, FeeHypothesis]) -> str:
    invoice_amount = row.get("invoice_amount")
    gross_amount = row.get("gross_amount")

    candidate_lines = []
    for c in candidates:
        hyp = fee_hypotheses.get(c["txn_id"])
        hyp_text = f" | fee_hypothesis: {hyp.note}" if hyp else ""
        candidate_lines.append(
            f"  - txn_id={c['txn_id']}, date={c['txn_date']}, amount={c['amount']}, "
            f"narration=\"{c['description']}\", found_via={c['match_basis']}{hyp_text}"
        )
    candidate_block = "\n".join(candidate_lines) if candidate_lines else "  (no candidate bank transactions found)"

    return f"""You are a finance reconciliation analyst. Resolve the following exception
that a deterministic rule-based matcher could not close automatically.

EXCEPTION
  payment_id: {row.get('payment_id')}
  exception_reason: {row.get('exception_reason')}
  invoice_id: {row.get('invoice_id')}
  customer_name: {row.get('customer_name')}
  invoice_amount: {invoice_amount}
  linked_payout_gross_amount: {gross_amount}

CANDIDATE BANK STATEMENT LINES (already pre-filtered by amount/date/UTR):
{candidate_block}

TASK
1. Decide whether one of the candidates is genuinely the same real-world payment as this
   invoice/payout, using the narration text as your primary evidence (bank narrations are
   messy: UPI/NEFT/IMPS/RTGS codes, gateway names, UTR numbers, inconsistent casing/whitespace).
2. If a match exists and its amount is lower than the invoice amount, decide whether the gap
   is explained by a standard payment gateway fee (MDR) plus tax on that fee, referencing the
   fee_hypothesis figures given for context, or whether it looks like something else (a partial
   payment, an unrelated transaction, or a genuinely unexplained gap).
3. If no candidate is a plausible match, say so rather than forcing one.

Call the emit_resolution tool with your conclusion."""


# --------------------------------------------------------------------------- #
# Model auto-discovery -- free-tier model lineups on Gemini and Groq get
# renamed/retired often enough that hardcoding one name isn't reliable. This
# checks what's actually available to the key BEFORE spending any resolution
# calls on a model that might already be gone, and falls back automatically.
# --------------------------------------------------------------------------- #

def list_available_models(provider: str, client: "anthropic.Anthropic | groq.Groq | genai.Client") -> list[str]:
    """Return the model IDs this API key can currently use for chat/tool-calling."""
    try:
        if provider == "gemini":
            names = []
            for m in client.models.list():
                actions = getattr(m, "supported_actions", None)
                # The Gemini Developer API doesn't always populate
                # supported_actions (it's more reliably set on Vertex AI) --
                # treat "not reported" as "assume usable" rather than
                # silently excluding every model when the field is empty.
                if actions is None or "generateContent" in actions:
                    # SDK returns e.g. "models/gemini-2.5-flash" -- strip the prefix.
                    names.append(m.name.split("/")[-1])
            return names

        if provider == "groq":
            response = client.models.list()
            return [m.id for m in response.data]

    except Exception as exc:
        # Listing is best-effort, but silently swallowing the error here
        # made a real problem (bad key, network block, wrong SDK version)
        # invisible -- print it so it's actually diagnosable.
        print(f"Note: could not list available {provider} models ({type(exc).__name__}: {exc}). Trying the configured model as-is.")
        return []

    return []


def pick_fallback_model(available: list[str]) -> str | None:
    """
    Heuristically pick a reasonable general-purpose chat/tool-calling model
    from what's actually available, favoring fast "flash"/general chat
    models and avoiding specialized variants (image/audio/embedding/preview)
    that don't fit this text + tool-calling use case.
    """
    if not available:
        return None

    exclude_markers = ["preview", "exp", "image", "audio", "tts", "embed", "vision", "live", "robotics", "vl"]
    candidates = [n for n in available if not any(mark in n.lower() for mark in exclude_markers)]
    pool = candidates or available

    # Prefer flash/general chat-style names; sort so the (roughly) newest
    # version sorts last -- not perfectly reliable across naming schemes,
    # but a reasonable best-effort default.
    preferred = [n for n in pool if any(tag in n.lower() for tag in ["flash", "versatile", "instant"])]
    ranked = sorted(preferred or pool)
    return ranked[-1]


def resolve_working_model(
    provider: str, client: "anthropic.Anthropic | groq.Groq | genai.Client | None", requested_model: str
) -> str:
    """
    Confirm the requested model is actually available before running the
    resolution loop. If it isn't, auto-fall-back to the best current match
    and warn loudly, rather than failing partway through a batch of
    exceptions or requiring a manual --model lookup every time a provider
    renames its free-tier lineup.
    """
    if provider not in ("gemini", "groq") or client is None:
        return requested_model  # Anthropic model IDs are stable enough not to bother checking.

    available = list_available_models(provider, client)
    if requested_model in available:
        return requested_model
    if not available:
        # list_available_models already printed why (exception, or the
        # provider genuinely returned zero usable models) -- proceed
        # optimistically with the requested model and let the real call
        # surface any error.
        return requested_model

    fallback = pick_fallback_model(available)
    if fallback:
        print(
            f"Note: model '{requested_model}' is not available on {provider} for this API key. "
            f"Auto-selected '{fallback}' instead (detected from {len(available)} available models)."
        )
        return fallback

    return requested_model


def get_client(provider: str) -> "anthropic.Anthropic | groq.Groq | genai.Client":
    """
    Build the LLM client for the chosen provider.

    Groq and Gemini both offer genuinely free API tiers (no credit card)
    while still supporting the forced tool-calling pattern used here for
    reliable structured output. Anthropic is supported too if you have a
    paid key for it.
    """
    if provider == "groq":
        if groq is None:
            raise SystemExit("The 'groq' package is not installed. Run: pip install groq")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise SystemExit(
                "GROQ_API_KEY environment variable is not set.\n"
                "Get a free key (no credit card required) at https://console.groq.com/keys\n"
                "then set it with: export GROQ_API_KEY=\"your-key-here\"\n"
                "Or re-run with --mock to test the pipeline without any live LLM calls."
            )
        return groq.Groq(api_key=api_key)

    if provider == "gemini":
        if genai is None:
            raise SystemExit("The 'google-genai' package is not installed. Run: pip install google-genai")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY environment variable is not set.\n"
                "Get a free key (no credit card required) at https://aistudio.google.com/app/apikey\n"
                "then set it with: export GEMINI_API_KEY=\"your-key-here\"\n"
                "Or re-run with --mock to test the pipeline without any live LLM calls."
            )
        return genai.Client(api_key=api_key)

    if provider == "anthropic":
        if anthropic is None:
            raise SystemExit("The 'anthropic' package is not installed. Run: pip install anthropic")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise SystemExit(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it, or re-run with --provider gemini or --provider groq for a free "
                "alternative, or --mock to test the pipeline without live LLM calls."
            )
        return anthropic.Anthropic(api_key=api_key)

    raise SystemExit(f"Unknown provider: {provider!r}. Use 'gemini', 'groq', or 'anthropic'.")


_last_call_time: dict[str, float] = {}


def _throttle_before_call(provider: str) -> None:
    """
    Proactively pace requests to stay under known tight free-tier limits
    (observed as low as 5 requests/minute per model on Gemini's free tier),
    instead of firing 34 requests back-to-back and immediately hitting 429s.
    """
    if provider != "gemini":
        return
    now = time.monotonic()
    last = _last_call_time.get(provider, 0.0)
    wait = GEMINI_MIN_SECONDS_BETWEEN_CALLS - (now - last)
    if wait > 0:
        time.sleep(wait)
    _last_call_time[provider] = time.monotonic()


def _parse_retry_delay_seconds(error_text: str, attempt: int) -> float:
    """
    Extract a provider-suggested retry delay from an error message if one is
    present (rate-limit errors usually include one). Otherwise fall back to
    exponential backoff -- appropriate for transient network errors (dropped
    connections, timeouts, brief 5xx blips) that don't come with a suggested
    wait time.
    """
    match = re.search(r"retry in ([\d.]+)\s*s", error_text, re.IGNORECASE)
    if not match:
        match = re.search(r"retryDelay['\"]?\s*[:=]\s*['\"]?([\d.]+)s", error_text)
    if match:
        return float(match.group(1)) + 1.0  # small safety buffer
    return min(5.0 * (2 ** attempt), 30.0)  # 5s, 10s, 20s, capped at 30s


def call_llm_for_resolution(
    provider: str,
    model_name: str,
    client: "anthropic.Anthropic | groq.Groq | genai.Client",
    row: pd.Series,
    candidates: list[dict],
    fee_hypotheses: dict[str, FeeHypothesis],
) -> dict:
    prompt = build_prompt(row, candidates, fee_hypotheses)

    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        _throttle_before_call(provider)
        try:
            if provider == "anthropic":
                response = client.messages.create(
                    model=model_name,
                    max_tokens=1024,
                    tools=[ANTHROPIC_TOOL],
                    tool_choice={"type": "tool", "name": TOOL_NAME},
                    messages=[{"role": "user", "content": prompt}],
                )
                for block in response.content:
                    if block.type == "tool_use":
                        return block.input

            elif provider == "groq":
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.2,  # keep tool-calling deterministic
                    tools=[GROQ_TOOL],
                    tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                    messages=[{"role": "user", "content": prompt}],
                )
                tool_calls = response.choices[0].message.tool_calls or []
                if tool_calls:
                    return json.loads(tool_calls[0].function.arguments)

            elif provider == "gemini":
                gemini_tool = genai_types.Tool(
                    function_declarations=[
                        genai_types.FunctionDeclaration(
                            name=TOOL_NAME,
                            description=TOOL_DESCRIPTION,
                            parameters=GEMINI_SCHEMA,
                        )
                    ]
                )
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        temperature=0.2,  # keep tool-calling deterministic
                        tools=[gemini_tool],
                        # mode="ANY" forces a function call rather than free text,
                        # and allowed_function_names pins it to our single tool.
                        tool_config=genai_types.ToolConfig(
                            function_calling_config=genai_types.FunctionCallingConfig(
                                mode="ANY",
                                allowed_function_names=[TOOL_NAME],
                            )
                        ),
                    ),
                )
                response_candidates = response.candidates or []
                if response_candidates and response_candidates[0].content.parts:
                    for part in response_candidates[0].content.parts:
                        if part.function_call:
                            return dict(part.function_call.args)

            # If we fell through without returning, the model responded but
            # didn't emit a tool call. Treat as a defensive no-op and stop
            # retrying (retrying won't fix a model that declined to call the tool).
            break

        except Exception as exc:
            error_text = str(exc)

            # Only a confirmed "this model ID doesn't exist / isn't callable"
            # error should hard-stop the whole run -- retrying can't fix that.
            # Everything else (rate limits, dropped connections, read
            # timeouts, brief 5xx blips -- e.g. the httpx.ReadError /
            # WinError 10053 "connection aborted" seen on flaky networks) is
            # treated as transient and retried with backoff.
            is_model_error = any(
                marker in error_text
                for marker in ["model_not_found", "does not exist", "NOT_FOUND", "404"]
            )

            if is_model_error:
                # Model IDs on free-tier providers get deprecated/renamed
                # periodically (or, as with some pinned Gemini models,
                # appear in the catalog but aren't actually invocable on
                # this key/tier). Fail with a clear, actionable message.
                lookup_hint = {
                    "groq": 'curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"',
                    "gemini": "https://ai.google.dev/gemini-api/docs/models",
                    "anthropic": "https://docs.claude.com/en/docs/about-claude/models/overview",
                }[provider]
                raise SystemExit(
                    f"Model '{model_name}' is not available on {provider} for this API key.\n"
                    f"Check current model IDs with: {lookup_hint}\n"
                    f"Then re-run with: --model <current-model-id>, or try --model gemini-flash-latest "
                    f"(Gemini) which auto-resolves to a currently-working model."
                ) from exc

            if attempt < MAX_RATE_LIMIT_RETRIES:
                delay = _parse_retry_delay_seconds(error_text, attempt)
                print(
                    f"Transient error calling {provider} ({type(exc).__name__}, "
                    f"attempt {attempt + 1}/{MAX_RATE_LIMIT_RETRIES}); waiting {delay:.0f}s "
                    f"before retrying payment_id={row.get('payment_id')}..."
                )
                time.sleep(delay)
                continue  # retry this row

            # Retries exhausted -- don't crash the whole batch over one
            # exception. Flag it for a human/re-run instead and move on.
            return {
                "match_status": "needs_human_review",
                "matched_bank_txn_id": None,
                "confidence": 0.0,
                "gap_diagnosis": "unexplained",
                "reasoning": (
                    f"Skipped: repeated {provider} errors ({type(exc).__name__}) persisted after "
                    f"{MAX_RATE_LIMIT_RETRIES} retries. Re-run the pipeline later to resolve this exception."
                ),
            }

    # Defensive fallback -- should not happen with tool_choice forcing the call.
    return {
        "match_status": "needs_human_review",
        "matched_bank_txn_id": None,
        "confidence": 0.0,
        "gap_diagnosis": "unexplained",
        "reasoning": "Model did not return a structured tool call.",
    }


# --------------------------------------------------------------------------- #
# Step 5: Orchestration
# --------------------------------------------------------------------------- #

def select_reference_amount(row: pd.Series) -> float | None:
    """
    Pick the correct ground-truth amount to compare a candidate bank credit
    against.

    For 'reference_matched_but_amount_mismatch' rows, Module 2 already
    identified the SPECIFIC payout this row represents -- so the right
    comparison is against that payout's own gross_amount (e.g. one
    installment of a split payment), never the invoice's full amount.
    Using the full invoice amount here was a real bug found by analyzing a
    real run: it made an ordinary ~2% gateway-fee gap on a single
    installment look like a 40-60% "severe mismatch", because the
    installment was being compared against the whole invoice instead of
    itself.

    For rows with no linked payout at all (no_matching_payout_reference /
    no_matching_invoice_reference), there is no specific installment to
    anchor to, so the invoice amount (falling back to gross_amount if
    missing) is still the correct reference.
    """
    if row.get("exception_reason") == "reference_matched_but_amount_mismatch":
        gross = row.get("gross_amount")
        if gross is not None and not pd.isna(gross):
            return gross

    reference_amount = row.get("invoice_amount")
    if pd.isna(reference_amount):
        reference_amount = row.get("gross_amount")
    return reference_amount


def resolve_exceptions(
    enriched_exceptions: pd.DataFrame,
    bank: pd.DataFrame,
    provider: str,
    model_name: str,
    client: "anthropic.Anthropic | groq.Groq | None",
    mock: bool,
) -> list[ExceptionResolution]:
    resolutions: list[ExceptionResolution] = []

    for _, row in enriched_exceptions.iterrows():
        candidates = shortlist_candidates(row, bank)

        reference_amount = select_reference_amount(row)

        # --- Pre-LLM deterministic filter (runs BEFORE mock or live LLM calls) ---
        # This is the injection point: apply_advanced_rules gets first look at
        # every exception, before any API call -- mock or live -- is even
        # considered. If it can already decide the outcome deterministically,
        # the LLM is never invoked for this row.
        filter_result = apply_advanced_rules(reference_amount, candidates)

        if not filter_result["needs_llm"]:
            resolutions.append(
                ExceptionResolution(
                    payment_id=row.get("payment_id"),
                    exception_reason=row.get("exception_reason"),
                    match_status=filter_result["match_status"],
                    matched_bank_txn_id=filter_result.get("matched_bank_txn_id"),
                    confidence=filter_result.get("confidence", 0.0),
                    amount_gap=filter_result.get("amount_gap"),
                    gap_diagnosis=filter_result.get("gap_diagnosis", "unexplained"),
                    reasoning=filter_result["reasoning"],
                    diagnostics={
                        "candidates_considered": len(candidates),
                        "filtered_by": "apply_advanced_rules",
                    },
                )
            )
            continue

        plausible_candidates = filter_result["plausible_candidates"]

        fee_hypotheses = {
            c["txn_id"]: hyp
            for c in plausible_candidates
            if (hyp := compute_fee_hypothesis(reference_amount, c["amount"])) is not None
        }

        if mock or client is None:
            # Deterministic-only mode: report the best heuristic candidate
            # without spending an LLM call. Useful for local testing/CI.
            best = plausible_candidates[0]
            hyp = fee_hypotheses.get(best["txn_id"])
            resolutions.append(
                ExceptionResolution(
                    payment_id=row.get("payment_id"),
                    exception_reason=row.get("exception_reason"),
                    match_status="needs_human_review",
                    matched_bank_txn_id=best["txn_id"],
                    confidence=0.5 if best["match_basis"] == "exact_utr_substring" else 0.3,
                    amount_gap=hyp.gap_amount if hyp else None,
                    gap_diagnosis="mdr_fee_and_tax" if (hyp and hyp.plausible_mdr_and_tax) else "unexplained",
                    reasoning="Mock mode: heuristic shortlist only, no LLM call was made.",
                    diagnostics={"candidates_considered": len(plausible_candidates), "fee_hypothesis": hyp.__dict__ if hyp else None},
                )
            )
            continue

        llm_result = call_llm_for_resolution(provider, model_name, client, row, plausible_candidates, fee_hypotheses)
        matched_hyp = fee_hypotheses.get(llm_result.get("matched_bank_txn_id"))

        resolutions.append(
            ExceptionResolution(
                payment_id=row.get("payment_id"),
                exception_reason=row.get("exception_reason"),
                match_status=llm_result.get("match_status", "needs_human_review"),
                matched_bank_txn_id=llm_result.get("matched_bank_txn_id"),
                confidence=float(llm_result.get("confidence", 0.0)),
                amount_gap=matched_hyp.gap_amount if matched_hyp else None,
                gap_diagnosis=llm_result.get("gap_diagnosis", "unexplained"),
                reasoning=llm_result.get("reasoning", ""),
                diagnostics={
                    "candidates_considered": len(plausible_candidates),
                    "fee_hypothesis": matched_hyp.__dict__ if matched_hyp else None,
                    "model_used": model_name,
                    "provider": provider,
                },
            )
        )

    return resolutions


# --------------------------------------------------------------------------- #
# Step 6: Save report
# --------------------------------------------------------------------------- #

def save_report(
    resolutions: list[ExceptionResolution], out_dir: Path, provider: str, model_name: str, mock: bool
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "resolution_report.json"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider if not mock else "none (mock mode)",
        "model_used": model_name if not mock else "none (mock mode)",
        "total_exceptions_processed": len(resolutions),
        "summary": {
            status: sum(1 for r in resolutions if r.match_status == status)
            for status in ["matched", "no_match_found", "needs_human_review"]
        },
        "resolutions": [r.__dict__ for r in resolutions],
    }

    out_path.write_text(json.dumps(report, indent=2, default=str))
    return out_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Module 3: LLM-assisted resolution of unmatched exceptions.")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--processed-dir", type=str, default=str(project_root / "data" / "processed"))
    parser.add_argument("--raw-dir", type=str, default=str(project_root / "data" / "raw"))
    parser.add_argument("--out-dir", type=str, default=str(project_root / "data" / "processed"))
    parser.add_argument(
        "--provider",
        type=str,
        choices=["gemini", "groq", "anthropic"],
        default=DEFAULT_PROVIDER,
        help=(
            "Which LLM API to use. 'gemini' (default) and 'groq' are both free, no credit "
            "card required. 'anthropic' needs a paid API key."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Override the model ID. Defaults to a current free Gemini/Groq model or a Claude "
            "model depending on --provider. Free-tier model lineups change periodically -- if "
            "the default 404s, pass a current model ID here without editing the script."
        ),
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run the full pipeline without calling any LLM (heuristics only). Useful for testing.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print the model IDs this API key actually has access to for --provider, then exit.",
    )
    args = parser.parse_args()

    if args.list_models:
        client = get_client(args.provider)
        models = list_available_models(args.provider, client)
        if models:
            print(f"Models available to this {args.provider} API key:")
            for name in models:
                print(f"  {name}")
        else:
            print(f"Could not retrieve a model list for provider '{args.provider}'.")
        return

    processed_dir = Path(args.processed_dir)
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.out_dir)

    default_models = {
        "gemini": GEMINI_MODEL_NAME,
        "groq": GROQ_MODEL_NAME,
        "anthropic": ANTHROPIC_MODEL_NAME,
    }
    model_name = args.model or default_models[args.provider]

    exceptions, invoices, payouts, bank = load_inputs(processed_dir, raw_dir)
    enriched = enrich_exceptions(exceptions, invoices, payouts)

    client = None if args.mock else get_client(args.provider)

    # Free-tier model access on Gemini/Groq varies by account, region, and
    # how recently the key was created -- verify (and auto-correct) the
    # model choice once up front rather than letting every single exception
    # fail on the same 404.
    if client is not None:
        model_name = resolve_working_model(args.provider, client, model_name)

    resolutions = resolve_exceptions(enriched, bank, args.provider, model_name, client, mock=args.mock)
    out_path = save_report(resolutions, out_dir, args.provider, model_name, mock=args.mock)

    matched = sum(1 for r in resolutions if r.match_status == "matched")
    label = "mock mode" if args.mock else f"{args.provider} / {model_name}"
    print(f"Processed {len(resolutions)} exceptions ({label}).")
    print(f"  matched:            {matched}")
    print(f"  no_match_found:     {sum(1 for r in resolutions if r.match_status == 'no_match_found')}")
    print(f"  needs_human_review: {sum(1 for r in resolutions if r.match_status == 'needs_human_review')}")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()