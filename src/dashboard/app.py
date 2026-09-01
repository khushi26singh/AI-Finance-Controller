"""
app.py

Module 4 of the AI Finance Controller.

Purpose
-------
A Streamlit dashboard that ties Modules 1-3 together into something a
finance user (not just a developer) can actually operate:

    1. Get data in -- either generate a fresh synthetic dataset (Module 1)
       or upload your own internal_invoices / gateway_payouts /
       bank_statements CSVs in the same schema.
    2. Run the deterministic exact-match pass (Module 2).
    3. Run the LLM-assisted resolution pass on whatever's left (Module 3),
       either in free/mock heuristic mode or with a live LLM call.
    4. Show the results honestly: a metrics banner, a table of everything
       that was actually resolved (with the audit trail behind each
       resolution visible, not hidden), and a separate, clearly-labelled
       list of genuine exceptions that still need a human.

Design principle: don't oversell the AI
------------------------------------------
An LLM saying "matched" is not the same as a human confirming a match. To
keep this dashboard honest rather than optimistic, an LLM resolution only
counts as "AI-Resolved" in the summary metrics if its self-reported
confidence clears CONFIDENCE_THRESHOLD. Anything below that threshold --
even if the model technically said "matched" -- is shown in the human
review list, not the resolved list. The AI's own reasoning and confidence
score are always visible in both tables, so nothing is hidden from the
reviewer.

Run with:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Make Modules 1-3 importable as sibling packages under src/
# --------------------------------------------------------------------------- #

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_generation.generate_synthetic_data import SyntheticFinanceDataGenerator  # noqa: E402
from reconciliation.exact_match_reconciler import exact_match_reconcile  # noqa: E402
from llm_resolution.llm_resolver import (  # noqa: E402
    ANTHROPIC_MODEL_NAME,
    GEMINI_MODEL_NAME,
    GROQ_MODEL_NAME,
    enrich_exceptions,
    get_client,
    resolve_exceptions,
    resolve_working_model,
)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="AI Finance Controller", page_icon="💰", layout="wide")

# An LLM resolution below this confidence is treated as "needs human review"
# in the summary metrics and tables, regardless of what match_status it
# reported. This is what keeps the "AI-Resolved" number honest.
CONFIDENCE_THRESHOLD = 0.75

DEFAULT_MODEL_BY_PROVIDER = {
    "gemini": GEMINI_MODEL_NAME,
    "groq": GROQ_MODEL_NAME,
    "anthropic": ANTHROPIC_MODEL_NAME,
}

REQUIRED_COLUMNS = {
    "invoices": {"invoice_id", "customer_name", "po_number", "invoice_amount", "invoice_date", "due_date", "status"},
    "payouts": {"payout_id", "gateway", "utr_number", "linked_invoice_id", "gross_amount", "fee_amount", "net_amount", "payout_date", "status"},
    "bank": {"txn_id", "txn_date", "description", "txn_type", "amount", "balance"},
}


# --------------------------------------------------------------------------- #
# Pipeline orchestration (pure functions, no Streamlit calls -- kept
# separate from the UI so the logic is easy to reason about and reuse)
# --------------------------------------------------------------------------- #

def generate_synthetic_tables(seed: int, num_invoices: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run Module 1's generator in-memory (no CSV round-trip needed)."""
    generator = SyntheticFinanceDataGenerator(seed=seed)
    generator.generate_invoices(count=num_invoices)
    generator.generate_payouts()
    generator.generate_bank_statements()
    return generator.to_dataframes()


def validate_uploaded_tables(invoices: pd.DataFrame, payouts: pd.DataFrame, bank: pd.DataFrame) -> list[str]:
    """Return a list of human-readable schema problems, empty if all good."""
    problems = []
    for name, df, required in [
        ("internal_invoices.csv", invoices, REQUIRED_COLUMNS["invoices"]),
        ("gateway_payouts.csv", payouts, REQUIRED_COLUMNS["payouts"]),
        ("bank_statements.csv", bank, REQUIRED_COLUMNS["bank"]),
    ]:
        missing = required - set(df.columns)
        if missing:
            problems.append(f"{name} is missing expected column(s): {', '.join(sorted(missing))}")
    return problems


def run_pipeline(
    invoices: pd.DataFrame,
    payouts: pd.DataFrame,
    bank: pd.DataFrame,
    use_live_ai: bool,
    provider: str,
    model_name: str,
) -> dict:
    """
    Run Module 2 then Module 3 end-to-end against already-loaded tables and
    return every artifact the dashboard needs to render: the exact matches,
    the raw exceptions, and the resolutions produced for those exceptions.
    """
    exact_matches, unmatched_exceptions = exact_match_reconcile(invoices, payouts)

    exceptions_df = pd.DataFrame(unmatched_exceptions)
    resolutions: list[dict] = []

    if not exceptions_df.empty:
        enriched = enrich_exceptions(exceptions_df, invoices, payouts)
        client = get_client(provider) if use_live_ai else None
        # Free-tier model access on Gemini/Groq varies by account/region and
        # changes over time -- verify (and auto-correct) the model choice
        # once up front rather than letting every exception fail the same way.
        if client is not None:
            model_name = resolve_working_model(provider, client, model_name)
        resolution_objects = resolve_exceptions(
            enriched, bank, provider, model_name, client, mock=not use_live_ai
        )
        resolutions = [asdict(r) for r in resolution_objects]

    return {
        "invoices": invoices,
        "payouts": payouts,
        "bank": bank,
        "exact_matches": exact_matches,
        "unmatched_exceptions": unmatched_exceptions,
        "resolutions": resolutions,
        "used_live_ai": use_live_ai,
        "provider": provider if use_live_ai else "mock (heuristics only)",
        "model_name": model_name if use_live_ai else "n/a",
    }


def is_ai_confirmed(resolution: dict) -> bool:
    """The honesty gate: only a high-confidence 'matched' verdict counts as
    genuinely AI-resolved. Everything else routes to human review."""
    return resolution.get("match_status") == "matched" and resolution.get("confidence", 0.0) >= CONFIDENCE_THRESHOLD


def compute_summary_metrics(results: dict) -> dict:
    total_records = len(results["invoices"])
    exact_count = len(results["exact_matches"])
    ai_confirmed = [r for r in results["resolutions"] if is_ai_confirmed(r)]
    needs_review = [r for r in results["resolutions"] if not is_ai_confirmed(r)]

    exact_rate = (exact_count / total_records * 100) if total_records else 0.0
    ai_rate = (len(ai_confirmed) / total_records * 100) if total_records else 0.0

    return {
        "total_records": total_records,
        "exact_count": exact_count,
        "exact_rate": exact_rate,
        "ai_confirmed_count": len(ai_confirmed),
        "ai_rate": ai_rate,
        "needs_review_count": len(needs_review),
    }


def build_resolved_table(results: dict) -> pd.DataFrame:
    """One row per resolved transaction, whichever stage resolved it, with
    the audit trail (why it was accepted) visible in every row."""
    rows = []

    for m in results["exact_matches"]:
        rows.append(
            {
                "payment_id": m["payment_id"],
                "resolved_by": "Exact match (Module 2)",
                "confidence": 1.00,
                "invoice_amount": m["invoice_amount"],
                "matched_amount": m["gross_amount"],
                "amount_gap": round(m["invoice_amount"] - m["gross_amount"], 2),
                "audit_reasoning": "Deterministic: payment_id reference and amount matched exactly, within rounding tolerance.",
                "matched_bank_txn_id": None,
            }
        )

    for r in results["resolutions"]:
        if is_ai_confirmed(r):
            diag = r.get("diagnostics") or {}
            rows.append(
                {
                    "payment_id": r["payment_id"],
                    "resolved_by": f"AI semantic match ({diag.get('provider', 'mock')})",
                    "confidence": r["confidence"],
                    "invoice_amount": None,
                    "matched_amount": None,
                    "amount_gap": r.get("amount_gap"),
                    "audit_reasoning": r.get("reasoning", ""),
                    "matched_bank_txn_id": r.get("matched_bank_txn_id"),
                }
            )

    return pd.DataFrame(rows)


def build_review_table(results: dict) -> pd.DataFrame:
    """Every genuine exception: things Module 2 couldn't join AND things
    the AI either couldn't resolve or wasn't confident enough about."""
    rows = []
    for r in results["resolutions"]:
        if not is_ai_confirmed(r):
            diag = r.get("diagnostics") or {}
            rows.append(
                {
                    "payment_id": r["payment_id"],
                    "original_exception_reason": r.get("exception_reason"),
                    "ai_match_status": r.get("match_status"),
                    "ai_confidence": r.get("confidence"),
                    "ai_candidate_bank_txn_id": r.get("matched_bank_txn_id"),
                    "ai_gap_diagnosis": r.get("gap_diagnosis"),
                    "ai_reasoning": r.get("reasoning", ""),
                    "candidates_considered": diag.get("candidates_considered", 0),
                    "why_flagged": (
                        "Below confidence threshold" if r.get("match_status") == "matched"
                        else "AI could not confirm a match"
                    ),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

def render_sidebar() -> dict:
    st.sidebar.header("1. Data source")
    data_source = st.sidebar.radio(
        "Choose how to load transactions",
        ["Generate synthetic dataset", "Upload my own CSVs"],
    )

    generated_tables = None
    uploaded_tables = None

    if data_source == "Generate synthetic dataset":
        seed = st.sidebar.number_input("Random seed", value=42, step=1)
        num_invoices = st.sidebar.slider("Number of invoices", min_value=20, max_value=200, value=60)
        if st.sidebar.button("🎲 Generate synthetic data"):
            with st.spinner("Generating synthetic invoices, payouts, and bank statements..."):
                generated_tables = generate_synthetic_tables(seed, num_invoices)
            st.session_state["source_tables"] = generated_tables
            st.session_state.pop("results", None)  # invalidate any stale pipeline run

    else:
        st.sidebar.caption("Each file must match Module 1's schema.")
        invoices_file = st.sidebar.file_uploader("internal_invoices.csv", type="csv", key="up_inv")
        payouts_file = st.sidebar.file_uploader("gateway_payouts.csv", type="csv", key="up_pay")
        bank_file = st.sidebar.file_uploader("bank_statements.csv", type="csv", key="up_bank")

        if st.sidebar.button("📤 Load uploaded data"):
            if not (invoices_file and payouts_file and bank_file):
                st.sidebar.error("Please upload all three CSV files before loading.")
            else:
                invoices_df = pd.read_csv(invoices_file)
                payouts_df = pd.read_csv(payouts_file)
                bank_df = pd.read_csv(bank_file)
                problems = validate_uploaded_tables(invoices_df, payouts_df, bank_df)
                if problems:
                    for p in problems:
                        st.sidebar.error(p)
                else:
                    uploaded_tables = (invoices_df, payouts_df, bank_df)
                    st.session_state["source_tables"] = uploaded_tables
                    st.session_state.pop("results", None)

    st.sidebar.header("2. AI resolution (Module 3)")
    use_live_ai = st.sidebar.checkbox("Use live AI matching (calls an LLM API)", value=False)
    provider = "gemini"
    model_name = DEFAULT_MODEL_BY_PROVIDER["gemini"]
    if use_live_ai:
        provider = st.sidebar.selectbox("Provider", ["gemini", "groq", "anthropic"])
        model_name = st.sidebar.text_input("Model ID", value=DEFAULT_MODEL_BY_PROVIDER[provider])
        st.sidebar.caption(
            "Requires the matching API key (GEMINI_API_KEY / GROQ_API_KEY / ANTHROPIC_API_KEY) "
            "set as an environment variable in the environment Streamlit is running in."
        )
    else:
        st.sidebar.caption(
            "Mock mode: runs the deterministic shortlisting + fee-hypothesis heuristics "
            "with no API calls and no cost."
        )

    run_clicked = st.sidebar.button("🚀 Run reconciliation pipeline", type="primary")

    return {
        "run_clicked": run_clicked,
        "use_live_ai": use_live_ai,
        "provider": provider,
        "model_name": model_name,
    }


def render_metrics_banner(metrics: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Total Records", f"{metrics['total_records']:,}")
    cols[1].metric(
        "Exact Match Rate",
        f"{metrics['exact_rate']:.1f}%",
        help=f"{metrics['exact_count']} of {metrics['total_records']} resolved by deterministic reference + amount match (Module 2).",
    )
    cols[2].metric(
        "AI-Resolved Rate",
        f"{metrics['ai_rate']:.1f}%",
        help=(
            f"{metrics['ai_confirmed_count']} of {metrics['total_records']} resolved by the LLM "
            f"with confidence ≥ {CONFIDENCE_THRESHOLD:.0%}. Lower-confidence AI matches are NOT "
            f"counted here -- they appear in the human review list instead."
        ),
    )
    cols[3].metric(
        "Unresolved Exceptions",
        f"{metrics['needs_review_count']}",
        help="Genuine exceptions: no confident match was found by either the rules engine or the AI.",
    )


def render_resolved_tab(results: dict) -> None:
    resolved_df = build_resolved_table(results)
    if resolved_df.empty:
        st.info("No transactions have been resolved yet.")
        return

    st.caption(
        "Every row here was resolved either by an exact deterministic rule or by an AI match "
        f"that cleared the {CONFIDENCE_THRESHOLD:.0%} confidence bar. The `resolved_by` and "
        "`audit_reasoning` columns show exactly why each row was accepted."
    )
    st.dataframe(
        resolved_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            "audit_reasoning": st.column_config.TextColumn("Audit reasoning", width="large"),
        },
    )
    st.download_button(
        "⬇ Download resolved transactions (CSV)",
        resolved_df.to_csv(index=False).encode("utf-8"),
        file_name="resolved_transactions.csv",
        mime="text/csv",
    )


def render_review_tab(results: dict) -> None:
    review_df = build_review_table(results)
    if review_df.empty:
        st.success("No exceptions need human review right now.")
        return

    st.warning(
        f"{len(review_df)} transaction(s) could not be confidently resolved by either the "
        "rules engine or the AI, and need a human to look at them. Nothing here has been "
        "silently auto-closed."
    )
    st.dataframe(
        review_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ai_confidence": st.column_config.ProgressColumn("AI confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            "ai_reasoning": st.column_config.TextColumn("AI reasoning", width="large"),
        },
    )
    st.download_button(
        "⬇ Download exceptions for review (CSV)",
        review_df.to_csv(index=False).encode("utf-8"),
        file_name="needs_human_review.csv",
        mime="text/csv",
    )


def render_raw_data_tab(results: dict) -> None:
    st.caption("The underlying tables used for this run, for reference.")
    sub_tabs = st.tabs(["Internal invoices", "Gateway payouts", "Bank statements"])
    with sub_tabs[0]:
        st.dataframe(results["invoices"], use_container_width=True, hide_index=True)
    with sub_tabs[1]:
        st.dataframe(results["payouts"], use_container_width=True, hide_index=True)
    with sub_tabs[2]:
        st.dataframe(results["bank"], use_container_width=True, hide_index=True)


def main() -> None:
    st.title("💰 AI Finance Controller")
    st.caption("Reconciliation dashboard for Modules 1-3: synthetic data, exact matching, and AI-assisted resolution.")

    controls = render_sidebar()

    if controls["run_clicked"]:
        if "source_tables" not in st.session_state:
            st.error("Load a data source first (generate synthetic data or upload your CSVs) before running the pipeline.")
        else:
            invoices, payouts, bank = st.session_state["source_tables"]
            spinner_msg = (
                "Running exact matching, then calling the live AI on the remaining exceptions..."
                if controls["use_live_ai"]
                else "Running exact matching, then heuristic-only resolution (mock mode, no API calls)..."
            )
            with st.spinner(spinner_msg):
                st.session_state["results"] = run_pipeline(
                    invoices, payouts, bank,
                    use_live_ai=controls["use_live_ai"],
                    provider=controls["provider"],
                    model_name=controls["model_name"],
                )

    if "results" not in st.session_state:
        st.info("👈 Choose a data source and click **Run reconciliation pipeline** to get started.")
        return

    results = st.session_state["results"]
    metrics = compute_summary_metrics(results)

    render_metrics_banner(metrics)
    st.caption(
        f"Resolution mode: **{results['provider']}**"
        + (f" / `{results['model_name']}`" if results["used_live_ai"] else "")
    )

    resolved_tab, review_tab, raw_tab = st.tabs(
        ["✅ Resolved Transactions", "⚠️ Needs Human Review", "📄 Raw Data"]
    )
    with resolved_tab:
        render_resolved_tab(results)
    with review_tab:
        render_review_tab(results)
    with raw_tab:
        render_raw_data_tab(results)


if __name__ == "__main__":
    main()