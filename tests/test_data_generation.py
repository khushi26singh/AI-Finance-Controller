"""
tests/test_data_generation.py

Covers Module 1's synthetic data generator (src/data_generation/generate_synthetic_data.py).
Directly addresses the judge checklist's reproducibility requirement: a judge
re-running the demo with the same seed should get the same quoted numbers.
"""

import pandas as pd

from data_generation.generate_synthetic_data import SyntheticFinanceDataGenerator


def _generate(seed: int, num_invoices: int = 60):
    gen = SyntheticFinanceDataGenerator(seed=seed)
    gen.generate_invoices(count=num_invoices)
    gen.generate_payouts()
    gen.generate_bank_statements()
    return gen.to_dataframes()


class TestReproducibility:
    def test_same_seed_produces_identical_invoices(self):
        invoices_a, _, _ = _generate(seed=42)
        invoices_b, _, _ = _generate(seed=42)
        pd.testing.assert_frame_equal(invoices_a, invoices_b)

    def test_same_seed_produces_identical_payouts(self):
        _, payouts_a, _ = _generate(seed=42)
        _, payouts_b, _ = _generate(seed=42)
        pd.testing.assert_frame_equal(payouts_a, payouts_b)

    def test_same_seed_produces_identical_bank_statements(self):
        _, _, bank_a = _generate(seed=42)
        _, _, bank_b = _generate(seed=42)
        pd.testing.assert_frame_equal(bank_a, bank_b)

    def test_different_seeds_produce_different_data(self):
        """Sanity check the other direction too: seeding actually does
        something, rather than the generator secretly ignoring it."""
        invoices_a, _, _ = _generate(seed=1)
        invoices_b, _, _ = _generate(seed=2)
        assert not invoices_a.equals(invoices_b)


class TestGeneratedDataIntegrity:
    """Structural guarantees a judge (or Module 2/3) can rely on."""

    def test_row_counts_match_requested_invoice_count(self):
        invoices, _, _ = _generate(seed=42, num_invoices=60)
        assert len(invoices) == 60

    def test_invoice_ids_are_unique(self):
        invoices, _, _ = _generate(seed=42)
        assert invoices["invoice_id"].is_unique

    def test_every_payout_references_a_real_invoice_id_prefix(self):
        """Even garbled references should still be a prefix of a real
        invoice_id, an empty string, or 'N/A' -- never something
        unrelated to any invoice (see Module 1's garble_link feature)."""
        invoices, payouts, _ = _generate(seed=42)
        real_ids = set(invoices["invoice_id"])
        real_id_prefixes = {inv_id[:6] for inv_id in real_ids}
        for ref in payouts["linked_invoice_id"]:
            assert ref in real_ids or ref in real_id_prefixes or ref in ("", "N/A")

    def test_no_payout_amount_exceeds_its_invoice_when_linked(self):
        """Gateway fees are deducted, never added -- a linked payout's
        gross_amount should never exceed the invoice it's paying."""
        invoices, payouts, _ = _generate(seed=42)
        merged = payouts.merge(
            invoices[["invoice_id", "invoice_amount"]],
            left_on="linked_invoice_id",
            right_on="invoice_id",
            how="inner",
        )
        assert (merged["gross_amount"] <= merged["invoice_amount"] + 0.01).all()