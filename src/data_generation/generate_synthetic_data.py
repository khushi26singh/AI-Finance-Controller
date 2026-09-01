"""
generate_synthetic_data.py

Generates a synthetic dataset for the AI Finance Controller project, simulating
three real-world data sources that a reconciliation engine would need to
cross-reference:

    1. internal_invoices.csv   -> Clean, structured records from an internal
                                   billing/ERP system.
    2. gateway_payouts.csv     -> Semi-structured payout logs from payment
                                   gateways (Razorpay, PayU, Cashfree, Stripe).
    3. bank_statements.csv     -> Messy, unstructured bank narration strings,
                                   the hardest source to parse (UPI/NEFT/IMPS/
                                   RTGS references, typos, inconsistent casing).

The generator deliberately injects realistic reconciliation challenges:
    - Gateway fees/TDS deductions so payout amount != invoice amount.
    - Invoices with no payout yet (unpaid / overdue).
    - Payouts with no matching invoice reference in the bank narration.
    - Duplicate/split payouts for a single invoice.
    - Refunds and bank charges that don't correspond to any invoice.
    - Free-text noise (extra whitespace, mixed case, truncated references).

Output: three CSV files written to the `data/raw/` directory, each containing
at least 50 records.

Usage:
    python generate_synthetic_data.py [--seed 42] [--out-dir ../../data/raw]
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from faker import Faker

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

NUM_INVOICES = 60
GATEWAYS = ["Razorpay", "PayU", "Cashfree", "Stripe"]
BANKS = ["HDFC0000123", "ICIC0001234", "SBIN0005678", "YESB0000001", "KKBK0000456"]
INVOICE_STATUSES_UNPAID = ["Sent", "Overdue"]

START_DATE = date(2025, 1, 1)


def random_date(start: date, day_span: int) -> date:
    return start + timedelta(days=random.randint(0, day_span))


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #

@dataclass
class Invoice:
    invoice_id: str
    customer_name: str
    po_number: str
    invoice_amount: float
    invoice_date: str
    due_date: str
    status: str


@dataclass
class GatewayPayout:
    payout_id: str
    gateway: str
    utr_number: str
    linked_invoice_id: str
    gross_amount: float
    fee_amount: float
    net_amount: float
    payout_date: str
    status: str


@dataclass
class BankStatementLine:
    txn_id: str
    txn_date: str
    description: str
    txn_type: str  # DR (debit) or CR (credit)
    amount: float
    balance: float


# --------------------------------------------------------------------------- #
# Generators
# --------------------------------------------------------------------------- #

class SyntheticFinanceDataGenerator:
    """Builds internally-consistent (but messy) finance datasets."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = random.Random(seed)
        random.seed(seed)
        self.faker = Faker()
        Faker.seed(seed)

        self.invoices: list[Invoice] = []
        self.payouts: list[GatewayPayout] = []
        self.bank_lines: list[BankStatementLine] = []
        self._running_balance = 500_000.00

    # -- Step 1: Internal invoices ---------------------------------------- #
    def generate_invoices(self, count: int = NUM_INVOICES) -> None:
        for i in range(1, count + 1):
            invoice_id = f"INV-{2025000 + i}"
            invoice_date = random_date(START_DATE, 240)
            due_date = invoice_date + timedelta(days=self.rng.choice([15, 30, 45]))
            amount = round(self.rng.uniform(1_500, 250_000), 2)

            self.invoices.append(
                Invoice(
                    invoice_id=invoice_id,
                    customer_name=self.faker.company(),
                    po_number=f"PO-{self.rng.randint(10000, 99999)}",
                    invoice_amount=amount,
                    invoice_date=invoice_date.isoformat(),
                    due_date=due_date.isoformat(),
                    status="Sent",  # finalized below once payout outcome is known
                )
            )

    # -- Step 2: Gateway payouts ------------------------------------------- #
    def generate_payouts(self) -> None:
        """
        For each invoice, probabilistically decide a payment outcome:
          - 70% paid in full via a single gateway payout
          - 10% paid via two split payouts (partial payments)
          - 5% paid but the gateway's linked_invoice_id field is blank/garbled
            (simulating a merchant reference that failed to propagate)
          - 15% unpaid (no payout at all) -> invoice stays Sent/Overdue
        """
        payout_counter = 1
        today = START_DATE + timedelta(days=260)

        for invoice in self.invoices:
            outcome = self.rng.random()
            invoice_date = date.fromisoformat(invoice.invoice_date)

            if outcome < 0.70:
                self._create_payout(invoice, invoice.invoice_amount, payout_counter)
                payout_counter += 1
                invoice.status = "Paid"

            elif outcome < 0.80:
                # Split into two partial payouts
                first_share = round(invoice.invoice_amount * self.rng.uniform(0.3, 0.6), 2)
                second_share = round(invoice.invoice_amount - first_share, 2)
                self._create_payout(invoice, first_share, payout_counter)
                payout_counter += 1
                self._create_payout(invoice, second_share, payout_counter)
                payout_counter += 1
                invoice.status = "Paid"

            elif outcome < 0.85:
                # Paid, but reference linkage is broken/garbled downstream
                self._create_payout(
                    invoice, invoice.invoice_amount, payout_counter, garble_link=True
                )
                payout_counter += 1
                invoice.status = "Paid"

            else:
                invoice.status = (
                    "Overdue" if invoice_date + timedelta(days=45) < today else "Sent"
                )

    def _create_payout(
        self, invoice: Invoice, gross_amount: float, counter: int, garble_link: bool = False
    ) -> None:
        gateway = self.rng.choice(GATEWAYS)
        fee_pct = self.rng.uniform(0.015, 0.029)  # typical gateway fee 1.5%-2.9%
        fee_amount = round(gross_amount * fee_pct, 2)
        net_amount = round(gross_amount - fee_amount, 2)
        payout_date = date.fromisoformat(invoice.invoice_date) + timedelta(
            days=self.rng.randint(1, 10)
        )

        linked_id = invoice.invoice_id
        if garble_link:
            # Simulate a mangled/truncated reference passed by the gateway
            linked_id = self.rng.choice(["", invoice.invoice_id[:6], "N/A"])

        self.payouts.append(
            GatewayPayout(
                payout_id=f"PAYOUT-{gateway[:3].upper()}-{counter:05d}",
                gateway=gateway,
                utr_number=f"UTR{self.rng.randint(10**11, 10**12 - 1)}",
                linked_invoice_id=linked_id,
                gross_amount=gross_amount,
                fee_amount=fee_amount,
                net_amount=net_amount,
                payout_date=payout_date.isoformat(),
                status="Settled",
            )
        )

    # -- Step 3: Bank statement (messy, unstructured) ---------------------- #
    def generate_bank_statements(self) -> None:
        txn_counter = 1

        # 3a. One bank credit line per settled payout, with messy narration.
        for payout in self.payouts:
            txn_counter = self._add_bank_line(
                txn_counter,
                txn_date=payout.payout_date,
                amount=payout.net_amount,
                txn_type="CR",
                description=self._build_payout_narration(payout),
            )

        # 3b. A handful of noise transactions unrelated to any invoice
        #     (rent, salaries, bank charges, refunds, misc UPI transfers).
        noise_count = max(0, 65 - len(self.payouts))
        for _ in range(noise_count):
            noise_date = random_date(START_DATE, 260)
            txn_type = self.rng.choice(["DR", "DR", "CR"])
            amount = round(self.rng.uniform(200, 80_000), 2)
            txn_counter = self._add_bank_line(
                txn_counter,
                txn_date=noise_date.isoformat(),
                amount=amount,
                txn_type=txn_type,
                description=self._build_noise_narration(),
            )

        # Sort the ledger chronologically and recompute a running balance
        self.bank_lines.sort(key=lambda line: line.txn_date)
        balance = 500_000.00
        for line in self.bank_lines:
            balance += line.amount if line.txn_type == "CR" else -line.amount
            line.balance = round(balance, 2)

    def _add_bank_line(
        self, counter: int, txn_date: str, amount: float, txn_type: str, description: str
    ) -> int:
        self.bank_lines.append(
            BankStatementLine(
                txn_id=f"TXN{counter:06d}",
                txn_date=txn_date,
                description=description,
                txn_type=txn_type,
                amount=amount,
                balance=0.0,  # recomputed later in chronological order
            )
        )
        return counter + 1

    def _build_payout_narration(self, payout: GatewayPayout) -> str:
        """Builds a messy, bank-style narration string for a gateway settlement."""
        bank_code = self.rng.choice(BANKS)
        rail = self.rng.choice(["UPI", "NEFT", "IMPS", "RTGS"])
        ref_tail = self.rng.randint(10000, 99999)

        templates = [
            f"UPI/CR/{ref_tail}/{payout.gateway}/{payout.utr_number}",
            f"NEFT-{bank_code}-{payout.gateway.upper()} SETTLEMENT-{payout.utr_number}",
            f"IMPS/P2A/{ref_tail}/{payout.gateway} Payouts Pvt Ltd/{payout.utr_number}",
            f"{rail}/{payout.utr_number}/{payout.gateway.lower()}settlement/{ref_tail}",
            f"NEFT/{bank_code}/{payout.gateway}/{payout.utr_number}  txn charges incl",
            f"  {rail}-{payout.utr_number}-{payout.gateway}   ",  # stray whitespace
        ]
        narration = self.rng.choice(templates)

        # Randomly mangle casing to mimic inconsistent bank exports
        if self.rng.random() < 0.3:
            narration = narration.upper()
        elif self.rng.random() < 0.3:
            narration = narration.lower()

        return narration

    def _build_noise_narration(self) -> str:
        options = [
            f"UPI/DR/{self.rng.randint(10000,99999)}/{self.faker.first_name()}{self.faker.last_name()}/paytm",
            f"NEFT-{self.rng.choice(BANKS)}-OFFICE RENT-AUG",
            f"SALARY-{self.faker.last_name().upper()}-{random_date(START_DATE, 260).strftime('%b%Y').upper()}",
            "BANK CHARGES-QUARTERLY AMC FEE",
            f"IMPS/P2P/{self.rng.randint(10000,99999)}/{self.faker.first_name()}/refund",
            f"RTGS/{self.rng.choice(BANKS)}/VENDOR PAYMENT/{self.faker.company()[:15].upper()}",
            "GST-AUTODEBIT-CHALLAN",
            f"  upi/dr/{self.rng.randint(10000,99999)}/razorpay/settlement  ",
        ]
        return self.rng.choice(options)

    # -- Export -------------------------------------------------------------#
    def to_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        invoices_df = pd.DataFrame([asdict(i) for i in self.invoices])
        payouts_df = pd.DataFrame([asdict(p) for p in self.payouts])
        bank_df = pd.DataFrame([asdict(b) for b in self.bank_lines])
        return invoices_df, payouts_df, bank_df


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic finance datasets.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "raw"),
        help="Directory where CSV files will be written.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = SyntheticFinanceDataGenerator(seed=args.seed)
    generator.generate_invoices()
    generator.generate_payouts()
    generator.generate_bank_statements()

    invoices_df, payouts_df, bank_df = generator.to_dataframes()

    invoices_path = out_dir / "internal_invoices.csv"
    payouts_path = out_dir / "gateway_payouts.csv"
    bank_path = out_dir / "bank_statements.csv"

    invoices_df.to_csv(invoices_path, index=False)
    payouts_df.to_csv(payouts_path, index=False)
    bank_df.to_csv(bank_path, index=False)

    print(f"Generated {len(invoices_df)} internal invoices     -> {invoices_path}")
    print(f"Generated {len(payouts_df)} gateway payout records -> {payouts_path}")
    print(f"Generated {len(bank_df)} bank statement lines      -> {bank_path}")


if __name__ == "__main__":
    main()