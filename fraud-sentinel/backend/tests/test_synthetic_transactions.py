"""Tests for deterministic synthetic transaction generation."""

import unittest
from collections import Counter

from app.data.synthetic_transactions import generate_dataset, validate_dataset


class SyntheticTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset, self.report = generate_dataset(seed=17, days=14, baseline_count=120)

    def test_generation_is_reproducible_for_a_fixed_seed(self) -> None:
        repeated_dataset, repeated_report = generate_dataset(
            seed=17, days=14, baseline_count=120
        )

        self.assertEqual(self.dataset.transactions, repeated_dataset.transactions)
        self.assertEqual(self.report, repeated_report)

    def test_baseline_is_safe_and_scenarios_are_labeled(self) -> None:
        baseline = [row for row in self.dataset.transactions if row.label == "safe"]
        self.assertTrue(all(row.label == "safe" for row in baseline))
        self.assertTrue(all(row.fraud_pattern is None for row in baseline))
        self.assertEqual(self.report.transaction_count, 129)
        self.assertEqual(self.report.safe_count, 120)
        self.assertEqual(self.report.fraud_count, 9)
        self.assertEqual(
            self.report.fraud_pattern_counts,
            {
                "large_collect_first_time_payee": 1,
                "rapid_mule_chain": 4,
                "circular_money_flow": 4,
            },
        )

    def test_payee_flags_follow_chronological_order(self) -> None:
        seen_pairs: set[tuple[str, str]] = set()
        for row in self.dataset.transactions:
            pair = (row.sender_account, row.receiver_account)
            self.assertEqual(row.first_time_payee, pair not in seen_pairs)
            seen_pairs.add(pair)

    def test_required_injected_cycle_is_checked(self) -> None:
        without_cycle = tuple(
            row
            for row in self.dataset.transactions
            if row.fraud_pattern != "circular_money_flow"
        )

        with self.assertRaisesRegex(ValueError, "circular money-flow pattern is missing"):
            validate_dataset(
                without_cycle,
                account_creation_times=self.dataset.account_creation_times,
                device_ids=self.dataset.device_ids,
                start=self.dataset.start,
                end=self.dataset.end,
                minimum_count=0,
            )

    def test_report_counts_match_the_records(self) -> None:
        labels = Counter(row.label for row in self.dataset.transactions)
        self.assertEqual(labels["safe"], self.report.safe_count)
        self.assertEqual(labels["fraud"], self.report.fraud_count)

    def test_duplicate_transaction_ids_are_rejected(self) -> None:
        rows = list(self.dataset.transactions)
        rows[1] = rows[1].model_copy(
            update={"transaction_id": rows[0].transaction_id}
        )

        with self.assertRaisesRegex(ValueError, "transaction IDs are not unique"):
            validate_dataset(
                rows,
                account_creation_times=self.dataset.account_creation_times,
                device_ids=self.dataset.device_ids,
                start=self.dataset.start,
                end=self.dataset.end,
                minimum_count=0,
            )

    def test_unknown_device_references_are_rejected(self) -> None:
        rows = list(self.dataset.transactions)
        rows[0] = rows[0].model_copy(update={"device_id": "DEV-UNKNOWN"})

        with self.assertRaisesRegex(ValueError, "references an unknown device"):
            validate_dataset(
                rows,
                account_creation_times=self.dataset.account_creation_times,
                device_ids=self.dataset.device_ids,
                start=self.dataset.start,
                end=self.dataset.end,
                minimum_count=0,
            )


if __name__ == "__main__":
    unittest.main()