"""Unit and integration-style tests for observable deterministic rules."""

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.data.synthetic_transactions import generate_dataset
from app.models.rule_evidence import TransactionObservation
from app.models.transaction import TransactionType
from app.services.rule_engine import (
    CIRCULAR_FLOW_PRECURSOR,
    LARGE_COLLECT_FIRST_TIME_PAYEE,
    RAPID_TRANSFER_BEHAVIOUR,
    RuleEngine,
    RuleThresholds,
    load_observations_csv,
)

START = datetime(2025, 2, 1, tzinfo=timezone.utc)
CREATED = datetime(2023, 1, 1, tzinfo=timezone.utc)
DEFAULT_THRESHOLDS = RuleThresholds(
    large_collect_amount_threshold=25_000,
    rapid_transfer_window_seconds=300,
    rapid_transfer_minimum_count=3,
    circular_flow_window_seconds=300,
    circular_flow_max_transfers=4,
)


def observation(
    transaction_id: str,
    *,
    offset_seconds: int = 0,
    sender: str = "A",
    receiver: str = "B",
    amount: float = 1_000,
    transaction_type: TransactionType = "transfer",
    first_time_payee: bool = False,
) -> TransactionObservation:
    return TransactionObservation(
        transaction_id=transaction_id,
        timestamp=START + timedelta(seconds=offset_seconds),
        sender_account=sender,
        receiver_account=receiver,
        amount=amount,
        transaction_type=transaction_type,
        device_id=f"DEV-{sender}",
        sender_account_created_at=CREATED,
        receiver_account_created_at=CREATED,
        first_time_payee=first_time_payee,
    )


class RuleEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = RuleEngine(DEFAULT_THRESHOLDS)

    def _signals(self, report, transaction_id: str):
        return report.transaction_evidence[
            next(
                index
                for index, row in enumerate(report.transaction_evidence)
                if row.transaction_id == transaction_id
            )
        ].rule_signals

    def test_large_first_time_collect_triggers_with_structured_evidence(self) -> None:
        row = observation(
            "collect-1",
            amount=25_000,
            transaction_type="collect_request",
            first_time_payee=True,
        )

        result = self.engine.evaluate_observations([row])

        self.assertEqual(result.transactions_with_rule_triggers, 1)
        signal = self._signals(result, "collect-1")[0]
        self.assertEqual(signal.rule_id, LARGE_COLLECT_FIRST_TIME_PAYEE)
        self.assertEqual(signal.severity, "high")
        self.assertEqual(signal.evidence["amount"], 25_000)
        self.assertTrue(signal.evidence["first_time_payee"])

    def test_normal_collect_and_threshold_below_boundary_do_not_trigger(self) -> None:
        rows = [
            observation(
                "below",
                amount=24_999.99,
                transaction_type="collect_request",
                first_time_payee=True,
            ),
            observation(
                "repeat-payee",
                amount=80_000,
                transaction_type="collect_request",
                first_time_payee=False,
            ),
            observation(
                "normal-collect",
                amount=1_000,
                transaction_type="collect_request",
                first_time_payee=True,
            ),
        ]

        result = self.engine.evaluate_observations(rows)

        self.assertEqual(result.rule_trigger_counts[LARGE_COLLECT_FIRST_TIME_PAYEE], 0)

    def test_collect_threshold_is_configurable_and_inclusive(self) -> None:
        engine = RuleEngine(DEFAULT_THRESHOLDS.model_copy(update={"large_collect_amount_threshold": 5_000}))
        result = engine.evaluate_observations(
            [
                observation(
                    "boundary",
                    amount=5_000,
                    transaction_type="collect_request",
                    first_time_payee=True,
                )
            ]
        )

        self.assertEqual(result.rule_trigger_counts[LARGE_COLLECT_FIRST_TIME_PAYEE], 1)

    def test_rapid_linked_transfers_trigger_with_full_account_context(self) -> None:
        rows = [
            observation("r1", offset_seconds=0, sender="A", receiver="B"),
            observation("r2", offset_seconds=120, sender="B", receiver="C"),
            observation("r3", offset_seconds=300, sender="C", receiver="D"),
        ]

        result = self.engine.evaluate_observations(rows)

        self.assertEqual(result.rule_trigger_counts[RAPID_TRANSFER_BEHAVIOUR], 3)
        signal = self._signals(result, "r2")[0]
        self.assertEqual(signal.rule_id, RAPID_TRANSFER_BEHAVIOUR)
        self.assertEqual(signal.evidence["related_transaction_ids"], ["r1", "r2", "r3"])
        self.assertEqual(signal.evidence["account_path"], ["A", "B", "C", "D"])

    def test_unlinked_or_spread_transfers_do_not_trigger_rapid_rule(self) -> None:
        unlinked = [
            observation("u1", offset_seconds=0, sender="A", receiver="B"),
            observation("u2", offset_seconds=1, sender="X", receiver="Y"),
            observation("u3", offset_seconds=2, sender="P", receiver="Q"),
        ]
        spread = [
            observation("s1", offset_seconds=0, sender="A", receiver="B"),
            observation("s2", offset_seconds=150, sender="B", receiver="C"),
            observation("s3", offset_seconds=301, sender="C", receiver="D"),
        ]

        self.assertEqual(
            self.engine.evaluate_observations(unlinked).rule_trigger_counts[
                RAPID_TRANSFER_BEHAVIOUR
            ],
            0,
        )
        self.assertEqual(
            self.engine.evaluate_observations(spread).rule_trigger_counts[
                RAPID_TRANSFER_BEHAVIOUR
            ],
            0,
        )

    def test_same_time_successors_are_considered(self) -> None:
        rows = [
            observation("t1", offset_seconds=0, sender="A", receiver="B"),
            observation("t2", offset_seconds=0, sender="B", receiver="C"),
            observation("t3", offset_seconds=0, sender="C", receiver="D"),
        ]
        result = self.engine.evaluate_observations(rows)
        self.assertEqual(result.rule_trigger_counts[RAPID_TRANSFER_BEHAVIOUR], 3)

    def test_short_cycle_triggers_but_open_path_does_not(self) -> None:
        cycle = [
            observation("c1", offset_seconds=0, sender="A", receiver="B"),
            observation("c2", offset_seconds=45, sender="B", receiver="C"),
            observation("c3", offset_seconds=90, sender="C", receiver="A"),
        ]
        open_path = [
            observation("p1", offset_seconds=0, sender="A", receiver="B"),
            observation("p2", offset_seconds=45, sender="B", receiver="C"),
            observation("p3", offset_seconds=90, sender="C", receiver="D"),
        ]

        cycle_result = self.engine.evaluate_observations(cycle)
        open_result = self.engine.evaluate_observations(open_path)
        self.assertEqual(cycle_result.rule_trigger_counts[CIRCULAR_FLOW_PRECURSOR], 3)
        self.assertEqual(open_result.rule_trigger_counts[CIRCULAR_FLOW_PRECURSOR], 0)

    def test_empty_input_returns_empty_report_and_duplicate_ids_are_rejected(self) -> None:
        empty = self.engine.evaluate_observations([])
        self.assertEqual(empty.total_transactions_evaluated, 0)
        with self.assertRaisesRegex(ValueError, "unique transaction IDs"):
            self.engine.evaluate_observations([observation("same"), observation("same")])

    def test_naive_timestamps_are_rejected(self) -> None:
        row = observation("naive").model_copy(
            update={"timestamp": datetime(2025, 2, 1)}
        )
        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            self.engine.evaluate_observations([row])

    def test_results_are_deterministic_and_ignore_ground_truth_fields(self) -> None:
        source, _ = generate_dataset(seed=42, days=14, baseline_count=30)
        original = self.engine.evaluate_transactions(source.transactions)
        relabeled = [
            row.model_copy(
                update={
                    "label": "safe" if row.label == "fraud" else "fraud",
                    "fraud_pattern": None if row.fraud_pattern else "rapid_mule_chain",
                }
            )
            for row in source.transactions
        ]
        repeated = self.engine.evaluate_transactions(relabeled)
        self.assertEqual(original, repeated)
        self.assertEqual(original, self.engine.evaluate_transactions(source.transactions))

    def test_injected_examples_trigger_expected_rules(self) -> None:
        source, _ = generate_dataset(seed=42, days=14, baseline_count=120)
        report = self.engine.evaluate_transactions(source.transactions)
        signals = {
            row.transaction_id: {signal.rule_id for signal in row.rule_signals}
            for row in report.transaction_evidence
        }
        examples = {
            row.fraud_pattern: row.transaction_id
            for row in source.transactions
            if row.label == "fraud"
            and row.fraud_pattern in {
                "large_collect_first_time_payee",
                "rapid_mule_chain",
                "circular_money_flow",
            }
        }

        self.assertIn(LARGE_COLLECT_FIRST_TIME_PAYEE, signals[examples["large_collect_first_time_payee"]])
        self.assertIn(RAPID_TRANSFER_BEHAVIOUR, signals[examples["rapid_mule_chain"]])
        self.assertIn(CIRCULAR_FLOW_PRECURSOR, signals[examples["circular_money_flow"]])

    def test_csv_loader_ignores_ground_truth_columns(self) -> None:
        dataset_path = Path(__file__).resolve().parents[1] / "data" / "synthetic_transactions.csv"
        if not dataset_path.exists():
            self.skipTest("Default synthetic transaction CSV has not been generated.")
        observations = load_observations_csv(dataset_path)
        self.assertEqual(len(observations), 20_009)
        self.assertFalse(hasattr(observations[0], "label"))
        self.assertFalse(hasattr(observations[0], "fraud_pattern"))


if __name__ == "__main__":
    unittest.main()