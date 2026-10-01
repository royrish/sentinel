"""Tests for observation-only transaction-network analysis."""

import csv
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import Settings
from app.data.synthetic_transactions import generate_dataset
from app.models.rule_evidence import TransactionObservation
from app.models.transaction import Transaction, TransactionType
from app.services.network_analysis import (
    NetworkThresholds,
    TransactionNetworkAnalyzer,
    build_transaction_graph,
    evaluate_csv,
    load_network_observations_csv,
)

START = datetime(2025, 3, 1, tzinfo=timezone.utc)
CREATED = datetime(2023, 1, 1, tzinfo=timezone.utc)


def observation(
    transaction_id: str,
    sender: str,
    receiver: str,
    *,
    offset_seconds: int = 0,
    amount: float = 100,
    transaction_type: TransactionType = "transfer",
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
        first_time_payee=True,
    )


def thresholds(
    *,
    rapid_window: int = 300,
    minimum_hops: int = 3,
    maximum_hops: int = 6,
    cycle_window: int = 300,
    maximum_cycle_length: int = 4,
    hub_degree: int = 31,
    hub_transactions: int = 12,
) -> NetworkThresholds:
    return NetworkThresholds(
        rapid_path_window_seconds=rapid_window,
        rapid_path_minimum_hops=minimum_hops,
        rapid_path_maximum_hops=maximum_hops,
        cycle_window_seconds=cycle_window,
        cycle_maximum_length=maximum_cycle_length,
        hub_degree_threshold=hub_degree,
        hub_minimum_transactions=hub_transactions,
    )


class NetworkAnalysisTests(unittest.TestCase):
    def test_graph_nodes_directed_edges_and_parallel_metadata(self) -> None:
        rows = [
            observation("tx-1", "A", "B", amount=25),
            observation(
                "tx-2",
                "A",
                "B",
                offset_seconds=20,
                amount=30,
                transaction_type="collect_request",
            ),
            observation("tx-3", "B", "C", offset_seconds=40, amount=45),
        ]

        graph = build_transaction_graph(rows)

        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 2)
        self.assertTrue(graph.has_edge("A", "B"))
        self.assertFalse(graph.has_edge("B", "A"))
        self.assertEqual(graph["A"]["B"]["transaction_count"], 2)
        self.assertEqual(graph["A"]["B"]["total_amount"], 55)
        self.assertEqual(
            [item.transaction_id for item in graph["A"]["B"]["transactions"]],
            ["tx-1", "tx-2"],
        )
        self.assertEqual(graph["A"]["B"]["transactions"][1].transaction_type, "collect_request")
        self.assertEqual(graph["A"]["B"]["transactions"][1].amount, 30)

    def test_rapid_multi_account_path_has_context_and_window(self) -> None:
        rows = [
            observation("r1", "A", "B", offset_seconds=0, amount=100),
            observation("r2", "B", "C", offset_seconds=100, amount=200),
            observation("r3", "C", "D", offset_seconds=300, amount=300),
        ]
        report = TransactionNetworkAnalyzer(thresholds()).analyze(rows)
        findings = [item for item in report.findings if item.pattern_type == "rapid_mule_chain"]

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].transaction_ids, ["r1", "r2", "r3"])
        self.assertEqual(findings[0].involved_accounts, ["A", "B", "C", "D"])
        self.assertEqual(findings[0].metrics["number_of_hops"], 3)
        self.assertEqual(findings[0].metrics["time_window_seconds"], 300)
        self.assertEqual(findings[0].metrics["total_amount"], 600)

    def test_rapid_path_respects_minimum_hops_and_window(self) -> None:
        too_short = [
            observation("s1", "A", "B", offset_seconds=0),
            observation("s2", "B", "C", offset_seconds=10),
        ]
        outside_window = [
            observation("w1", "A", "B", offset_seconds=0),
            observation("w2", "B", "C", offset_seconds=150),
            observation("w3", "C", "D", offset_seconds=301),
        ]

        self.assertFalse(
            any(item.pattern_type == "rapid_mule_chain" for item in TransactionNetworkAnalyzer(thresholds()).analyze(too_short).findings)
        )
        self.assertFalse(
            any(item.pattern_type == "rapid_mule_chain" for item in TransactionNetworkAnalyzer(thresholds()).analyze(outside_window).findings)
        )

    def test_four_account_cycle_uses_bounded_networkx_cycles(self) -> None:
        rows = [
            observation("c1", "A", "B", offset_seconds=0, amount=10),
            observation("c2", "B", "C", offset_seconds=40, amount=20),
            observation("c3", "C", "D", offset_seconds=80, amount=30),
            observation("c4", "D", "A", offset_seconds=120, amount=40),
        ]

        report = TransactionNetworkAnalyzer(thresholds()).analyze(rows)
        findings = [item for item in report.findings if item.pattern_type == "circular_flow"]

        self.assertEqual(len(findings), 1)
        self.assertCountEqual(findings[0].transaction_ids, ["c1", "c2", "c3", "c4"])
        self.assertEqual(findings[0].metrics["cycle_length"], 4)
        self.assertEqual(findings[0].metrics["total_amount"], 100)

    def test_cycle_minimum_length_and_time_window_are_enforced(self) -> None:
        two_edge_cycle = [
            observation("a1", "A", "B", offset_seconds=0),
            observation("a2", "B", "A", offset_seconds=10),
        ]
        slow_cycle = [
            observation("b1", "A", "B", offset_seconds=0),
            observation("b2", "B", "C", offset_seconds=100),
            observation("b3", "C", "A", offset_seconds=301),
        ]
        analyzer = TransactionNetworkAnalyzer(thresholds())

        self.assertEqual(analyzer.analyze(two_edge_cycle).statistics.finding_counts.circular_flow, 0)
        self.assertEqual(analyzer.analyze(slow_cycle).statistics.finding_counts.circular_flow, 0)

    def test_hub_evidence_includes_metrics_and_thresholds(self) -> None:
        rows = [
            observation("o1", "H", "A"),
            observation("o2", "H", "B", offset_seconds=10),
            observation("o3", "H", "C", offset_seconds=20),
            observation("i1", "D", "H", offset_seconds=30),
            observation("i2", "E", "H", offset_seconds=40),
            observation("i3", "F", "H", offset_seconds=50),
        ]
        report = TransactionNetworkAnalyzer(
            thresholds(hub_degree=6, hub_transactions=6)
        ).analyze(rows)
        findings = [item for item in report.findings if item.pattern_type == "high_connectivity_hub"]

        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].involved_accounts, ["H"])
        self.assertEqual(findings[0].metrics["in_degree"], 3)
        self.assertEqual(findings[0].metrics["out_degree"], 3)
        self.assertEqual(findings[0].metrics["total_degree"], 6)
        self.assertEqual(findings[0].metrics["hub_degree_threshold"], 6)
        self.assertEqual(findings[0].metrics["transaction_count"], 6)
        self.assertEqual(findings[0].metrics["weighted_transaction_volume"], 600)

    def test_settings_configure_network_thresholds(self) -> None:
        settings = Settings(
            network_rapid_path_window_seconds=90,
            network_rapid_path_minimum_hops=2,
            network_rapid_path_maximum_hops=4,
            network_cycle_window_seconds=120,
            network_cycle_maximum_length=3,
            network_hub_degree_threshold=15,
            network_hub_minimum_transactions=8,
        )
        configured = NetworkThresholds.from_settings(settings)
        self.assertEqual(configured.rapid_path_window_seconds, 90)
        self.assertEqual(configured.rapid_path_minimum_hops, 2)
        self.assertEqual(configured.cycle_maximum_length, 3)
        self.assertEqual(configured.hub_degree_threshold, 15)

    def test_empty_and_invalid_input(self) -> None:
        empty_report = TransactionNetworkAnalyzer(thresholds()).analyze([])
        self.assertEqual(empty_report.statistics.node_count, 0)
        self.assertEqual(empty_report.statistics.edge_count, 0)
        self.assertEqual(empty_report.statistics.weakly_connected_component_count, 0)

        with self.assertRaisesRegex(ValueError, "unique transaction IDs"):
            build_transaction_graph([observation("same", "A", "B"), observation("same", "B", "C")])
        naive = observation("naive", "A", "B").model_copy(
            update={"timestamp": datetime(2025, 3, 1)}
        )
        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            build_transaction_graph([naive])

    def test_ground_truth_mutations_and_removal_do_not_change_graph_or_findings(self) -> None:
        dataset, _ = generate_dataset(seed=42, days=14, baseline_count=100)
        transactions = list(dataset.transactions)
        mutated = [
            item.model_copy(
                update={
                    "label": "safe" if item.label == "fraud" else "fraud",
                    "fraud_pattern": (
                        "rapid_mule_chain"
                        if item.fraud_pattern != "rapid_mule_chain"
                        else "circular_money_flow"
                    ),
                }
            )
            for item in transactions
        ]

        def observable(rows: list[Transaction]) -> list[TransactionObservation]:
            return [
                TransactionObservation.model_validate(
                    row.model_dump(exclude={"label", "fraud_pattern"})
                )
                for row in rows
            ]

        original_observations = observable(transactions)
        mutated_observations = observable(mutated)
        analyzer = TransactionNetworkAnalyzer(thresholds())
        self.assertEqual(
            list(build_transaction_graph(original_observations).edges(data=True)),
            list(build_transaction_graph(mutated_observations).edges(data=True)),
        )
        self.assertEqual(analyzer.analyze(original_observations), analyzer.analyze(mutated_observations))

    def test_injected_structures_are_detected_without_scenario_metadata(self) -> None:
        dataset, _ = generate_dataset(seed=42, days=14, baseline_count=120)
        observations = [
            TransactionObservation.model_validate(
                item.model_dump(exclude={"label", "fraud_pattern"})
            )
            for item in dataset.transactions
        ]
        report = TransactionNetworkAnalyzer(thresholds()).analyze(observations)
        all_finding_ids = {
            transaction_id for finding in report.findings for transaction_id in finding.transaction_ids
        }
        chain_ids = {f"TX-{number:08d}" for number in range(122, 126)}
        cycle_ids = {f"TX-{number:08d}" for number in range(126, 130)}

        self.assertTrue(chain_ids.issubset(all_finding_ids))
        self.assertTrue(cycle_ids.issubset(all_finding_ids))
        self.assertGreaterEqual(report.statistics.finding_counts.rapid_mule_chain, 1)
        self.assertGreaterEqual(report.statistics.finding_counts.circular_flow, 1)

    def test_csv_batch_ignores_ground_truth_and_preserves_input(self) -> None:
        dataset, _ = generate_dataset(seed=18, days=14, baseline_count=15)
        rows = [item.model_dump(mode="json") for item in dataset.transactions]
        fieldnames = list(rows[0])
        for row in rows:
            row["label"] = "not-used"
            row["fraud_pattern"] = "not-a-scenario-feature"

        with tempfile.TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "transactions.csv"
            output_path = Path(temporary_directory) / "network.json"
            with input_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            before = input_path.read_bytes()

            report = evaluate_csv(input_path, output_path)

            self.assertEqual(input_path.read_bytes(), before)
            self.assertEqual(report.statistics.transaction_count, len(rows))
            self.assertTrue(output_path.exists())
            parsed = load_network_observations_csv(input_path)
            self.assertFalse(hasattr(parsed[0], "label"))
            self.assertFalse(hasattr(parsed[0], "fraud_pattern"))


if __name__ == "__main__":
    unittest.main()