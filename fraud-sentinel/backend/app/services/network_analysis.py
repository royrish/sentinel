"""Bounded transaction-network analysis producing independent structured evidence."""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import networkx as nx
from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import Settings, get_settings
from app.models.network_evidence import (
    NetworkEvaluationReport,
    NetworkFinding,
    NetworkFindingCounts,
    NetworkGraphStatistics,
)
from app.models.rule_evidence import TransactionObservation


class NetworkThresholds(BaseModel):
    """Validated thresholds for demo-scale temporal network pattern searches."""

    model_config = ConfigDict(frozen=True)

    rapid_path_window_seconds: int
    rapid_path_minimum_hops: int
    rapid_path_maximum_hops: int
    cycle_window_seconds: int
    cycle_maximum_length: int
    hub_degree_threshold: int
    hub_minimum_transactions: int

    @classmethod
    def from_settings(cls, settings: Settings) -> NetworkThresholds:
        return cls(
            rapid_path_window_seconds=settings.network_rapid_path_window_seconds,
            rapid_path_minimum_hops=settings.network_rapid_path_minimum_hops,
            rapid_path_maximum_hops=settings.network_rapid_path_maximum_hops,
            cycle_window_seconds=settings.network_cycle_window_seconds,
            cycle_maximum_length=settings.network_cycle_maximum_length,
            hub_degree_threshold=settings.network_hub_degree_threshold,
            hub_minimum_transactions=settings.network_hub_minimum_transactions,
        )


@dataclass(frozen=True)
class NetworkTransaction:
    """Observable transaction metadata stored on one directed graph edge."""

    transaction_id: str
    timestamp: datetime
    amount: float
    transaction_type: str


def build_transaction_graph(
    transactions: Sequence[TransactionObservation],
) -> nx.DiGraph:
    """Build an account DiGraph using observable edges, preserving parallel tx metadata."""

    identifiers = [row.transaction_id for row in transactions]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Network construction requires unique transaction IDs.")
    graph = nx.DiGraph()
    graph.graph["transaction_count"] = len(transactions)
    for row in transactions:
        if row.timestamp.tzinfo is None or row.timestamp.utcoffset() is None:
            raise ValueError(f"{row.transaction_id} timestamp must include a timezone.")
        if not row.sender_account or not row.receiver_account or row.amount <= 0:
            raise ValueError(f"{row.transaction_id} has invalid graph edge attributes.")
        graph.add_node(row.sender_account)
        graph.add_node(row.receiver_account)
        transaction = NetworkTransaction(
            transaction_id=row.transaction_id,
            timestamp=row.timestamp,
            amount=row.amount,
            transaction_type=row.transaction_type,
        )
        if graph.has_edge(row.sender_account, row.receiver_account):
            edge_data = graph[row.sender_account][row.receiver_account]
            edge_data["transactions"].append(transaction)
            edge_data["transaction_count"] += 1
            edge_data["total_amount"] += row.amount
        else:
            graph.add_edge(
                row.sender_account,
                row.receiver_account,
                transactions=[transaction],
                transaction_count=1,
                total_amount=row.amount,
            )
    for _, _, edge_data in graph.edges(data=True):
        edge_data["transactions"].sort(key=lambda item: (item.timestamp, item.transaction_id))
    return graph


class TransactionNetworkAnalyzer:
    """Analyze a transaction graph without accessing ground-truth label columns."""

    def __init__(self, thresholds: NetworkThresholds | None = None) -> None:
        self.thresholds = thresholds or NetworkThresholds.from_settings(get_settings())

    def analyze(
        self, transactions: Sequence[TransactionObservation]
    ) -> NetworkEvaluationReport:
        graph = build_transaction_graph(transactions)
        findings = [
            *self._find_rapid_paths(graph),
            *self._find_cycles(graph),
            *self._find_hubs(graph),
        ]
        counts = NetworkFindingCounts(
            rapid_mule_chain=sum(item.pattern_type == "rapid_mule_chain" for item in findings),
            circular_flow=sum(item.pattern_type == "circular_flow" for item in findings),
            high_connectivity_hub=sum(
                item.pattern_type == "high_connectivity_hub" for item in findings
            ),
        )
        component_count = (
            nx.number_weakly_connected_components(graph) if graph.number_of_nodes() else 0
        )
        return NetworkEvaluationReport(
            statistics=NetworkGraphStatistics(
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
                transaction_count=len(transactions),
                weakly_connected_component_count=component_count,
                finding_counts=counts,
            ),
            findings=findings,
        )

    def _find_rapid_paths(self, graph: nx.DiGraph) -> list[NetworkFinding]:
        transfer_edges = _get_transfer_edges(graph)
        findings: dict[tuple[str, ...], NetworkFinding] = {}
        window = timedelta(seconds=self.thresholds.rapid_path_window_seconds)
        outgoing_transactions: dict[str, list[tuple[str, str, NetworkTransaction]]] = defaultdict(list)
        for sender, receiver, transaction in transfer_edges:
            outgoing_transactions[sender].append((sender, receiver, transaction))
        for events in outgoing_transactions.values():
            events.sort(key=lambda item: (item[2].timestamp, item[2].transaction_id))

        for sender, receiver, first in transfer_edges:
            def follow(
                current_receiver: str,
                path: tuple[tuple[str, str, NetworkTransaction], ...],
                deadline,
            ) -> tuple[tuple[str, str, NetworkTransaction], ...] | None:
                hops = len(path)
                if hops < self.thresholds.rapid_path_maximum_hops:
                    candidates = outgoing_transactions.get(current_receiver, [])
                    event_times = [item[2].timestamp for item in candidates]
                    start_index = bisect_right(event_times, path[-1][2].timestamp)
                    end_index = bisect_right(event_times, deadline)
                    used_ids = {item[2].transaction_id for item in path}
                    for edge in candidates[start_index:end_index]:
                        if edge[2].transaction_id in used_ids:
                            continue
                        found = follow(edge[1], (*path, edge), deadline)
                        if found is not None:
                            return found
                if hops >= self.thresholds.rapid_path_minimum_hops:
                    return path
                return None

            path = follow(receiver, ((sender, receiver, first),), first.timestamp + window)
            if path is None:
                continue
            transaction_ids = tuple(sorted(item[2].transaction_id for item in path))
            if transaction_ids in findings:
                continue
            account_path = [path[0][0], *(item[1] for item in path)]
            timestamps = [item[2].timestamp for item in path]
            findings[transaction_ids] = NetworkFinding(
                pattern_type="rapid_mule_chain",
                involved_accounts=account_path,
                transaction_ids=[item[2].transaction_id for item in path],
                severity="medium",
                reason="Linked transfers move through multiple accounts within a short time window",
                metrics={
                    "number_of_hops": len(path),
                    "time_window_seconds": int((timestamps[-1] - timestamps[0]).total_seconds()),
                    "configured_window_seconds": self.thresholds.rapid_path_window_seconds,
                    "total_amount": round(sum(item[2].amount for item in path), 2),
                    "timestamps": [timestamp.isoformat() for timestamp in timestamps],
                },
            )
        transaction_sets = {key: set(key) for key in findings}
        return [
            finding
            for key, finding in findings.items()
            if not any(
                transaction_sets[key] < other_set
                for other_key, other_set in transaction_sets.items()
                if other_key != key
            )
        ]

    def _find_cycles(self, graph: nx.DiGraph) -> list[NetworkFinding]:
        transfer_edges = _get_transfer_edges(graph)
        if not transfer_edges:
            return []
        transfer_edges.sort(key=lambda item: (item[2].timestamp, item[2].transaction_id))
        timestamps = [item[2].timestamp for item in transfer_edges]
        window = timedelta(seconds=self.thresholds.cycle_window_seconds)
        step_seconds = max(1, self.thresholds.cycle_window_seconds // 2)
        first_time, last_time = timestamps[0], timestamps[-1]
        findings: dict[tuple[str, ...], NetworkFinding] = {}
        window_start = first_time
        while window_start <= last_time:
            window_end = window_start + window
            start_index = bisect_right(timestamps, window_start - timedelta(microseconds=1))
            end_index = bisect_right(timestamps, window_end)
            window_edges = transfer_edges[start_index:end_index]
            if len(window_edges) >= 3:
                temporal_graph = nx.DiGraph()
                pair_transactions: dict[tuple[str, str], list[NetworkTransaction]] = defaultdict(list)
                for sender, receiver, transaction in window_edges:
                    temporal_graph.add_edge(sender, receiver)
                    pair_transactions[(sender, receiver)].append(transaction)
                for cycle in nx.simple_cycles(
                    temporal_graph,
                    length_bound=self.thresholds.cycle_maximum_length,
                ):
                    if not 3 <= len(cycle) <= self.thresholds.cycle_maximum_length:
                        continue
                    selected = _select_temporal_cycle_edges(cycle, pair_transactions, window_end)
                    if selected is None:
                        continue
                    key = tuple(sorted(item[2].transaction_id for item in selected))
                    if key in findings:
                        continue
                    cycle_accounts = [selected[0][0], *(item[1] for item in selected)]
                    cycle_timestamps = [item[2].timestamp for item in selected]
                    elapsed = int((cycle_timestamps[-1] - cycle_timestamps[0]).total_seconds())
                    findings[key] = NetworkFinding(
                        pattern_type="circular_flow",
                        involved_accounts=cycle_accounts[:-1],
                        transaction_ids=[item[2].transaction_id for item in selected],
                        severity="medium",
                        reason="Directed transfers form a bounded, time-localized account cycle",
                        metrics={
                            "cycle_length": len(selected),
                            "time_window_seconds": elapsed,
                            "configured_window_seconds": self.thresholds.cycle_window_seconds,
                            "total_amount": round(sum(item[2].amount for item in selected), 2),
                            "account_cycle": cycle_accounts,
                            "timestamps": [timestamp.isoformat() for timestamp in cycle_timestamps],
                        },
                    )
            window_start += timedelta(seconds=step_seconds)
        return list(findings.values())

    def _find_hubs(self, graph: nx.DiGraph) -> list[NetworkFinding]:
        findings: list[NetworkFinding] = []
        for account in sorted(graph.nodes):
            in_degree = graph.in_degree(account)
            out_degree = graph.out_degree(account)
            total_degree = in_degree + out_degree
            incident_edges = list(graph.in_edges(account, data=True)) + list(
                graph.out_edges(account, data=True)
            )
            incident_transactions = {
                transaction.transaction_id: transaction
                for _, _, edge_data in incident_edges
                for transaction in edge_data["transactions"]
            }
            transaction_count = len(incident_transactions)
            weighted_volume = sum(
                transaction.amount for transaction in incident_transactions.values()
            )
            if (
                total_degree < self.thresholds.hub_degree_threshold
                or transaction_count < self.thresholds.hub_minimum_transactions
            ):
                continue
            transaction_ids = sorted(incident_transactions)
            findings.append(
                NetworkFinding(
                    pattern_type="high_connectivity_hub",
                    involved_accounts=[account],
                    transaction_ids=transaction_ids,
                    severity="low",
                    reason="Account has high observed transaction connectivity; this is not a fraud verdict",
                    metrics={
                        "in_degree": in_degree,
                        "out_degree": out_degree,
                        "total_degree": total_degree,
                        "hub_degree_threshold": self.thresholds.hub_degree_threshold,
                        "transaction_count": transaction_count,
                        "minimum_transaction_count": self.thresholds.hub_minimum_transactions,
                        "weighted_transaction_volume": round(weighted_volume, 2),
                    },
                )
            )
        return findings


def _get_transfer_edges(
    graph: nx.DiGraph,
) -> list[tuple[str, str, NetworkTransaction]]:
    return [
        (sender, receiver, transaction)
        for sender, receiver, edge_data in graph.edges(data=True)
        for transaction in edge_data["transactions"]
        if transaction.transaction_type == "transfer"
    ]


def _select_temporal_cycle_edges(
    cycle: Sequence[str],
    pair_transactions: dict[tuple[str, str], list[NetworkTransaction]],
    window_end,
) -> list[tuple[str, str, NetworkTransaction]] | None:
    for rotation in range(len(cycle)):
        rotated_cycle = (*cycle[rotation:], *cycle[:rotation])
        selected: list[tuple[str, str, NetworkTransaction]] = []
        previous_timestamp = None
        for index, sender in enumerate(rotated_cycle):
            receiver = rotated_cycle[(index + 1) % len(rotated_cycle)]
            candidates = pair_transactions[(sender, receiver)]
            candidate_times = [item.timestamp for item in candidates]
            candidate_index = (
                bisect_right(candidate_times, previous_timestamp)
                if previous_timestamp is not None
                else 0
            )
            if candidate_index >= len(candidates):
                break
            transaction = candidates[candidate_index]
            if transaction.timestamp > window_end:
                break
            selected.append((sender, receiver, transaction))
            previous_timestamp = transaction.timestamp
        if len(selected) == len(rotated_cycle):
            return selected
    return None


def load_network_observations_csv(path: Path) -> list[TransactionObservation]:
    """Read only graph-observable transaction fields; ground-truth columns are ignored."""

    required_fields = set(TransactionObservation.model_fields)
    observations: list[TransactionObservation] = []
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(required_fields.difference(reader.fieldnames))
        if missing:
            raise ValueError("CSV is missing observable transaction columns: " + ", ".join(missing))
        for line_number, row in enumerate(reader, start=2):
            try:
                observations.append(TransactionObservation.model_validate(row))
            except ValidationError as error:
                raise ValueError(f"Invalid transaction data on CSV line {line_number}: {error}") from error
    if not observations:
        raise ValueError(f"CSV contains no transactions: {path}")
    return observations


def evaluate_csv(
    input_path: Path,
    output_path: Path,
    *,
    settings: Settings | None = None,
) -> NetworkEvaluationReport:
    observations = load_network_observations_csv(input_path)
    report = TransactionNetworkAnalyzer(NetworkThresholds.from_settings(settings or get_settings())).analyze(
        observations
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "synthetic_transactions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "network_evaluation.json",
    )
    args = parser.parse_args()
    report = evaluate_csv(args.input, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "graph_statistics": report.statistics.model_dump(),
                "finding_count": len(report.findings),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()