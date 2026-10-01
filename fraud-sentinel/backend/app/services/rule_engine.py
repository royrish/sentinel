"""Deterministic rules that emit evidence without making fraud decisions."""

from __future__ import annotations

import csv
import json
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import Settings, get_settings
from app.models.rule_evidence import (
    RuleEvaluationReport,
    RuleSignal,
    TransactionObservation,
    TransactionRuleEvidence,
)
from app.models.transaction import Transaction

LARGE_COLLECT_FIRST_TIME_PAYEE = "LARGE_COLLECT_FIRST_TIME_PAYEE"
RAPID_TRANSFER_BEHAVIOUR = "RAPID_TRANSFER_BEHAVIOUR"
CIRCULAR_FLOW_PRECURSOR = "CIRCULAR_FLOW_PRECURSOR"


class RuleThresholds(BaseModel):
    """Validated rule settings; thresholds can be supplied directly in tests."""

    model_config = ConfigDict(frozen=True)

    large_collect_amount_threshold: float = Field(gt=0)
    rapid_transfer_window_seconds: int = Field(gt=0)
    rapid_transfer_minimum_count: int = Field(ge=2)
    circular_flow_window_seconds: int = Field(gt=0)
    circular_flow_max_transfers: int = Field(ge=3, le=8)

    @classmethod
    def from_settings(cls, settings: Settings) -> RuleThresholds:
        return cls(
            large_collect_amount_threshold=settings.rule_large_collect_amount_threshold,
            rapid_transfer_window_seconds=settings.rule_rapid_transfer_window_seconds,
            rapid_transfer_minimum_count=settings.rule_rapid_transfer_minimum_count,
            circular_flow_window_seconds=settings.rule_circular_flow_window_seconds,
            circular_flow_max_transfers=settings.rule_circular_flow_max_transfers,
        )


class RuleEngine:
    """Evaluate observable transaction attributes and produce structured signals."""

    def __init__(self, thresholds: RuleThresholds | None = None) -> None:
        self.thresholds = thresholds or RuleThresholds.from_settings(get_settings())

    def evaluate_transactions(
        self, transactions: Sequence[Transaction]
    ) -> RuleEvaluationReport:
        """Evaluate source records after projecting away their ground-truth fields."""

        observations = [
            TransactionObservation(
                transaction_id=row.transaction_id,
                timestamp=row.timestamp,
                sender_account=row.sender_account,
                receiver_account=row.receiver_account,
                amount=row.amount,
                transaction_type=row.transaction_type,
                device_id=row.device_id,
                sender_account_created_at=row.sender_account_created_at,
                receiver_account_created_at=row.receiver_account_created_at,
                first_time_payee=row.first_time_payee,
            )
            for row in transactions
        ]
        return self.evaluate_observations(observations)

    def evaluate_observations(
        self, observations: Sequence[TransactionObservation]
    ) -> RuleEvaluationReport:
        """Evaluate observation-only rows; empty input returns an empty report."""

        ids = [row.transaction_id for row in observations]
        if len(ids) != len(set(ids)):
            raise ValueError("Rule evaluation requires unique transaction IDs.")
        for row in observations:
            if row.timestamp.tzinfo is None or row.timestamp.utcoffset() is None:
                raise ValueError(f"{row.transaction_id} timestamp must include a timezone.")

        ordered = sorted(observations, key=lambda row: (row.timestamp, row.transaction_id))
        signals_by_id: dict[str, list[RuleSignal]] = {row.transaction_id: [] for row in ordered}

        for row in ordered:
            if (
                row.transaction_type == "collect_request"
                and row.first_time_payee
                and row.amount >= self.thresholds.large_collect_amount_threshold
            ):
                signals_by_id[row.transaction_id].append(
                    RuleSignal(
                        rule_id=LARGE_COLLECT_FIRST_TIME_PAYEE,
                        severity="high",
                        reason="Large collect request involving a first-time payee",
                        evidence={
                            "amount": row.amount,
                            "amount_threshold": self.thresholds.large_collect_amount_threshold,
                            "transaction_type": row.transaction_type,
                            "first_time_payee": row.first_time_payee,
                        },
                    )
                )

        transfers_by_sender = _index_transfers(ordered)
        for sequence in _find_rapid_transfer_sequences(
            ordered,
            transfers_by_sender,
            minimum_count=self.thresholds.rapid_transfer_minimum_count,
            window_seconds=self.thresholds.rapid_transfer_window_seconds,
        ):
            ids_in_sequence = [row.transaction_id for row in sequence]
            accounts = [sequence[0].sender_account, *(row.receiver_account for row in sequence)]
            for row in sequence:
                _append_once(
                    signals_by_id[row.transaction_id],
                    RuleSignal(
                        rule_id=RAPID_TRANSFER_BEHAVIOUR,
                        severity="medium",
                        reason=(
                            f"Part of {len(sequence)} linked transfers within "
                            f"{self.thresholds.rapid_transfer_window_seconds} seconds"
                        ),
                        evidence={
                            "related_transaction_ids": ids_in_sequence,
                            "account_path": accounts,
                            "transfer_count": len(sequence),
                            "window_seconds": self.thresholds.rapid_transfer_window_seconds,
                        },
                    ),
                )

        for cycle in _find_short_cycles(
            ordered,
            transfers_by_sender,
            window_seconds=self.thresholds.circular_flow_window_seconds,
            maximum_transfers=self.thresholds.circular_flow_max_transfers,
        ):
            ids_in_cycle = [row.transaction_id for row in cycle]
            accounts = [cycle[0].sender_account, *(row.receiver_account for row in cycle)]
            elapsed_seconds = int((cycle[-1].timestamp - cycle[0].timestamp).total_seconds())
            for row in cycle:
                _append_once(
                    signals_by_id[row.transaction_id],
                    RuleSignal(
                        rule_id=CIRCULAR_FLOW_PRECURSOR,
                        severity="medium",
                        reason="Short sequence of transfers returns to its starting account",
                        evidence={
                            "related_transaction_ids": ids_in_cycle,
                            "account_cycle": accounts,
                            "transfer_count": len(cycle),
                            "elapsed_seconds": elapsed_seconds,
                            "window_seconds": self.thresholds.circular_flow_window_seconds,
                        },
                    ),
                )

        evaluations = [
            TransactionRuleEvidence(
                transaction_id=row.transaction_id,
                rule_signals=signals_by_id[row.transaction_id],
            )
            for row in ordered
        ]
        rule_ids = (
            LARGE_COLLECT_FIRST_TIME_PAYEE,
            RAPID_TRANSFER_BEHAVIOUR,
            CIRCULAR_FLOW_PRECURSOR,
        )
        counts = {
            rule_id: sum(
                any(signal.rule_id == rule_id for signal in evaluation.rule_signals)
                for evaluation in evaluations
            )
            for rule_id in rule_ids
        }
        triggered_by_rule = {
            rule_id: [
                evaluation.transaction_id
                for evaluation in evaluations
                if any(signal.rule_id == rule_id for signal in evaluation.rule_signals)
            ]
            for rule_id in rule_ids
        }
        triggered_ids = [
            evaluation.transaction_id
            for evaluation in evaluations
            if evaluation.rule_signals
        ]
        return RuleEvaluationReport(
            total_transactions_evaluated=len(evaluations),
            transactions_with_rule_triggers=len(triggered_ids),
            rule_trigger_counts=counts,
            triggered_transaction_ids_by_rule=triggered_by_rule,
            triggered_transaction_ids=triggered_ids,
            transaction_evidence=evaluations,
        )


def load_observations_csv(path: Path) -> list[TransactionObservation]:
    """Load only rule-observable columns; ground-truth CSV fields are ignored."""

    required_fields = set(TransactionObservation.model_fields)
    observations: list[TransactionObservation] = []
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = sorted(required_fields.difference(reader.fieldnames))
        if missing:
            raise ValueError("CSV is missing required observable columns: " + ", ".join(missing))
        for line_number, row in enumerate(reader, start=2):
            try:
                observations.append(TransactionObservation.model_validate(row))
            except ValidationError as error:
                raise ValueError(f"Invalid observable data on CSV line {line_number}: {error}") from error
    if not observations:
        raise ValueError(f"CSV contains no transactions: {path}")
    return observations


def evaluate_csv(input_path: Path, output_path: Path) -> RuleEvaluationReport:
    """Run configured rules over an input CSV and save a separate JSON report."""

    observations = load_observations_csv(input_path)
    report = RuleEngine().evaluate_observations(observations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    return report


def _index_transfers(
    ordered: Sequence[TransactionObservation],
) -> dict[str, list[TransactionObservation]]:
    by_sender: dict[str, list[TransactionObservation]] = defaultdict(list)
    for row in ordered:
        if row.transaction_type == "transfer":
            by_sender[row.sender_account].append(row)
    return dict(by_sender)


def _find_rapid_transfer_sequences(
    ordered: Sequence[TransactionObservation],
    by_sender: dict[str, list[TransactionObservation]],
    *,
    minimum_count: int,
    window_seconds: int,
) -> list[tuple[TransactionObservation, ...]]:
    """Find one bounded linked path per starting transfer; no graph is constructed."""

    result: dict[tuple[str, ...], tuple[TransactionObservation, ...]] = {}
    window = timedelta(seconds=window_seconds)

    def search(
        path: tuple[TransactionObservation, ...],
        end_time,
    ) -> tuple[TransactionObservation, ...] | None:
        if len(path) >= minimum_count:
            return path
        previous = path[-1]
        candidates = _eligible_successors(
            by_sender.get(previous.receiver_account, []),
            after=previous.timestamp,
            before=end_time,
        )
        for candidate in candidates:
            if candidate.transaction_id in {item.transaction_id for item in path}:
                continue
            found = search((*path, candidate), end_time)
            if found is not None:
                return found
        return None

    for row in ordered:
        if row.transaction_type != "transfer":
            continue
        path = search((row,), row.timestamp + window)
        if path is not None:
            result.setdefault(tuple(item.transaction_id for item in path), path)
    return list(result.values())


def _find_short_cycles(
    ordered: Sequence[TransactionObservation],
    by_sender: dict[str, list[TransactionObservation]],
    *,
    window_seconds: int,
    maximum_transfers: int,
) -> list[tuple[TransactionObservation, ...]]:
    """Match simple 3-to-N transfer closures as a precursor, not general graph analysis."""

    result: dict[tuple[str, ...], tuple[TransactionObservation, ...]] = {}
    window = timedelta(seconds=window_seconds)

    def search(
        start_account: str,
        path: tuple[TransactionObservation, ...],
        visited_accounts: frozenset[str],
        end_time,
    ) -> tuple[TransactionObservation, ...] | None:
        previous = path[-1]
        candidates = _eligible_successors(
            by_sender.get(previous.receiver_account, []),
            after=previous.timestamp,
            before=end_time,
        )
        for candidate in candidates:
            if candidate.transaction_id in {item.transaction_id for item in path}:
                continue
            next_path = (*path, candidate)
            if candidate.receiver_account == start_account and len(next_path) >= 3:
                return next_path
            if (
                len(next_path) < maximum_transfers
                and candidate.receiver_account not in visited_accounts
            ):
                found = search(
                    start_account,
                    next_path,
                    visited_accounts | {candidate.receiver_account},
                    end_time,
                )
                if found is not None:
                    return found
        return None

    for row in ordered:
        if row.transaction_type != "transfer":
            continue
        path = search(
            row.sender_account,
            (row,),
            frozenset({row.sender_account, row.receiver_account}),
            row.timestamp + window,
        )
        if path is not None:
            # Rotation-equivalent cycles are deduplicated by their transaction IDs.
            key = tuple(sorted(item.transaction_id for item in path))
            result.setdefault(key, path)
    return list(result.values())


def _eligible_successors(
    candidates: Sequence[TransactionObservation], *, after, before
) -> Sequence[TransactionObservation]:
    timestamps = [row.timestamp for row in candidates]
    start = bisect_left(timestamps, after)
    end = bisect_right(timestamps, before)
    return candidates[start:end]


def _append_once(signals: list[RuleSignal], signal: RuleSignal) -> None:
    if not any(existing.rule_id == signal.rule_id for existing in signals):
        signals.append(signal)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate deterministic rules over transactions.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "synthetic_transactions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "rule_evaluation.json",
    )
    args = parser.parse_args()
    report = evaluate_csv(args.input, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "total_transactions_evaluated": report.total_transactions_evaluated,
                "transactions_with_rule_triggers": report.transactions_with_rule_triggers,
                "rule_trigger_counts": report.rule_trigger_counts,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()