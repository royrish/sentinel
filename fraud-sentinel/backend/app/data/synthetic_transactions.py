"""Reproducible synthetic UPI transaction generation and validation CLI."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Sequence

from app.config import get_settings
from app.models.transaction import FraudPattern, Transaction, TransactionType

DEFAULT_START = datetime(2025, 1, 1, tzinfo=timezone.utc)
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "data" / "synthetic_transactions.csv"
FRAUD_PATTERNS: tuple[FraudPattern, ...] = (
    "large_collect_first_time_payee",
    "rapid_mule_chain",
    "circular_money_flow",
)


@dataclass(frozen=True)
class SyntheticDataset:
    """Generated transactions and registries needed for referential validation."""

    transactions: tuple[Transaction, ...]
    account_creation_times: dict[str, datetime]
    device_ids: frozenset[str]
    start: datetime
    end: datetime


@dataclass(frozen=True)
class ValidationReport:
    """Counts from a successfully validated generated dataset."""

    transaction_count: int
    safe_count: int
    fraud_count: int
    fraud_pattern_counts: dict[str, int]


def generate_baseline_transactions(
    *,
    count: int,
    start: datetime,
    days: int,
    rng: random.Random,
    account_creation_times: dict[str, datetime],
    account_devices: dict[str, tuple[str, ...]],
) -> list[Transaction]:
    """Generate ordinary labeled-safe transactions without fraud scenarios."""

    if count < 1:
        raise ValueError("Baseline transaction count must be positive.")
    accounts = tuple(account_creation_times)
    transaction_types: tuple[TransactionType, ...] = (
        "transfer",
        "collect_request",
        "merchant_payment",
        "bill_payment",
    )
    transactions: list[Transaction] = []

    for index in range(count):
        sender, receiver = rng.sample(accounts, 2)
        timestamp = start + timedelta(seconds=rng.randrange(days * 24 * 60 * 60))
        amount = round(min(rng.lognormvariate(6.45, 0.95), 150_000), 2)
        transactions.append(
            Transaction(
                transaction_id=f"TX-{index + 1:08d}",
                timestamp=timestamp,
                sender_account=sender,
                receiver_account=receiver,
                amount=max(amount, 0.01),
                transaction_type=rng.choices(
                    transaction_types, weights=(72, 5, 15, 8), k=1
                )[0],
                device_id=rng.choice(account_devices[sender]),
                sender_account_created_at=account_creation_times[sender],
                receiver_account_created_at=account_creation_times[receiver],
                first_time_payee=False,
                label="safe",
                fraud_pattern=None,
            )
        )

    transactions.sort(key=lambda row: (row.timestamp, row.transaction_id))
    seen_payees: set[tuple[str, str]] = set()
    chronological_transactions: list[Transaction] = []
    for row in transactions:
        pair = (row.sender_account, row.receiver_account)
        chronological_transactions.append(
            row.model_copy(update={"first_time_payee": pair not in seen_payees})
        )
        seen_payees.add(pair)
    return chronological_transactions


def inject_fraud_patterns(
    baseline: Sequence[Transaction],
    *,
    start: datetime,
    days: int,
    account_creation_times: dict[str, datetime],
    account_devices: dict[str, tuple[str, ...]],
) -> list[Transaction]:
    """Append deterministic ground-truth examples of the three target scenarios."""

    if len(account_creation_times) < 14:
        raise ValueError("At least 14 accounts are required to inject scenarios.")
    existing_pairs = {(row.sender_account, row.receiver_account) for row in baseline}
    accounts = tuple(account_creation_times)
    collect_pair = _find_unused_pair(accounts, existing_pairs)
    transactions = list(baseline)
    next_id = len(transactions) + 1
    end = start + timedelta(days=days)

    def append_fraud(
        *,
        timestamp: datetime,
        sender: str,
        receiver: str,
        amount: float,
        transaction_type: TransactionType,
        pattern: FraudPattern,
        first_time_payee: bool = False,
    ) -> None:
        nonlocal next_id
        if timestamp < start or timestamp >= end:
            raise ValueError("Injected timestamp must fall within the dataset period.")
        transactions.append(
            Transaction(
                transaction_id=f"TX-{next_id:08d}",
                timestamp=timestamp,
                sender_account=sender,
                receiver_account=receiver,
                amount=amount,
                transaction_type=transaction_type,
                device_id=account_devices[sender][0],
                sender_account_created_at=account_creation_times[sender],
                receiver_account_created_at=account_creation_times[receiver],
                first_time_payee=first_time_payee,
                label="fraud",
                fraud_pattern=pattern,
            )
        )
        next_id += 1

    # An unused account pair makes the first-time-payee claim true by construction.
    append_fraud(
        timestamp=start + timedelta(hours=8),
        sender=collect_pair[0],
        receiver=collect_pair[1],
        amount=42_500.00,
        transaction_type="collect_request",
        pattern="large_collect_first_time_payee",
        first_time_payee=True,
    )

    mule_accounts = accounts[2:7]
    mule_time = start + timedelta(days=days // 3, hours=2)
    for edge_index, (sender, receiver) in enumerate(zip(mule_accounts, mule_accounts[1:])):
        append_fraud(
            timestamp=mule_time + timedelta(seconds=edge_index * 35),
            sender=sender,
            receiver=receiver,
            amount=9_800.00 - edge_index * 125,
            transaction_type="transfer",
            pattern="rapid_mule_chain",
        )

    ring_accounts = accounts[8:12]
    ring_time = start + timedelta(days=(2 * days) // 3, hours=4)
    ring_edges = tuple(zip(ring_accounts, ring_accounts[1:] + ring_accounts[:1]))
    for edge_index, (sender, receiver) in enumerate(ring_edges):
        append_fraud(
            timestamp=ring_time + timedelta(seconds=edge_index * 45),
            sender=sender,
            receiver=receiver,
            amount=14_200.00 - edge_index * 100,
            transaction_type="transfer",
            pattern="circular_money_flow",
        )

    return transactions


def validate_dataset(
    transactions: Sequence[Transaction],
    *,
    account_creation_times: dict[str, datetime],
    device_ids: frozenset[str],
    start: datetime,
    end: datetime,
    minimum_count: int = 20_000,
) -> ValidationReport:
    """Raise ValueError if records or scenario examples violate the data contract."""

    problems: list[str] = []
    transaction_ids = [row.transaction_id for row in transactions]
    if len(transaction_ids) != len(set(transaction_ids)):
        problems.append("transaction IDs are not unique")
    if len(transactions) < minimum_count:
        problems.append(f"dataset contains fewer than {minimum_count} transactions")

    for row in transactions:
        if not start <= row.timestamp < end:
            problems.append(f"{row.transaction_id} timestamp is outside the dataset period")
        if row.amount <= 0:
            problems.append(f"{row.transaction_id} amount is not positive")
        if row.sender_account not in account_creation_times:
            problems.append(f"{row.transaction_id} references an unknown sender")
        if row.receiver_account not in account_creation_times:
            problems.append(f"{row.transaction_id} references an unknown receiver")
        if row.device_id not in device_ids:
            problems.append(f"{row.transaction_id} references an unknown device")
        if row.sender_account in account_creation_times and (
            row.sender_account_created_at != account_creation_times[row.sender_account]
        ):
            problems.append(f"{row.transaction_id} sender creation time does not match account")
        if row.receiver_account in account_creation_times and (
            row.receiver_account_created_at != account_creation_times[row.receiver_account]
        ):
            problems.append(f"{row.transaction_id} receiver creation time does not match account")
        if row.label == "safe" and row.fraud_pattern is not None:
            problems.append(f"{row.transaction_id} safe record has a fraud pattern")
        if row.label == "fraud" and row.fraud_pattern is None:
            problems.append(f"{row.transaction_id} fraud record has no fraud pattern")

    if not any(
        row.label == "fraud"
        and row.fraud_pattern == "large_collect_first_time_payee"
        and row.transaction_type == "collect_request"
        and row.amount >= 25_000
        and row.first_time_payee
        for row in transactions
    ):
        problems.append("large collect request to a first-time payee is missing")

    if not _has_rapid_path(transactions):
        problems.append("rapid multi-account mule chain is missing")
    if not _has_rapid_cycle(transactions):
        problems.append("circular money-flow pattern is missing")

    if problems:
        raise ValueError("Dataset validation failed: " + "; ".join(problems))

    fraud_rows = [row for row in transactions if row.label == "fraud"]
    pattern_counts = Counter(row.fraud_pattern for row in fraud_rows)
    if any(pattern_counts[pattern] == 0 for pattern in FRAUD_PATTERNS):
        raise ValueError("Dataset validation failed: one or more fraud labels are missing")

    return ValidationReport(
        transaction_count=len(transactions),
        safe_count=len(transactions) - len(fraud_rows),
        fraud_count=len(fraud_rows),
        fraud_pattern_counts={pattern: pattern_counts[pattern] for pattern in FRAUD_PATTERNS},
    )


def generate_dataset(
    *,
    seed: int = 42,
    days: int = 14,
    baseline_count: int = 20_000,
    start: datetime = DEFAULT_START,
) -> tuple[SyntheticDataset, ValidationReport]:
    """Generate and validate a complete deterministic synthetic dataset."""

    if days < 3:
        raise ValueError("Dataset period must be at least 3 days for scenario placement.")
    if baseline_count < 1:
        raise ValueError("Baseline transaction count must be positive.")
    if start.tzinfo is None:
        raise ValueError("Dataset start must include a timezone.")

    rng = random.Random(seed)
    account_creation_times = _create_accounts(start, count=2_000, rng=rng)
    account_devices = _create_devices(account_creation_times, rng=rng)
    device_ids = frozenset(device for devices in account_devices.values() for device in devices)
    baseline = generate_baseline_transactions(
        count=baseline_count,
        start=start,
        days=days,
        rng=rng,
        account_creation_times=account_creation_times,
        account_devices=account_devices,
    )
    transactions = inject_fraud_patterns(
        baseline,
        start=start,
        days=days,
        account_creation_times=account_creation_times,
        account_devices=account_devices,
    )
    transactions.sort(key=lambda row: (row.timestamp, row.transaction_id))
    seen_payees: set[tuple[str, str]] = set()
    chronological_transactions: list[Transaction] = []
    for row in transactions:
        pair = (row.sender_account, row.receiver_account)
        chronological_transactions.append(
            row.model_copy(update={"first_time_payee": pair not in seen_payees})
        )
        seen_payees.add(pair)
    transactions = chronological_transactions
    dataset = SyntheticDataset(
        transactions=tuple(transactions),
        account_creation_times=account_creation_times,
        device_ids=device_ids,
        start=start,
        end=start + timedelta(days=days),
    )
    report = validate_dataset(
        dataset.transactions,
        account_creation_times=dataset.account_creation_times,
        device_ids=dataset.device_ids,
        start=dataset.start,
        end=dataset.end,
        minimum_count=baseline_count,
    )
    return dataset, report


def write_csv(transactions: Iterable[Transaction], output: Path) -> None:
    """Write transaction records as a headered UTF-8 CSV file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(transactions)
    if not rows:
        raise ValueError("Cannot write an empty transaction dataset.")
    fieldnames = list(rows[0].model_dump().keys())
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump(mode="json"))


def _create_accounts(
    start: datetime, *, count: int, rng: random.Random
) -> dict[str, datetime]:
    earliest = start - timedelta(days=730)
    span_seconds = int((start - earliest).total_seconds())
    return {
        f"ACC-{index + 1:06d}": earliest + timedelta(seconds=rng.randrange(span_seconds))
        for index in range(count)
    }


def _create_devices(
    accounts: dict[str, datetime], *, rng: random.Random
) -> dict[str, tuple[str, ...]]:
    device_count = 1_200
    devices = tuple(f"DEV-{index + 1:05d}" for index in range(device_count))
    return {
        account: tuple(rng.sample(devices, rng.choices((1, 2, 3), weights=(75, 20, 5), k=1)[0]))
        for account in accounts
    }


def _find_unused_pair(
    accounts: Sequence[str], existing_pairs: set[tuple[str, str]]
) -> tuple[str, str]:
    for sender in accounts:
        for receiver in accounts:
            if sender != receiver and (sender, receiver) not in existing_pairs:
                return sender, receiver
    raise ValueError("Could not find an unused pair for the first-time-payee scenario.")


def _has_rapid_path(transactions: Sequence[Transaction]) -> bool:
    rows = [
        row
        for row in transactions
        if row.label == "fraud" and row.fraud_pattern == "rapid_mule_chain"
    ]
    rows.sort(key=lambda row: row.timestamp)
    return (
        len(rows) >= 4
        and all(row.transaction_type == "transfer" for row in rows)
        and all(left.receiver_account == right.sender_account for left, right in zip(rows, rows[1:]))
        and rows[-1].timestamp - rows[0].timestamp <= timedelta(minutes=5)
    )


def _has_rapid_cycle(transactions: Sequence[Transaction]) -> bool:
    rows = [
        row
        for row in transactions
        if row.label == "fraud" and row.fraud_pattern == "circular_money_flow"
    ]
    rows.sort(key=lambda row: row.timestamp)
    return (
        len(rows) >= 4
        and all(row.transaction_type == "transfer" for row in rows)
        and all(left.receiver_account == right.sender_account for left, right in zip(rows, rows[1:]))
        and rows[-1].receiver_account == rows[0].sender_account
        and rows[-1].timestamp - rows[0].timestamp <= timedelta(minutes=5)
    )


def _parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=settings.synthetic_seed)
    parser.add_argument("--days", type=int, default=settings.synthetic_days)
    parser.add_argument(
        "--baseline-count",
        type=int,
        default=settings.synthetic_baseline_transactions,
        help="Safe baseline rows; injected fraud rows are added on top.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset, report = generate_dataset(
        seed=args.seed,
        days=args.days,
        baseline_count=args.baseline_count,
    )
    write_csv(dataset.transactions, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "seed": args.seed,
                "period_start": dataset.start.isoformat(),
                "period_end_exclusive": dataset.end.isoformat(),
                "transaction_count": report.transaction_count,
                "safe_count": report.safe_count,
                "fraud_count": report.fraud_count,
                "fraud_pattern_counts": report.fraud_pattern_counts,
                "sample_records": [
                    row.model_dump(mode="json")
                    for row in dataset.transactions[: max(args.sample_size, 0)]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()