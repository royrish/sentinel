"""Typed NetworkX findings and batch graph statistics."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

NetworkPatternType = Literal["rapid_mule_chain", "circular_flow", "high_connectivity_hub"]


class NetworkFinding(BaseModel):
    """One explainable network structure; not a fraud verdict or risk score."""

    model_config = ConfigDict(frozen=True)

    pattern_type: NetworkPatternType
    involved_accounts: list[str] = Field(min_length=1)
    transaction_ids: list[str] = Field(min_length=1)
    severity: Literal["low", "medium", "high"]
    reason: str
    metrics: dict[str, int | float | str | list[str]]


class NetworkFindingCounts(BaseModel):
    rapid_mule_chain: int
    circular_flow: int
    high_connectivity_hub: int


class NetworkGraphStatistics(BaseModel):
    node_count: int
    edge_count: int
    transaction_count: int
    weakly_connected_component_count: int
    finding_counts: NetworkFindingCounts


class NetworkEvaluationReport(BaseModel):
    """Independent graph findings with no ground-truth target fields."""

    model_name: Literal["NetworkX"] = "NetworkX"
    graph_model: Literal["directed_account_graph"] = "directed_account_graph"
    ground_truth_used: Literal[False] = False
    statistics: NetworkGraphStatistics
    findings: list[NetworkFinding]