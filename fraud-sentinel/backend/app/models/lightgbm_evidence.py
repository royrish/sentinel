"""Structured output for the supervised LightGBM model signal."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.rule_evidence import TransactionObservation


class LabeledTransactionObservation(TransactionObservation):
    """Observation fields plus supervised target; ground-truth pattern is not modeled."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    label: Literal["safe", "fraud"]


class LightGBMConfiguration(BaseModel):
    """Effective model, split, threshold, and imbalance settings."""

    test_size: float
    random_state: int
    prediction_threshold: float
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    scale_pos_weight: float


class LightGBMEvidence(BaseModel):
    """One model prediction, explicitly scoped as evidence rather than a decision."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str
    fraud_probability: float = Field(ge=0, le=1)
    predicted_fraud: bool
    model_name: Literal["LightGBM"] = "LightGBM"
    dataset_partition: Literal["train", "test"]


class HoldoutMetrics(BaseModel):
    """Metrics evaluated only for stratified holdout rows."""

    evaluation_scope: Literal["synthetic_dataset_holdout"] = "synthetic_dataset_holdout"
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    average_precision: float | None
    confusion_matrix: dict[Literal["tn", "fp", "fn", "tp"], int]


class LightGBMEvaluationReport(BaseModel):
    """Batch model report with holdout metrics and separate predictions."""

    model_name: Literal["LightGBM"] = "LightGBM"
    target_definition: Literal["label == fraud"] = "label == fraud"
    split_strategy: Literal["stratified_train_test_split"] = "stratified_train_test_split"
    class_imbalance_strategy: Literal["scale_pos_weight_from_training_partition"] = (
        "scale_pos_weight_from_training_partition"
    )
    metrics_note: Literal["Metrics are calculated only on the held-out test set."] = (
        "Metrics are calculated only on the held-out test set."
    )
    feature_names: list[str]
    dataset_size: int
    train_size: int
    test_size: int
    class_counts: dict[Literal["safe", "fraud"], int]
    train_class_counts: dict[Literal["safe", "fraud"], int]
    test_class_counts: dict[Literal["safe", "fraud"], int]
    configuration: LightGBMConfiguration
    metrics: HoldoutMetrics
    predictions: list[LightGBMEvidence]
    predicted_fraud_count: int
