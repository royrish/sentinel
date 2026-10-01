# Fraud Sentinel

Fraud Sentinel is an explainable UPI fraud investigation platform. The frontend and backend are separate applications. The backend generates synthetic transactions and independent rule, anomaly, supervised-model, and network evidence, which a demo Risk Engine combines into review alerts. No model signal or risk score is a final fraud decision or production performance claim.

## Project Structure

```text
fraud-sentinel/
|-- backend/   FastAPI service and future fraud engine
|-- frontend/  Next.js analyst interface
|-- README.md
`-- .gitignore
```

## Architecture

```text
                    FRAUD SENTINEL
                         |
              +----------+----------+
              |                     |
          FRONTEND               BACKEND
        Next.js/TS              FastAPI
              |                     |
              |              +------+------+
              |              |             |
              |          Detection      Analytics
              |              |
              |      +-------+--------+
              |      |       |        |
              |    Rules     ML     NetworkX
              |             |
              |      Isolation Forest
              |      LightGBM
              |
              +--------- REST API --------+
```

The eventual detection flow is planned as:

```text
Detection Engine -> Structured Evidence -> Risk Engine -> Claude -> Human-readable explanation
```

Fraud decisions will be made by deterministic rules, trained models, and network analysis. Claude is strictly an explanation layer: it will receive structured evidence from the detection engine and will not decide whether a transaction is fraudulent.

## Backend Setup

```powershell
cd fraud-sentinel\backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload
```

The health endpoint is available at `http://localhost:8000/api/health`.

Run the backend test from the `backend` directory:

```powershell
python -m unittest discover -s tests
```

## Synthetic Transaction Dataset

Generate the default dataset from the `backend` directory:

```powershell
python -m app.data.synthetic_transactions --seed 42 --days 14 --baseline-count 20000 --output data/synthetic_transactions.csv
```

The command writes `backend/data/synthetic_transactions.csv` and prints validation counts plus sample records. The default contains 20,000 safe baseline transactions and 9 injected fraud transactions, for 20,009 rows across a 14-day UTC interval. Override `--seed`, `--days`, `--baseline-count`, `--output`, or `--sample-size` when invoking the CLI. The seed, period, and baseline size also have `SYNTHETIC_SEED`, `SYNTHETIC_DAYS`, and `SYNTHETIC_BASELINE_TRANSACTIONS` configuration defaults.

Each CSV row follows the Pydantic transaction schema in `backend/app/models/transaction.py`:

| Field | Meaning |
|---|---|
| `transaction_id` | Unique generated transaction identifier |
| `timestamp` | UTC transaction time in the generation window |
| `sender_account`, `receiver_account` | Generated account identifiers |
| `amount` | Positive transaction amount |
| `transaction_type` | Transfer, collect request, merchant payment, or bill payment |
| `device_id` | Generated device associated with the sender |
| `sender_account_created_at`, `receiver_account_created_at` | Account creation timestamps for age features |
| `first_time_payee` | Whether this directed account pair is new at that point in chronological history |
| `label` | Ground truth: `safe` or `fraud` |
| `fraud_pattern` | Nullable scenario label; null for safe baseline rows |

The generator keeps baseline creation, scenario injection, and dataset validation separate. The injected ground-truth scenarios are:

- One large collect request (`42,500`) to a first-time payee.
- A four-transfer rapid path through five accounts, with 35 seconds between transfers.
- A four-transfer directed cycle through four accounts, with 45 seconds between transfers.

Fraud labels are included for later supervised-model evaluation. These scenario counts are deliberately engineered test cases, not an estimate of real UPI fraud prevalence. Synthetic data is used because the project does not yet have an authorized transaction dataset; it enables data-contract, pipeline, and later model-evaluation work without exposing real customer data. Generated labels are ground truth only: this phase produces no model predictions, risk scores, or detection decisions.

The validation step checks transaction ID uniqueness, timestamps and amounts, account/device references, label consistency, minimum baseline size, and structural presence of all three injected patterns. It prints total, safe, fraud, and per-pattern counts and fails generation if an invariant is broken.

## Deterministic Rule Engine

The rule engine evaluates only observable transaction attributes and returns typed, per-transaction evidence. It does not combine signals into a risk score, classify a transaction as fraudulent, or call ML, graph analysis, or Claude. Source `label` and `fraud_pattern` values are explicitly projected away before evaluation; they remain ground truth for later evaluation only.

Initial rules:

- `LARGE_COLLECT_FIRST_TIME_PAYEE`: emits a high-severity signal for a collect request to a first-time payee when `amount >=` the configured threshold.
- `RAPID_TRANSFER_BEHAVIOUR`: finds a temporally ordered path of linked transfers (each receiver is the next sender) meeting the configured minimum count and time window. Evidence identifies all participating transaction IDs and the account path.
- `CIRCULAR_FLOW_PRECURSOR`: checks bounded simple transfer paths of 3 through the configured maximum number of transfers that return to the starting account within its time window. It is intentionally a small precursor matcher, not a general graph/cycle engine; broad network analysis remains a later NetworkX phase.

Default settings are `RULE_LARGE_COLLECT_AMOUNT_THRESHOLD=25000`, `RULE_RAPID_TRANSFER_WINDOW_SECONDS=300`, `RULE_RAPID_TRANSFER_MINIMUM_COUNT=3`, `RULE_CIRCULAR_FLOW_WINDOW_SECONDS=300`, and `RULE_CIRCULAR_FLOW_MAX_TRANSFERS=4`. Configure these in `backend/.env`; the same defaults are listed in `backend/.env.example`. Amount matching is inclusive, and transfer-window boundaries are inclusive.

Run the rules from the `backend` directory:

```powershell
python -m app.services.rule_engine --input data/synthetic_transactions.csv --output data/rule_evaluation.json
```

The input CSV is read-only. The machine-readable JSON report is written separately and contains the total evaluated, unique transactions with at least one signal, per-rule trigger counts and transaction IDs, and an evidence entry for every input transaction (empty `rule_signals` means no rule triggered). A signal includes its rule ID, severity, reason, and structured context. Example:

```json
{
  "transaction_id": "TX-00020001",
  "rule_signals": [
    {
      "rule_id": "LARGE_COLLECT_FIRST_TIME_PAYEE",
      "triggered": true,
      "severity": "high",
      "reason": "Large collect request involving a first-time payee",
      "evidence": {
        "amount": 42500,
        "amount_threshold": 25000,
        "transaction_type": "collect_request",
        "first_time_payee": true
      }
    }
  ]
}
```

Rule signals are deterministic observations, not labels. A triggered rule does not itself establish fraud, while ground-truth `label` and `fraud_pattern` are never used by the rule logic. In the synthetic data, the rapid mule-chain scenario is modeled as linked transfers and the circular scenario as a four-transfer closed path; these construction assumptions make the scenarios suitable for verifying the corresponding rules, not for estimating live-world performance.

## Isolation Forest Behavioural Signal

Isolation Forest is an unsupervised anomaly detector that isolates observations which differ from the fitted population. Fraud Sentinel uses it as a behavioural signal alongside deterministic rule evidence; an anomaly is not proof of fraud, and this service does not make a final fraud/safe decision.

The detector uses a fixed observation-only input model. It does not read `label`, `fraud_pattern`, or any scenario identity, and those columns are discarded by the CSV loader. Features are derived from existing transaction fields:

- Log-transformed positive amount (`log_amount`).
- Hour-of-day sine/cosine encoding (`hour_sin`, `hour_cos`) to preserve time-of-day wraparound.
- One-hot transaction type for transfer, collect request, merchant payment, and bill payment.
- First-time-payee indicator.
- Sender and receiver account age in days at the transaction timestamp.
- Sender outgoing and receiver incoming transaction counts in the preceding one-hour window, including the current transaction.

The report includes a normalized `anomaly_score` from 0 to 1. Higher means more unusual relative to this fitted batch. The score is a sigmoid transform of the model's decision-function distance, centered and scaled by the batch median and median absolute deviation; it is for ordering/inspection and is not a probability. `is_anomalous` is the Isolation Forest prediction using the configured contamination threshold.

Configuration lives in `backend/.env` and defaults are shown in `backend/.env.example`:

- `ISOLATION_FOREST_CONTAMINATION=0.01`
- `ISOLATION_FOREST_N_ESTIMATORS=200`
- `ISOLATION_FOREST_RANDOM_STATE=42`

The 1% contamination value is a demo setting for this synthetic dataset. It is not a real-world fraud prevalence estimate. The model is fit on the supplied transactions using observable features only; it does not use synthetic ground-truth labels to select training rows.

Run batch scoring from the `backend` directory:

```powershell
python -m app.services.isolation_forest --input data/synthetic_transactions.csv --output data/isolation_forest_evaluation.json
```

The command writes structured per-transaction evidence to a separate JSON artifact and leaves the source CSV untouched. The report includes model configuration, number scored, number marked anomalous, and feature values for each transaction. Results from this generated dataset are synthetic/demo observations only. No accuracy, precision, recall, F1, or production performance claim is made; an unsupervised anomaly signal may flag safe examples and miss fraud patterns, especially given the engineered and limited synthetic population.

## LightGBM Supervised Model Signal

LightGBM is used as a lightweight supervised binary classifier: it learns from the synthetic `label` target and emits a fraud probability/prediction signal. This is separate from the unsupervised Isolation Forest signal and deterministic Rule Engine evidence. It is not the final Fraud Sentinel fraud decision and is not combined with the other outputs in this phase.

The model reuses the Isolation Forest service's observation-only behavioral feature extractor. The exact feature columns are `log_amount`, `hour_sin`, `hour_cos`, `transaction_type_transfer`, `transaction_type_collect_request`, `transaction_type_merchant_payment`, `transaction_type_bill_payment`, `first_time_payee`, `sender_account_age_days`, `receiver_account_age_days`, `sender_outgoing_count_1h`, and `receiver_incoming_count_1h`. Transaction identifiers are retained only to associate predictions with rows. `label` is mapped to a separate target (`safe=0`, `fraud=1`) and is never in the feature matrix. `fraud_pattern` and scenario identifiers are not used as features or to select training rows.

The default split is a deterministic 80/20 stratified train/test split with `LIGHTGBM_RANDOM_STATE=42`. The model fits only on the training partition. `scale_pos_weight` is computed as the safe-to-fraud ratio in that training partition; rows are not duplicated or synthetically rebalanced. The default prediction threshold is explicitly `LIGHTGBM_THRESHOLD=0.5`. A probability at or above that threshold sets `predicted_fraud=true`; it remains a model signal, not proof of fraud.

Other lightweight demo defaults are `LIGHTGBM_N_ESTIMATORS=100`, `LIGHTGBM_LEARNING_RATE=0.05`, `LIGHTGBM_NUM_LEAVES=7`, `LIGHTGBM_MAX_DEPTH=4`, and `LIGHTGBM_MIN_CHILD_SAMPLES=5`. These settings and split/threshold controls can be overridden via `backend/.env`; `.env.example` lists them.

Run batch training/evaluation from the `backend` directory:

```powershell
python -m app.services.lightgbm_classifier --input data/synthetic_transactions.csv --output data/lightgbm_evaluation.json
```

The JSON report includes dataset/train/test sizes and class counts, effective configuration including `scale_pos_weight`, held-out precision/recall/F1, ROC-AUC and average precision when defined, confusion matrix, and a probability/prediction record for each row tagged with its train/test partition. **All reported metrics are calculated only on the stratified held-out test partition**; training-row predictions are in-sample evidence and are not used for those metrics.

The current generated dataset has only nine fraud-labeled rows, so an 80/20 stratified test split contains approximately two fraud cases. This makes the holdout metrics highly unstable and unsuitable for claims about accuracy or real-world fraud performance. The synthetic labels describe engineered examples, not observed prevalence; model performance must be reevaluated on a larger, representative, authorized dataset before drawing operational conclusions.

## Risk Engine and Unified Alerts

The Risk Engine is the only integration point that combines the four independent outputs: Rule Engine evidence, Isolation Forest evidence, LightGBM model evidence, and NetworkX findings. It consumes those typed reports by `transaction_id`; it does not rerun or alter detector logic. The score uses detector evidence only and does not read `label` or `fraud_pattern`.

The deterministic demo score is calculated as follows:

```text
rule_component = min(30, 15 per LARGE_COLLECT_FIRST_TIME_PAYEE
                          + 10 per RAPID_TRANSFER_BEHAVIOUR
                          + 10 per CIRCULAR_FLOW_PRECURSOR)

anomaly_component = 20 * anomaly_score, if is_anomalous; otherwise 0
ml_component = 30 * fraud_probability
network_component = min(20, 10 per rapid_mule_chain finding
                            + 10 per circular_flow finding
                            + 5 per high_connectivity_hub finding)

risk_score = clamp(rule_component + anomaly_component + ml_component
                   + network_component, 0, 100)
```

Each component is capped at its named maximum, for a maximum combined score of 100. The score is an uncalibrated demo decision-support score, not a probability and not proof of fraud. Risk levels use configurable demo boundaries: below `RISK_MEDIUM_THRESHOLD` is low; at or above medium and below `RISK_HIGH_THRESHOLD` is medium; at or above high is high. Defaults are `RISK_MEDIUM_THRESHOLD=30` and `RISK_HIGH_THRESHOLD=70`. “High” means investigation recommended, not confirmed fraud. Only transactions with `risk_score >= RISK_ALERT_THRESHOLD` (default 30) appear in the alert list. Complete risk/evidence records for all transactions remain in the artifact and can be retrieved through the transaction-evidence API even when below the alert cutoff.

The one-command integration runs the four existing detector services independently, joins their structured results, and writes all transaction evidence plus selected `open` alerts:

```powershell
python -m app.services.risk_engine --input data/synthetic_transactions.csv --output data/fraud_alerts.json
```

If the artifact is absent, FastAPI startup runs the same integration once and loads the resulting JSON into an in-memory store. Regenerate the artifact after changing input data or thresholds. Feedback changes are in-memory only and reset when the server restarts; no database is used. The alert status starts as `open`; only analyst feedback changes it to `confirmed_fraud` or `marked_safe`.

### Alert API

FastAPI documents the typed request and response models at [http://localhost:8000/docs](http://localhost:8000/docs).

- `GET /api/alerts?risk_level=high&status=open&limit=20` lists selected alerts and supports risk-level, status, and limit filters.
- `GET /api/alerts/{alert_id}` returns transaction details, risk components, independent detector evidence, deterministic explanation, and status.
- `GET /api/alerts/{alert_id}/graph` returns account nodes and transaction edges associated with that alert's network findings; it does not expose a NetworkX object.
- `GET /api/transactions/{transaction_id}/evidence` retrieves complete risk evidence for a transaction even when it is below the alert threshold.
- `POST /api/feedback` accepts `{"alert_id":"FS-TX-00020006","status":"marked_safe"}` or `confirmed_fraud` and updates runtime memory only.
- `GET /api/summary` returns selected alert totals by risk level and review status.

Thresholds and scores are for the synthetic hackathon demo, not calibrated operational decisions. The supervised model was trained/evaluated on a synthetic dataset with very few positive examples, and its training-partition model signals are in-sample. Do not interpret the combined score, a high-risk level, or an alert as a confirmed fraud determination.

## NetworkX Transaction-Network Evidence

The NetworkX layer analyzes relationships across transactions that are not visible from a single row. It produces independent structured network evidence only; it does not assign a final fraud verdict or risk score and does not combine its results with rules, Isolation Forest, or LightGBM.

The graph is a directed `DiGraph`: account IDs are nodes and each transaction adds an edge from `sender_account` to `receiver_account`. Repeated account pairs share a directed edge whose `transactions` attribute preserves every transaction's ID, timestamp, amount, and transaction type; edge-level transaction count and total amount are also retained. Graph construction reads the observation-only transaction schema, not `label` or `fraud_pattern`.

Three structures are reported:

- `rapid_mule_chain`: a time-ordered path of directed transfer transactions through multiple accounts, meeting configured hop and time-window bounds. Evidence includes transaction IDs, account path, hops, timestamps, elapsed seconds, and total amount.
- `circular_flow`: a directed cycle found with NetworkX `simple_cycles` bounded by the configured maximum cycle length. Candidate edges are restricted to overlapping short time windows and then checked for chronological order within the window. This is a structural signal, not proof of a fraud ring.
- `high_connectivity_hub`: an account exceeding both a total directed degree threshold and a minimum incident transaction count. Evidence includes in/out/total degree, transaction count, threshold values, and incident transaction volume. Connectivity alone does not imply fraud.

Demo defaults are `NETWORK_RAPID_PATH_WINDOW_SECONDS=300`, `NETWORK_RAPID_PATH_MINIMUM_HOPS=3`, `NETWORK_RAPID_PATH_MAXIMUM_HOPS=6`, `NETWORK_CYCLE_WINDOW_SECONDS=300`, `NETWORK_CYCLE_MAXIMUM_LENGTH=4`, `NETWORK_HUB_DEGREE_THRESHOLD=31`, and `NETWORK_HUB_MINIMUM_TRANSACTIONS=12`. Configure them in `backend/.env`; `.env.example` lists the defaults. The hub-degree default is selective for the current synthetic account graph and is not a general-purpose or production threshold.

Run the batch analysis from the `backend` directory:

```powershell
python -m app.services.network_analysis --input data/synthetic_transactions.csv --output data/network_evaluation.json
```

The separate JSON artifact reports graph node/edge/transaction counts, weakly connected component count, finding counts by pattern, and structured findings. The source CSV is not modified. NetworkX here is a local transaction-network analysis layer, not a production-scale graph database or a guaranteed fraud-ring detector. This hackathon/demo implementation operates on engineered synthetic data; its detected structures and thresholds must not be interpreted as real-world prevalence or proof that any connected account is fraudulent.

## Frontend Setup

```powershell
cd fraud-sentinel\frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

The frontend is available at `http://localhost:3000`. Its backend base URL is configured with `NEXT_PUBLIC_API_URL` in `frontend/.env.local`.

## Current Scope

The backend provides configuration, local-development CORS, a typed health endpoint, reproducible synthetic transaction generation, deterministic rule evidence, an observation-based Isolation Forest signal, a supervised LightGBM model signal, independent NetworkX transaction-network evidence, a demo Risk Engine, unified alerts, and alert APIs. The frontend remains a product shell. Claude/LLM integration, database, authentication, and deployment are not implemented. Claude remains planned as an optional explanation layer for structured evidence, never as the fraud decision-maker.