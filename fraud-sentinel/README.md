# Fraud Sentinel

Fraud Sentinel is an explainable UPI fraud-detection and investigation platform. The frontend and backend are separate applications. The current backend phase provides a reproducible synthetic transaction dataset; it does not provide fraud detection or analytics.

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

## Frontend Setup

```powershell
cd fraud-sentinel\frontend
npm install
Copy-Item .env.example .env.local
npm run dev
```

The frontend is available at `http://localhost:3000`. Its backend base URL is configured with `NEXT_PUBLIC_API_URL` in `frontend/.env.local`.

## Current Scope

The backend provides configuration, local-development CORS, a typed health endpoint, and reproducible synthetic transaction generation. The frontend provides product navigation and an empty Command Center state. Detection rules, Isolation Forest, LightGBM training or inference, graph analysis, risk scoring, Claude integration, fraud analytics, database, and authentication are not implemented. Claude remains planned as an explanation layer for structured evidence, never as the fraud decision-maker.