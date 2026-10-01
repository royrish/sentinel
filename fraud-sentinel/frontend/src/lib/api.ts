export type RiskLevel = "low" | "medium" | "high";
export type AlertStatus = "open" | "confirmed_fraud" | "marked_safe";
export type FeedbackStatus = Exclude<AlertStatus, "open">;
export type TransactionType =
  | "transfer"
  | "collect_request"
  | "merchant_payment"
  | "bill_payment";

export interface AlertSummary {
  total_alerts: number;
  high_risk: number;
  medium_risk: number;
  low_risk: number;
  open_alerts: number;
  confirmed_fraud: number;
  marked_safe: number;
}

export interface RuleSignal {
  rule_id: string;
  triggered: true;
  severity: "low" | "medium" | "high";
  reason: string;
  evidence: Record<string, string | number | boolean | string[] | null>;
}

export interface AnomalyEvidence {
  transaction_id: string;
  anomaly_score: number;
  is_anomalous: boolean;
  model_name: "IsolationForest";
  feature_values: Record<string, string | number | boolean>;
}

export interface MLEvidence {
  transaction_id: string;
  fraud_probability: number;
  predicted_fraud: boolean;
  model_name: "LightGBM";
  dataset_partition: "train" | "test";
}

export type NetworkMetric = number | string | string[];

export interface NetworkEvidence {
  pattern_type: "rapid_mule_chain" | "circular_flow" | "high_connectivity_hub";
  involved_accounts: string[];
  transaction_ids: string[];
  severity: "low" | "medium" | "high";
  reason: string;
  metrics: Record<string, NetworkMetric>;
}

export interface FraudAlert {
  alert_id: string;
  transaction_id: string;
  timestamp: string;
  sender_account: string;
  receiver_account: string;
  amount: number;
  transaction_type: TransactionType;
  risk_score: number;
  risk_level: RiskLevel;
  rule_component: number;
  anomaly_component: number;
  ml_component: number;
  network_component: number;
  rule_evidence: RuleSignal[];
  anomaly_evidence: AnomalyEvidence | null;
  ml_evidence: MLEvidence | null;
  network_evidence: NetworkEvidence[];
  explanation_summary: string;
  status: AlertStatus;
}

export interface AlertsResponse {
  total: number;
  items: FraudAlert[];
}

export interface AlertGraphNode {
  id: string;
  type: "account";
}

export interface AlertGraphEdge {
  source: string;
  target: string;
  transaction_id: string;
  amount: number;
  timestamp: string;
}

export interface AlertGraph {
  alert_id: string;
  nodes: AlertGraphNode[];
  edges: AlertGraphEdge[];
}

export type TransactionRiskEvidence = Omit<FraudAlert, "status">;

export interface FeedbackResponse {
  alert_id: string;
  status: FeedbackStatus;
}

export interface AlertsQuery {
  risk_level?: RiskLevel;
  status?: AlertStatus;
  limit?: number;
}

export interface HealthResponse {
  status: "ok";
  service: "fraud-sentinel-backend";
}

async function apiRequest<T>(path: string, init?: RequestInit): Promise<T> {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
  let response: Response;

  try {
    response = await fetch(`${baseUrl.replace(/\/$/, "")}${path}`, {
      ...init,
      cache: "no-store",
      headers: {
        Accept: "application/json",
        ...(init?.body ? { "Content-Type": "application/json" } : {}),
        ...init?.headers,
      },
    });
  } catch {
    throw new Error("Fraud Sentinel API unavailable. Make sure FastAPI is running on localhost:8000.");
  }

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`API request failed (${response.status})${detail ? `: ${detail}` : ""}`);
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return apiRequest<HealthResponse>("/api/health");
}

export function getSummary(): Promise<AlertSummary> {
  return apiRequest<AlertSummary>("/api/summary");
}

export function getAlerts(query: AlertsQuery = {}): Promise<AlertsResponse> {
  const params = new URLSearchParams();
  if (query.risk_level) params.set("risk_level", query.risk_level);
  if (query.status) params.set("status", query.status);
  params.set("limit", String(query.limit ?? 1000));
  return apiRequest<AlertsResponse>(`/api/alerts?${params.toString()}`);
}

export function getAlert(alertId: string): Promise<FraudAlert> {
  return apiRequest<FraudAlert>(`/api/alerts/${encodeURIComponent(alertId)}`);
}

export function getAlertGraph(alertId: string): Promise<AlertGraph> {
  return apiRequest<AlertGraph>(`/api/alerts/${encodeURIComponent(alertId)}/graph`);
}

export function submitFeedback(
  alertId: string,
  status: FeedbackStatus,
): Promise<FeedbackResponse> {
  return apiRequest<FeedbackResponse>("/api/feedback", {
    method: "POST",
    body: JSON.stringify({ alert_id: alertId, status }),
  });
}

export function getTransactionEvidence(
  transactionId: string,
): Promise<TransactionRiskEvidence> {
  return apiRequest<TransactionRiskEvidence>(
    `/api/transactions/${encodeURIComponent(transactionId)}/evidence`,
  );
}