"use client";

import {
  Activity,
  AlertTriangle,
  ArrowDownRight,
  ArrowRight,
  ArrowUpRight,
  BadgeCheck,
  BellRing,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Command,
  Eye,
  Filter,
  Gauge,
  GitBranch,
  LoaderCircle,
  Network,
  RefreshCw,
  Search,
  Shield,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Waypoints,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";

import {
  getAlert,
  getAlertExplanation,
  getAlertGraph,
  getAlerts,
  getSummary,
  getTransactionEvidence,
  submitFeedback,
  type AlertGraph,
  type AlertStatus,
  type AlertSummary,
  type AnalystExplanation,
  type FeedbackStatus,
  type FraudAlert,
  type NetworkEvidence,
  type RiskLevel,
  type TransactionRiskEvidence,
} from "@/lib/api";

type RiskFilter = RiskLevel | "all";
type StatusFilter = AlertStatus | "all";
type LoadState = "loading" | "ready" | "error";

const moneyFormatter = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
});

const dateFormatter = new Intl.DateTimeFormat("en-IN", {
  day: "2-digit",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

function formatAmount(amount: number): string {
  return moneyFormatter.format(amount);
}

function formatDate(value: string): string {
  return dateFormatter.format(new Date(value));
}

function riskTitle(level: RiskLevel): string {
  return `${level[0].toUpperCase()}${level.slice(1)} Risk`;
}

function statusTitle(status: AlertStatus): string {
  if (status === "confirmed_fraud") return "Confirmed fraud";
  if (status === "marked_safe") return "Marked safe";
  return "Open";
}

function transactionTitle(type: FraudAlert["transaction_type"]): string {
  return type.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function evidenceLabel(value: string): string {
  return value
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatMetric(value: string | number | string[]): string {
  if (Array.isArray(value)) return value.join(" → ");
  if (typeof value === "number") return value.toLocaleString("en-IN");
  return value;
}

function componentWidth(value: number, maximum: number): string {
  return `${Math.min(100, Math.max(0, (value / maximum) * 100))}%`;
}

function formatPoints(value: number): string {
  return new Intl.NumberFormat("en-IN", { maximumFractionDigits: 1 }).format(value);
}

function StatCard({
  label,
  value,
  icon: Icon,
  tone,
  note,
}: {
  label: string;
  value: number | null;
  icon: typeof BellRing;
  tone: "neutral" | "high" | "medium" | "safe";
  note?: string;
}) {
  return (
    <article className={`stat-card stat-${tone}`}>
      <div className="stat-topline">
        <span>{label}</span>
        <Icon size={16} strokeWidth={1.8} aria-hidden="true" />
      </div>
      <div className="stat-value">{value === null ? <span className="skeleton skeleton-number" /> : value.toLocaleString("en-IN")}</div>
      {note && <div className="stat-note">{note}</div>}
    </article>
  );
}

function NetworkDiagram({
  graph,
  selectedTransactionId,
  onSelectTransaction,
}: {
  graph: AlertGraph;
  selectedTransactionId: string;
  onSelectTransaction: (transactionId: string) => void;
}) {
  const width = 840;
  const height = 390;
  const centerX = width / 2;
  const centerY = height / 2;
  const radiusX = Math.min(width * 0.39, 325);
  const radiusY = Math.min(height * 0.37, 145);
  const positions = new Map(
    graph.nodes.map((node, index) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * index) / Math.max(graph.nodes.length, 1);
      return [node.id, { x: centerX + radiusX * Math.cos(angle), y: centerY + radiusY * Math.sin(angle) }];
    }),
  );

  return (
    <div className="network-canvas">
      <svg className="network-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Directed transaction account network">
        <defs>
          <marker id="transaction-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="strokeWidth">
            <path d="M 0 0 L 8 4 L 0 8 z" fill="context-stroke" />
          </marker>
        </defs>
        <g className="graph-grid" aria-hidden="true">
          <circle cx={centerX} cy={centerY} r="92" />
          <circle cx={centerX} cy={centerY} r="176" />
          <line x1="32" y1={centerY} x2={width - 32} y2={centerY} />
        </g>
        {graph.edges.map((edge) => {
          const source = positions.get(edge.source);
          const target = positions.get(edge.target);
          if (!source || !target) return null;
          const dx = target.x - source.x;
          const dy = target.y - source.y;
          const length = Math.max(Math.hypot(dx, dy), 1);
          const ux = dx / length;
          const uy = dy / length;
          const startX = source.x + ux * 23;
          const startY = source.y + uy * 23;
          const endX = target.x - ux * 28;
          const endY = target.y - uy * 28;
          const selected = edge.transaction_id === selectedTransactionId;
          return (
            <g
              className={`graph-edge${selected ? " graph-edge-selected" : ""}`}
              key={edge.transaction_id}
              role="button"
              tabIndex={0}
              aria-label={`${edge.source} to ${edge.target}, ${formatAmount(edge.amount)}, transaction ${edge.transaction_id}`}
              onClick={() => onSelectTransaction(edge.transaction_id)}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  onSelectTransaction(edge.transaction_id);
                }
              }}
            >
              <title>{`${edge.source} → ${edge.target} · ${formatAmount(edge.amount)} · ${formatDate(edge.timestamp)}`}</title>
              <line x1={startX} y1={startY} x2={endX} y2={endY} markerEnd="url(#transaction-arrow)" />
              {selected && (
                <text x={(startX + endX) / 2} y={(startY + endY) / 2 - 8} className="graph-edge-label">
                  {formatAmount(edge.amount)}
                </text>
              )}
            </g>
          );
        })}
        {graph.nodes.map((node) => {
          const point = positions.get(node.id);
          if (!point) return null;
          const isInSelected = graph.edges.some(
            (edge) => edge.transaction_id === selectedTransactionId && (edge.source === node.id || edge.target === node.id),
          );
          return (
            <g className={`graph-node${isInSelected ? " graph-node-selected" : ""}`} key={node.id}>
              <title>{node.id}</title>
              <circle cx={point.x} cy={point.y} r="21" />
              <text x={point.x} y={point.y + 4}>{node.id.slice(-4)}</text>
            </g>
          );
        })}
      </svg>
      <div className="graph-legend">
        <span><i className="legend-account" /> Account</span>
        <span><i className="legend-edge" /> Transaction direction</span>
        <span><i className="legend-focus" /> Selected transaction</span>
        <span className="graph-hint">Select an edge to inspect its transaction</span>
      </div>
    </div>
  );
}

function NetworkFindingCard({ finding }: { finding: NetworkEvidence }) {
  const title = {
    rapid_mule_chain: "Rapid multi-account path",
    circular_flow: "Circular money flow",
    high_connectivity_hub: "High-connectivity account",
  }[finding.pattern_type];
  const accountPath = finding.metrics.account_cycle ?? finding.metrics.account_path;
  const extraMetrics = Object.entries(finding.metrics).filter(
    ([key]) => !["account_cycle", "account_path", "timestamps", "configured_window_seconds"].includes(key),
  );

  return (
    <article className="network-finding">
      <div className="finding-heading">
        <div className="finding-type-icon"><GitBranch size={15} /></div>
        <div className="finding-title-wrap">
          <strong>{title}</strong>
          <span className={`severity severity-${finding.severity}`}>{finding.severity} signal</span>
        </div>
      </div>
      <p className="finding-reason">{finding.reason}</p>
      {Array.isArray(accountPath) && (
        <div className="account-path" aria-label="Account sequence">
          {accountPath.map((account, index) => (
            <span className="account-step" key={`${account}-${index}`}>
              {index > 0 && <ArrowRight size={13} aria-hidden="true" />}
              <code>{account}</code>
            </span>
          ))}
        </div>
      )}
      <div className="finding-metrics">
        {extraMetrics.map(([key, value]) => (
          <div className="finding-metric" key={key}>
            <span>{evidenceLabel(key)}</span>
            <strong>{key.includes("amount") || key.includes("volume") ? formatAmount(Number(value)) : formatMetric(value)}</strong>
          </div>
        ))}
      </div>
      <div className="transaction-chips" aria-label="Related transactions">
        {finding.transaction_ids.map((transactionId) => (
          <span className="transaction-chip" key={transactionId}>{transactionId}</span>
        ))}
      </div>
    </article>
  );
}

export default function Home() {
  const [summary, setSummary] = useState<AlertSummary | null>(null);
  const [alerts, setAlerts] = useState<FraudAlert[]>([]);
  const [selectedAlertId, setSelectedAlertId] = useState<string | null>(null);
  const [detail, setDetail] = useState<FraudAlert | null>(null);
  const [graph, setGraph] = useState<AlertGraph | null>(null);
  const [relatedEvidence, setRelatedEvidence] = useState<TransactionRiskEvidence | null>(null);
  const [selectedGraphTransactionId, setSelectedGraphTransactionId] = useState("");
  const [riskFilter, setRiskFilter] = useState<RiskFilter>("all");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [loadState, setLoadState] = useState<LoadState>("loading");
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [analystExplanation, setAnalystExplanation] = useState<AnalystExplanation | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState("");
  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [feedbackPending, setFeedbackPending] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    let current = true;
    setLoadState("loading");
    setError("");
    const query = {
      ...(riskFilter !== "all" ? { risk_level: riskFilter } : {}),
      ...(statusFilter !== "all" ? { status: statusFilter } : {}),
      limit: 1000,
    };

    Promise.all([getSummary(), getAlerts(query)])
      .then(([nextSummary, response]) => {
        if (!current) return;
        setSummary(nextSummary);
        setAlerts(response.items);
        setSelectedAlertId((existing) => {
          if (existing && response.items.some((alert) => alert.alert_id === existing)) return existing;
          const preferred = response.items.find((alert) => alert.risk_level === "high") ?? response.items[0];
          return preferred?.alert_id ?? null;
        });
        setLoadState("ready");
      })
      .catch((requestError: unknown) => {
        if (!current) return;
        setError(requestError instanceof Error ? requestError.message : "Unable to load analyst data.");
        setLoadState("error");
      });

    return () => {
      current = false;
    };
  }, [riskFilter, statusFilter]);

  useEffect(() => {
    if (!selectedAlertId) {
      setDetail(null);
      setGraph(null);
      return;
    }
    let current = true;
    setDetailLoading(true);
    setDetailError("");
    setAnalystExplanation(null);
    setExplanationError("");
    setFeedbackMessage("");
    setRelatedEvidence(null);
    setSelectedGraphTransactionId("");

    Promise.allSettled([getAlert(selectedAlertId), getAlertGraph(selectedAlertId)]).then(([alertResult, graphResult]) => {
      if (!current) return;
      if (alertResult.status === "fulfilled") setDetail(alertResult.value);
      else setDetailError(alertResult.reason instanceof Error ? alertResult.reason.message : "Unable to load alert detail.");
      if (graphResult.status === "fulfilled") setGraph(graphResult.value);
      else setGraph(null);
      setDetailLoading(false);
    });

    return () => {
      current = false;
    };
  }, [selectedAlertId]);

  async function refreshDashboard(): Promise<void> {
    setRefreshing(true);
    try {
      const query = {
        ...(riskFilter !== "all" ? { risk_level: riskFilter } : {}),
        ...(statusFilter !== "all" ? { status: statusFilter } : {}),
        limit: 1000,
      };
      const [nextSummary, response] = await Promise.all([getSummary(), getAlerts(query)]);
      setSummary(nextSummary);
      setAlerts(response.items);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to refresh dashboard data.");
    } finally {
      setRefreshing(false);
    }
  }

  async function handleFeedback(status: FeedbackStatus): Promise<void> {
    if (!detail) return;
    setFeedbackPending(true);
    setFeedbackMessage("");
    try {
      const result = await submitFeedback(detail.alert_id, status);
      setDetail((current) => current ? { ...current, status: result.status } : current);
      setAlerts((current) => current.map((alert) => alert.alert_id === result.alert_id ? { ...alert, status: result.status } : alert));
      setFeedbackMessage(`Analyst feedback recorded: ${statusTitle(result.status)}. This demo status resets when the backend restarts.`);
      const [nextSummary, response] = await Promise.all([
        getSummary(),
        getAlerts({
          ...(riskFilter !== "all" ? { risk_level: riskFilter } : {}),
          ...(statusFilter !== "all" ? { status: statusFilter } : {}),
          limit: 1000,
        }),
      ]);
      setSummary(nextSummary);
      setAlerts(response.items);
      if (!response.items.some((alert) => alert.alert_id === result.alert_id)) {
        setSelectedAlertId(response.items[0]?.alert_id ?? null);
      }
    } catch (requestError) {
      setFeedbackMessage(requestError instanceof Error ? requestError.message : "Unable to record analyst feedback.");
    } finally {
      setFeedbackPending(false);
    }
  }

  async function inspectTransaction(transactionId: string): Promise<void> {
    setSelectedGraphTransactionId(transactionId);
    setRelatedEvidence(null);
    try {
      setRelatedEvidence(await getTransactionEvidence(transactionId));
    } catch {
      setDetailError("Transaction evidence could not be loaded.");
    }
  }

  async function generateAnalystExplanation(): Promise<void> {
    if (!detail) return;
    setExplanationLoading(true);
    setExplanationError("");
    try {
      setAnalystExplanation(await getAlertExplanation(detail.alert_id));
    } catch (requestError) {
      setExplanationError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to retrieve an analyst explanation.",
      );
    } finally {
      setExplanationLoading(false);
    }
  }

  return (
    <main className="console-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" aria-label="Fraud Sentinel overview">
          <span className="brand-mark"><Shield size={20} strokeWidth={1.8} /></span>
          <span className="brand-copy"><strong>Fraud Sentinel</strong><small>INTELLIGENCE CONSOLE</small></span>
        </a>

        <div className="nav-caption">WORKSPACE</div>
        <nav className="primary-nav" aria-label="Workspace navigation">
          <a className="nav-link nav-link-active" href="#overview"><Command size={17} />Overview</a>
          <a className="nav-link" href="#alerts"><BellRing size={17} />Alerts{summary && <span className="nav-count">{summary.total_alerts}</span>}</a>
          <a className="nav-link" href="#network"><Waypoints size={17} />Network</a>
        </nav>

        <div className="sidebar-lower">
          <div className="nav-caption">SYSTEM</div>
          <div className="nav-link nav-link-disabled" aria-disabled="true"><SlidersHorizontal size={17} />Settings<span className="soon-tag">SOON</span></div>
          <div className="sidebar-operator">
            <div className="operator-avatar">FA</div>
            <div><strong>Fraud Analyst</strong><span>Investigation desk</span></div>
            <CircleHelp size={16} className="operator-help" />
          </div>
        </div>
      </aside>

      <section className="main-column">
        <header className="topbar">
          <div className="crumbs"><span>Operations</span><span className="crumb-slash">/</span><strong>Fraud intelligence</strong></div>
          <div className={`api-indicator api-${loadState}`} role="status" aria-live="polite">
            <span className="api-light" />
            {loadState === "loading" ? "Connecting to API" : loadState === "error" ? "API unavailable" : "Live API connected"}
          </div>
        </header>

        <div className="dashboard-content">
          <section className="page-intro" id="overview">
            <div>
              <div className="eyebrow"><span className="eyebrow-mark" /> FRAUD OPERATIONS <span className="eyebrow-divider">/</span> COMMAND CENTER</div>
              <h1>Fraud intelligence</h1>
              <p>Review independent signals, investigate transaction relationships, and record analyst feedback.</p>
            </div>
            <div className="intro-actions">
              <div className="feed-indicator"><span /> SYNTHETIC DEMO FEED</div>
              <button className="icon-button" type="button" onClick={() => void refreshDashboard()} disabled={refreshing} aria-label="Refresh dashboard" title="Refresh dashboard">
                <RefreshCw size={16} className={refreshing ? "spin" : ""} />
              </button>
            </div>
          </section>

          {error && (
            <section className="api-error" role="alert">
              <AlertTriangle size={19} />
              <div><strong>Fraud Sentinel API unavailable</strong><span>{error} Make sure FastAPI is running on localhost:8000.</span></div>
              <button type="button" onClick={() => void refreshDashboard()}>Retry</button>
            </section>
          )}

          <section className="stats-grid" aria-label="Alert summary">
            <StatCard label="Total alerts" value={summary?.total_alerts ?? (loadState === "loading" ? null : 0)} icon={BellRing} tone="neutral" note="Selected for review" />
            <StatCard label="High risk" value={summary?.high_risk ?? (loadState === "loading" ? null : 0)} icon={ShieldAlert} tone="high" note="Investigation recommended" />
            <StatCard label="Medium risk" value={summary?.medium_risk ?? (loadState === "loading" ? null : 0)} icon={Activity} tone="medium" note="Review evidence" />
            <StatCard label="Low risk" value={summary?.low_risk ?? (loadState === "loading" ? null : 0)} icon={Eye} tone="neutral" note="Routine monitoring" />
            <StatCard label="Open" value={summary?.open_alerts ?? (loadState === "loading" ? null : 0)} icon={Clock3} tone="neutral" />
            <StatCard label="Confirmed fraud" value={summary?.confirmed_fraud ?? (loadState === "loading" ? null : 0)} icon={BadgeCheck} tone="high" />
            <StatCard label="Marked safe" value={summary?.marked_safe ?? (loadState === "loading" ? null : 0)} icon={Check} tone="safe" />
          </section>

          <section className="workbench" id="alerts">
            <aside className="alert-panel panel">
              <div className="panel-heading alert-panel-heading">
                <div><div className="section-kicker">QUEUE <span className="live-pip" /></div><h2>Alerts <span className="heading-count">{alerts.length.toLocaleString("en-IN")}</span></h2></div>
                <button className="icon-button small" type="button" onClick={() => void refreshDashboard()} aria-label="Refresh alerts" title="Refresh alerts"><RefreshCw size={15} /></button>
              </div>

              <div className="filter-row">
                <label className="filter-control"><Filter size={14} /><select aria-label="Filter by risk" value={riskFilter} onChange={(event) => setRiskFilter(event.target.value as RiskFilter)}>
                  <option value="all">All risk</option><option value="high">High risk</option><option value="medium">Medium risk</option><option value="low">Low risk</option>
                </select><ChevronDown size={13} /></label>
                <label className="filter-control"><select aria-label="Filter by status" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value as StatusFilter)}>
                  <option value="all">All status</option><option value="open">Open</option><option value="confirmed_fraud">Confirmed fraud</option><option value="marked_safe">Marked safe</option>
                </select><ChevronDown size={13} /></label>
              </div>

              <div className="alert-table-head"><span>TRANSACTION / TIME</span><span>RISK</span><span>STATUS</span></div>
              <div className="alert-list" aria-label="Fraud alerts">
                {loadState === "loading" && <div className="list-state"><LoaderCircle className="spin" size={18} /> Loading alert queue…</div>}
                {loadState === "ready" && alerts.length === 0 && <div className="list-state empty-list"><Search size={21} /><strong>No alerts in this view</strong><span>Try another risk or status filter.</span></div>}
                {alerts.map((alert) => (
                  <button
                    className={`alert-row${selectedAlertId === alert.alert_id ? " alert-row-selected" : ""}`}
                    key={alert.alert_id}
                    type="button"
                    onClick={() => setSelectedAlertId(alert.alert_id)}
                    aria-pressed={selectedAlertId === alert.alert_id}
                  >
                    <span className={`row-priority priority-${alert.risk_level}`} />
                    <span className="alert-row-main">
                      <span className="row-id">{alert.transaction_id}</span>
                      <span className="row-subline"><span>{formatAmount(alert.amount)}</span><i />{formatDate(alert.timestamp)}</span>
                      <span className="row-type">{transactionTitle(alert.transaction_type)}</span>
                    </span>
                    <span className="alert-row-risk"><strong>{alert.risk_score}</strong><span className={`risk-pill risk-${alert.risk_level}`}>{riskTitle(alert.risk_level)}</span></span>
                    <span className={`status-pill status-${alert.status}`}>{statusTitle(alert.status)}</span>
                    <ArrowUpRight size={14} className="row-open" />
                  </button>
                ))}
              </div>
              <div className="queue-foot"><span>Showing {alerts.length.toLocaleString("en-IN")} alerts</span><span>Sorted by risk score <ArrowDownRight size={12} /></span></div>
            </aside>

            <section className="investigation-column" aria-label="Alert investigation">
              {detailLoading && <div className="panel detail-loading"><LoaderCircle size={22} className="spin" /><span>Loading investigation evidence…</span></div>}
              {!detailLoading && detailError && <div className="panel detail-error" role="alert"><AlertTriangle size={19} /><span>{detailError}</span></div>}
              {!detailLoading && !detailError && !detail && <div className="panel detail-empty"><Shield size={30} /><strong>Select an alert to investigate</strong><span>Transaction, model, rule and network evidence will appear here.</span></div>}

              {!detailLoading && detail && (
                <>
                  {(() => {
                    const contributingLayers = [
                      detail.rule_evidence.length > 0,
                      detail.anomaly_evidence?.is_anomalous ?? false,
                      detail.ml_evidence?.predicted_fraud ?? false,
                      detail.network_evidence.length > 0,
                    ].filter(Boolean).length;
                    return contributingLayers > 1 ? (
                      <div className="signal-convergence"><Sparkles size={14} /><span><strong>{contributingLayers} independent signal layers contributed to this alert.</strong> Each layer remains separate evidence; the displayed score is from the backend Risk Engine.</span></div>
                    ) : null;
                  })()}
                  <article className="panel investigation-panel">
                    <div className="panel-heading investigation-heading">
                      <div><div className="section-kicker">INVESTIGATION <span className="heading-slash">/</span> {detail.alert_id}</div><h2>Alert detail</h2></div>
                      <div className="detail-heading-right"><span className={`risk-pill risk-${detail.risk_level}`}>{riskTitle(detail.risk_level)}</span><button className="icon-button small" type="button" onClick={() => void refreshDashboard()} aria-label="Refresh alert"><RefreshCw size={15} /></button></div>
                    </div>

                    <div className="risk-hero">
                      <div className={`score-orbit orbit-${detail.risk_level}`}><span className="score-caption">RISK</span><strong>{Math.round(detail.risk_score)}</strong><span className="score-out-of">/ 100</span></div>
                      <div className="risk-hero-copy">
                        <div className="score-level">{riskTitle(detail.risk_level)} <span>·</span> {detail.risk_level === "high" ? "Investigation recommended" : detail.risk_level === "medium" ? "Review supporting evidence" : "Routine monitoring"}</div>
                        <p>{detail.explanation_summary}</p>
                        <div className="component-breakdown" aria-label="Backend risk score components">
                          {([
                            ["Rules", detail.rule_component, 30, "component-rules"],
                            ["Behaviour", detail.anomaly_component, 20, "component-behaviour"],
                            ["ML model", detail.ml_component, 30, "component-ml"],
                            ["Network", detail.network_component, 20, "component-network"],
                          ] as const).map(([label, value, maximum, style]) => (
                            <div className="component-item" key={label}>
                              <div><span>{label}</span><strong>{formatPoints(value)} <small>/ {maximum}</small></strong></div>
                              <div className="component-track"><i className={style} style={{ width: componentWidth(value, maximum) }} /></div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="section-divider" />
                    <div className="transaction-head"><div className="section-kicker">TRANSACTION</div><span className={`status-pill status-${detail.status}`}>{statusTitle(detail.status)}</span></div>
                    <div className="transaction-grid">
                      <div className="transaction-fact fact-amount"><span>Amount</span><strong>{formatAmount(detail.amount)}</strong></div>
                      <div className="transaction-fact"><span>Transaction ID</span><strong className="mono">{detail.transaction_id}</strong></div>
                      <div className="transaction-fact"><span>Timestamp</span><strong>{formatDate(detail.timestamp)}</strong></div>
                      <div className="transaction-fact"><span>Type</span><strong>{transactionTitle(detail.transaction_type)}</strong></div>
                      <div className="transaction-fact"><span>Sender account</span><strong className="mono">{detail.sender_account}</strong></div>
                      <div className="transaction-fact"><span>Receiver account</span><strong className="mono">{detail.receiver_account}</strong></div>
                    </div>

                    <div className="analyst-actions">
                      <div className="feedback-copy"><span className="section-kicker">ANALYST FEEDBACK</span><span>Record a human review. This does not alter the model score.</span></div>
                      <div className="feedback-buttons">
                        <button className="feedback-button button-confirm" type="button" disabled={feedbackPending || detail.status === "confirmed_fraud"} onClick={() => void handleFeedback("confirmed_fraud")}><BadgeCheck size={15} /> Confirm Fraud</button>
                        <button className="feedback-button button-safe" type="button" disabled={feedbackPending || detail.status === "marked_safe"} onClick={() => void handleFeedback("marked_safe")}><Check size={15} /> Mark Safe</button>
                      </div>
                    </div>
                    {feedbackMessage && <div className={`feedback-message${feedbackMessage.includes("Unable") || feedbackMessage.includes("API") ? " feedback-error" : ""}`} role="status">{feedbackMessage}</div>}
                  </article>

                  <section className="panel analyst-explanation" aria-labelledby="analyst-explanation-title">
                    <div className="explanation-heading">
                      <div className="explanation-heading-copy">
                        <span className="evidence-icon explanation-icon"><Sparkles size={16} /></span>
                        <div><div className="section-kicker">OPTIONAL EXPLANATION</div><h2 id="analyst-explanation-title">AI Analyst Explanation</h2></div>
                      </div>
                      <div className="explanation-heading-actions">
                        {analystExplanation && <span className={`source-badge source-${analystExplanation.source}`}>{analystExplanation.source === "llm" ? "AI" : "Deterministic"}</span>}
                        <button className="explanation-generate" type="button" onClick={() => void generateAnalystExplanation()} disabled={explanationLoading}>
                          {explanationLoading ? <LoaderCircle size={14} className="spin" /> : <Sparkles size={14} />}
                          {explanationLoading ? "Generating…" : analystExplanation ? "Regenerate" : "Generate Explanation"}
                        </button>
                      </div>
                    </div>
                    <p className="explanation-disclaimer">Explanation only. The detector evidence and backend risk score remain authoritative; no fraud decision is made here.</p>
                    {explanationError && <div className="explanation-error" role="alert"><AlertTriangle size={14} />{explanationError}</div>}
                    {!analystExplanation && !explanationLoading && !explanationError && <div className="explanation-placeholder">Request a concise explanation grounded in this alert’s structured evidence.</div>}
                    {explanationLoading && <div className="explanation-placeholder"><LoaderCircle size={15} className="spin" /> Preparing an analyst explanation from the supplied evidence…</div>}
                    {analystExplanation && <div className="explanation-content">
                      <p className="explanation-summary">{analystExplanation.summary}</p>
                      <div className="explanation-columns">
                        <div className="explanation-list"><h3>Why this alert was surfaced</h3>{analystExplanation.why_flagged.length ? <ul>{analystExplanation.why_flagged.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul> : <p>No specific trigger was returned.</p>}</div>
                        <div className="explanation-list"><h3>Key evidence</h3>{analystExplanation.key_evidence.length ? <ul>{analystExplanation.key_evidence.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul> : <p>No key evidence was returned.</p>}</div>
                      </div>
                      <div className="analyst-action"><span>Suggested analyst action</span><p>{analystExplanation.analyst_action}</p></div>
                    </div>}
                  </section>

                  <section className="evidence-section" aria-labelledby="evidence-title">
                    <div className="section-heading"><div><div className="section-kicker">INDEPENDENT SIGNALS</div><h2 id="evidence-title">Evidence breakdown</h2></div><span className="independent-tag"><Sparkles size={13} /> Separate detector outputs</span></div>
                    <div className="evidence-grid">
                      <article className="panel evidence-card rule-card">
                        <div className="evidence-card-heading"><span className="evidence-icon rule-icon"><ShieldAlert size={16} /></span><div><h3>Rule evidence</h3><span>Deterministic signals</span></div><span className="evidence-count">{detail.rule_evidence.length}</span></div>
                        {detail.rule_evidence.length === 0 ? <div className="evidence-empty">No rule signals contributed to this alert.</div> : <div className="rule-list">
                          {detail.rule_evidence.map((signal, index) => (
                            <div className="rule-item" key={`${signal.rule_id}-${index}`}>
                              <div className="rule-item-top"><strong>{evidenceLabel(signal.rule_id)}</strong><span className={`severity severity-${signal.severity}`}>{signal.severity}</span></div>
                              <p>{signal.reason}</p>
                              <div className="rule-values">{Object.entries(signal.evidence).map(([key, value]) => <span key={key}><small>{evidenceLabel(key)}</small><strong>{typeof value === "number" && key.includes("amount") ? formatAmount(value) : Array.isArray(value) ? value.join(" → ") : String(value ?? "—")}</strong></span>)}</div>
                            </div>
                          ))}
                        </div>}
                      </article>

                      <article className="panel evidence-card anomaly-card">
                        <div className="evidence-card-heading"><span className="evidence-icon anomaly-icon"><Activity size={16} /></span><div><h3>Behavioural anomaly</h3><span>Isolation Forest signal</span></div><span className={`signal-state${detail.anomaly_evidence?.is_anomalous ? " signal-alert" : ""}`}>{detail.anomaly_evidence ? detail.anomaly_evidence.is_anomalous ? "Anomalous" : "Normal" : "No signal"}</span></div>
                        {!detail.anomaly_evidence ? <div className="evidence-empty">No anomaly evidence returned.</div> : <>
                          <div className="signal-score-row"><strong>{detail.anomaly_evidence.anomaly_score.toFixed(3)}</strong><span>anomaly score <i>· higher is more unusual</i></span></div>
                          <div className="feature-grid">{Object.entries(detail.anomaly_evidence.feature_values).filter(([key]) => ["log_amount", "first_time_payee", "sender_account_age_days", "receiver_account_age_days", "sender_outgoing_count_1h", "receiver_incoming_count_1h"].includes(key)).map(([key, value]) => <div className="feature-cell" key={key}><span>{evidenceLabel(key)}</span><strong>{typeof value === "number" && key.includes("age") ? `${value.toFixed(0)} days` : String(value)}</strong></div>)}</div>
                        </>}
                        <div className="model-disclaimer">Behavioural anomaly signal · not proof of fraud</div>
                      </article>

                      <article className="panel evidence-card ml-card">
                        <div className="evidence-card-heading"><span className="evidence-icon ml-icon"><Gauge size={16} /></span><div><h3>ML fraud probability</h3><span>LightGBM supervised signal</span></div><span className="evidence-count">{detail.ml_evidence?.dataset_partition === "test" ? "HOLDOUT" : detail.ml_evidence?.dataset_partition === "train" ? "TRAIN" : "—"}</span></div>
                        {!detail.ml_evidence ? <div className="evidence-empty">No model prediction returned.</div> : <>
                          <div className="probability-row"><strong>{(detail.ml_evidence.fraud_probability * 100).toFixed(1)}<small>%</small></strong><div className="probability-track"><i style={{ width: `${detail.ml_evidence.fraud_probability * 100}%` }} /></div></div>
                          <div className="ml-result"><span>Model output</span><strong className={detail.ml_evidence.predicted_fraud ? "ml-predicted" : ""}>{detail.ml_evidence.predicted_fraud ? "Predicted fraud" : "Predicted safe"}</strong></div>
                          <div className="model-disclaimer">Model probability is a signal, not a fraud determination.</div>
                        </>}
                      </article>

                      <article className="panel evidence-card network-card" id="network">
                        <div className="evidence-card-heading"><span className="evidence-icon network-icon"><Network size={16} /></span><div><h3>Network evidence</h3><span>NetworkX relationship signals</span></div><span className="evidence-count">{detail.network_evidence.length}</span></div>
                        {detail.network_evidence.length === 0 ? <div className="evidence-empty">No network evidence available for this alert.</div> : <div className="network-finding-list">{detail.network_evidence.map((finding, index) => <NetworkFindingCard finding={finding} key={`${finding.pattern_type}-${index}`} />)}</div>}
                      </article>
                    </div>
                  </section>

                  <section className="panel graph-panel">
                    <div className="panel-heading graph-heading">
                      <div><div className="section-kicker">RELATIONSHIP VIEW</div><h2>Transaction network</h2><p>Accounts are nodes. Directed edges are observed transactions.</p></div>
                      <div className="graph-counts"><span><strong>{graph?.nodes.length ?? 0}</strong> accounts</span><span><strong>{graph?.edges.length ?? 0}</strong> transactions</span></div>
                    </div>
                    {detail.network_evidence.length === 0 ? <div className="graph-empty"><Network size={22} /><span>No network evidence available for this alert.</span></div> : !graph ? <div className="graph-empty"><LoaderCircle className="spin" size={19} /><span>Loading transaction network…</span></div> : graph.edges.length === 0 ? <div className="graph-empty"><Network size={22} /><span>No transaction edges available for this alert.</span></div> : <NetworkDiagram graph={graph} selectedTransactionId={selectedGraphTransactionId || detail.transaction_id} onSelectTransaction={(transactionId) => void inspectTransaction(transactionId)} />}
                    {relatedEvidence && <div className="related-transaction"><div><span className="section-kicker">SELECTED EDGE</span><strong>{relatedEvidence.transaction_id}</strong></div><span>{relatedEvidence.sender_account} <ArrowRight size={13} /> {relatedEvidence.receiver_account}</span><span>{formatAmount(relatedEvidence.amount)}</span><span>{formatDate(relatedEvidence.timestamp)}</span><button type="button" onClick={() => setRelatedEvidence(null)} aria-label="Close transaction detail"><X size={15} /></button></div>}
                  </section>
                </>
              )}

              <footer className="content-footer"><span>FRAUD SENTINEL <i>/</i> ANALYST CONSOLE</span><span><span className="footer-dot" /> Decision support only · synthetic demo data</span></footer>
            </section>
          </section>
        </div>
      </section>
    </main>
  );
}
