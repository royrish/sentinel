"use client";

import {
  Activity,
  ArrowUpRight,
  BellRing,
  CircleDot,
  Command,
  GitFork,
  Layers3,
  ShieldCheck,
  Waypoints,
} from "lucide-react";
import { useEffect, useState } from "react";

import { getHealth } from "@/lib/api";

type ConnectionState = "checking" | "connected" | "unavailable";

const navigation = [
  { label: "Command Center", icon: Command, active: true },
  { label: "Alerts", icon: BellRing, active: false },
  { label: "Transactions", icon: Activity, active: false },
  { label: "Networks", icon: Waypoints, active: false },
  { label: "Patterns", icon: Layers3, active: false },
  { label: "Simulator", icon: GitFork, active: false },
];

export default function Home() {
  const [connection, setConnection] = useState<ConnectionState>("checking");

  useEffect(() => {
    let active = true;

    getHealth()
      .then(() => {
        if (active) setConnection("connected");
      })
      .catch(() => {
        if (active) setConnection("unavailable");
      });

    return () => {
      active = false;
    };
  }, []);

  const connectionLabel = {
    checking: "Checking service",
    connected: "API connected",
    unavailable: "API unavailable",
  }[connection];

  return (
    <main className="app-frame">
      <aside className="sidebar">
        <a className="brand" href="#command-center" aria-label="Fraud Sentinel home">
          <span className="brand-mark"><ShieldCheck size={20} strokeWidth={1.8} /></span>
          <span className="brand-name">Fraud<span>Sentinel</span></span>
        </a>

        <div className="workspace-label">INVESTIGATION</div>
        <nav className="primary-nav" aria-label="Primary navigation">
          {navigation.map(({ label, icon: Icon, active }) => (
            <button
              className={`nav-item${active ? " nav-item-active" : ""}`}
              key={label}
              type="button"
              disabled={!active}
              aria-current={active ? "page" : undefined}
              title={active ? label : `${label} will be available in a later phase`}
            >
              <Icon size={17} strokeWidth={1.8} />
              <span>{label}</span>
              {active && <ArrowUpRight className="nav-current" size={14} />}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="environment-tag"><span className="environment-dot" /> LOCAL ENVIRONMENT</div>
          <div className="sidebar-caption">Analyst workspace</div>
        </div>
      </aside>

      <section className="content-area" id="command-center">
        <header className="topbar">
          <div className="breadcrumb"><span>Workspace</span><span className="breadcrumb-divider">/</span><strong>Command Center</strong></div>
          <div className={`service-status status-${connection}`} role="status" aria-live="polite">
            <span className="status-dot" />{connectionLabel}
          </div>
        </header>

        <div className="page-content">
          <div className="page-heading">
            <div>
              <div className="eyebrow"><CircleDot size={13} /> FRAUD OPERATIONS</div>
              <h1>Command Center</h1>
              <p className="page-subtitle">Investigation workspace</p>
            </div>
            <div className="date-stamp">FOUNDATION BUILD <span>·</span> NO DATA FEED</div>
          </div>

          <section className="empty-state" aria-labelledby="empty-title">
            <div className="empty-icon"><ShieldCheck size={25} strokeWidth={1.6} /></div>
            <div className="empty-copy">
              <div className="eyebrow">SYSTEM READY</div>
              <h2 id="empty-title">Your investigation workspace is ready.</h2>
              <p>Connect a transaction data source in a later phase to begin reviewing evidence and investigating activity.</p>
            </div>
            <div className="empty-status">
              <span className="status-line"><span className="status-dot status-dot-ready" /> Interface available</span>
              <span className="status-line"><span className={`status-dot${connection === "connected" ? " status-dot-ready" : ""}`} /> {connectionLabel}</span>
              <span className="status-line"><span className="status-dot" /> Data source not configured</span>
            </div>
          </section>

          <footer className="page-footer">
            <span>FRAUD SENTINEL <span className="footer-separator">/</span> ANALYST CONSOLE</span>
            <span>DETECTION CAPABILITIES NOT YET CONFIGURED</span>
          </footer>
        </div>
      </section>
    </main>
  );
}