/**
 * AnomalyDashboard.tsx
 *
 * Full-featured React dashboard for visualising anomaly detection alerts,
 * scan history, severity distribution, and detection configuration.
 *
 * Features:
 *   • KPI cards (total alerts, unresolved, critical, avg confidence)
 *   • Severity distribution bar chart (SVG)
 *   • Alert table with severity badges and action buttons
 *   • Scan history timeline
 *   • Config panel with toggle switches
 *   • Search & filter bar
 */

import React, { useState, useMemo } from "react";
import type {
  AnomalyAlert,
  AnomalyScan,
  AnomalySummary,
  AnomalySeverity,
  AnomalyType,
} from "./anomalyTypes";
import { SEVERITY_COLORS, SEVERITY_ICONS, TYPE_LABELS, TYPE_ICONS } from "./anomalyTypes";
import { mockAlerts, mockScans, mockSummary } from "./anomalyMockData";

// ---------------------------------------------------------------------------
// SVG Severity Bar Chart
// ---------------------------------------------------------------------------

function SeverityBarChart({ distribution, width = 400, height = 140 }: {
  distribution: Record<string, number>;
  width?: number;
  height?: number;
}) {
  const severities: AnomalySeverity[] = ["critical", "high", "medium", "low", "info"];
  const maxVal = Math.max(1, ...severities.map((s) => distribution[s] || 0));
  const barWidth = (width - 60) / severities.length - 8;

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      {severities.map((sev, i) => {
        const val = distribution[sev] || 0;
        const barH = (val / maxVal) * (height - 50);
        const x = 30 + i * (barWidth + 8);
        const y = height - 30 - barH;

        return (
          <g key={sev}>
            <rect
              x={x}
              y={y}
              width={barWidth}
              height={barH}
              rx={4}
              fill={SEVERITY_COLORS[sev]}
              opacity={0.85}
            />
            <text
              x={x + barWidth / 2}
              y={y - 6}
              textAnchor="middle"
              fill="#e2e8f0"
              fontSize={12}
              fontWeight={700}
            >
              {val}
            </text>
            <text
              x={x + barWidth / 2}
              y={height - 12}
              textAnchor="middle"
              fill="#6b7280"
              fontSize={10}
            >
              {sev}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Severity Badge
// ---------------------------------------------------------------------------

function SeverityBadge({ severity }: { severity: AnomalySeverity }) {
  const color = SEVERITY_COLORS[severity];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 4,
        padding: "2px 10px",
        borderRadius: 12,
        background: `${color}22`,
        color,
        fontWeight: 700,
        fontSize: 12,
      }}
    >
      {SEVERITY_ICONS[severity]} {severity.toUpperCase()}
    </span>
  );
}

// ---------------------------------------------------------------------------
// KPI Card
// ---------------------------------------------------------------------------

function KpiCard({ label, value, sub, icon, accent }: {
  label: string;
  value: string | number;
  sub?: string;
  icon: string;
  accent?: string;
}) {
  return (
    <div
      style={{
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: 12,
        padding: "18px 20px",
        minWidth: 180,
        flex: "1 1 180px",
        borderTop: accent ? `3px solid ${accent}` : undefined,
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 20 }}>{icon}</span>
        <span style={{ color: "#9ca3af", fontSize: 13, fontWeight: 500 }}>{label}</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: accent || "#f1f5f9" }}>{value}</div>
      {sub && <div style={{ color: "#6b7280", fontSize: 12, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Search Bar
// ---------------------------------------------------------------------------

function SearchBar({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder="Search alerts by title, type, or affected docs..."
      style={{
        width: "100%",
        padding: "10px 16px",
        borderRadius: 10,
        border: "1px solid rgba(255,255,255,0.1)",
        background: "rgba(255,255,255,0.05)",
        color: "#e2e8f0",
        fontSize: 14,
        outline: "none",
      }}
    />
  );
}

// ---------------------------------------------------------------------------
// Alert Table
// ---------------------------------------------------------------------------

function AlertTable({ alerts }: { alerts: AnomalyAlert[] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
            {["Severity", "Type", "Title", "Confidence", "Docs", "Status", "Detected"].map(
              (h) => (
                <th
                  key={h}
                  style={{
                    textAlign: "left",
                    padding: "10px 12px",
                    color: "#9ca3af",
                    fontWeight: 600,
                    fontSize: 12,
                    textTransform: "uppercase",
                    letterSpacing: 0.5,
                  }}
                >
                  {h}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {alerts.map((a) => {
            let statusLabel = "Active";
            let statusColor = "#f97316";
            if (a.is_resolved) {
              statusLabel = "Resolved";
              statusColor = "#22c55e";
            } else if (a.is_acknowledged) {
              statusLabel = "Acknowledged";
              statusColor = "#6366f1";
            }

            return (
              <tr
                key={a.id}
                style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}
              >
                <td style={{ padding: "10px 12px" }}>
                  <SeverityBadge severity={a.severity} />
                </td>
                <td style={{ padding: "10px 12px", color: "#e2e8f0" }}>
                  {TYPE_ICONS[a.anomaly_type]} {TYPE_LABELS[a.anomaly_type]}
                </td>
                <td style={{ padding: "10px 12px", color: "#e2e8f0", fontWeight: 500 }}>
                  {a.title}
                </td>
                <td style={{ padding: "10px 12px", color: "#9ca3af" }}>
                  {(a.confidence * 100).toFixed(1)}%
                </td>
                <td style={{ padding: "10px 12px", color: "#9ca3af" }}>
                  {a.affected_docs.length}
                </td>
                <td style={{ padding: "10px 12px" }}>
                  <span
                    style={{
                      padding: "2px 8px",
                      borderRadius: 8,
                      background: `${statusColor}22`,
                      color: statusColor,
                      fontWeight: 600,
                      fontSize: 11,
                    }}
                  >
                    {statusLabel}
                  </span>
                </td>
                <td style={{ padding: "10px 12px", color: "#6b7280", fontSize: 12 }}>
                  {new Date(a.detected_at).toLocaleDateString()}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Scan Timeline
// ---------------------------------------------------------------------------

function ScanTimeline({ scans }: { scans: AnomalyScan[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      {scans.map((s) => {
        let statusColor = "#22c55e";
        if (s.status === "running") statusColor = "#6366f1";
        else if (s.status === "failed") statusColor = "#ef4444";

        return (
          <div
            key={s.id}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 14,
              padding: "12px 16px",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 10,
            }}
          >
            <div
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                background: statusColor,
                flexShrink: 0,
              }}
            />
            <div style={{ flex: 1 }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontWeight: 700, color: "#f1f5f9", fontSize: 13 }}>
                  Scan #{s.id}
                </span>
                <span
                  style={{
                    padding: "1px 8px",
                    borderRadius: 6,
                    background: `${statusColor}22`,
                    color: statusColor,
                    fontSize: 11,
                    fontWeight: 600,
                  }}
                >
                  {s.status}
                </span>
                <span style={{ color: "#6b7280", fontSize: 11 }}>{s.scan_type}</span>
              </div>
              <div style={{ color: "#6b7280", fontSize: 11, marginTop: 2 }}>
                {s.documents_scanned} docs scanned • {s.anomalies_found} anomalies • by {s.triggered_by}
              </div>
            </div>
            <div style={{ color: "#4b5563", fontSize: 11 }}>
              {new Date(s.started_at).toLocaleDateString()}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Config Panel
// ---------------------------------------------------------------------------

function ConfigPanel() {
  const [config, setConfig] = useState({
    z_score_threshold: 2.5,
    cluster_min_size: 3,
    cluster_similarity: 0.85,
    collusion_threshold: 0.80,
    template_threshold: 0.75,
    enable_statistical: true,
    enable_cluster: true,
    enable_pattern: true,
    enable_collusion: true,
  });

  const toggle = (key: keyof typeof config) => {
    setConfig((prev) => ({ ...prev, [key]: !prev[key as keyof typeof config] }));
  };

  const Toggle = ({ label, enabled, onToggle }: { label: string; enabled: boolean; onToggle: () => void }) => (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "8px 0" }}>
      <span style={{ color: "#e2e8f0", fontSize: 13 }}>{label}</span>
      <button
        onClick={onToggle}
        style={{
          width: 44,
          height: 24,
          borderRadius: 12,
          border: "none",
          background: enabled ? "#6366f1" : "#374151",
          cursor: "pointer",
          position: "relative",
          transition: "background 0.2s",
        }}
      >
        <div
          style={{
            width: 18,
            height: 18,
            borderRadius: "50%",
            background: "#fff",
            position: "absolute",
            top: 3,
            left: enabled ? 23 : 3,
            transition: "left 0.2s",
          }}
        />
      </button>
    </div>
  );

  return (
    <div
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 12,
        padding: 20,
      }}
    >
      <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16, color: "#e2e8f0" }}>
        ⚙️ Detection Configuration
      </h3>
      <Toggle label="Statistical Detection (Z-score)" enabled={config.enable_statistical} onToggle={() => toggle("enable_statistical")} />
      <Toggle label="Cluster Detection" enabled={config.enable_cluster} onToggle={() => toggle("enable_cluster")} />
      <Toggle label="Pattern Matching" enabled={config.enable_pattern} onToggle={() => toggle("enable_pattern")} />
      <Toggle label="Collusion Detection" enabled={config.enable_collusion} onToggle={() => toggle("enable_collusion")} />
      <div style={{ marginTop: 16, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        {[
          { label: "Z-Score Threshold", value: config.z_score_threshold },
          { label: "Cluster Min Size", value: config.cluster_min_size },
          { label: "Cluster Similarity", value: config.cluster_similarity },
          { label: "Collusion Threshold", value: config.collusion_threshold },
        ].map((item) => (
          <div key={item.label}>
            <div style={{ color: "#6b7280", fontSize: 11, marginBottom: 4 }}>{item.label}</div>
            <div style={{ color: "#e2e8f0", fontSize: 16, fontWeight: 700 }}>{item.value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Dashboard Component
// ---------------------------------------------------------------------------

export default function AnomalyDashboard() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<"alerts" | "scans" | "config">("alerts");
  const [severityFilter, setSeverityFilter] = useState<AnomalySeverity | "all">("all");

  const summary: AnomalySummary = useMemo(() => mockSummary(), []);
  const allAlerts: AnomalyAlert[] = useMemo(() => mockAlerts(30), []);
  const scans: AnomalyScan[] = useMemo(() => mockScans(12), []);

  const severityDistribution = useMemo(() => {
    const dist: Record<string, number> = { critical: 0, high: 0, medium: 0, low: 0, info: 0 };
    allAlerts.forEach((a) => { dist[a.severity] = (dist[a.severity] || 0) + 1; });
    return dist;
  }, [allAlerts]);

  const filteredAlerts = useMemo(() => {
    let list = allAlerts;
    if (severityFilter !== "all") {
      list = list.filter((a) => a.severity === severityFilter);
    }
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter(
        (a) =>
          a.title.toLowerCase().includes(q) ||
          a.anomaly_type.toLowerCase().includes(q) ||
          a.affected_docs.some((d) => d.toLowerCase().includes(q))
      );
    }
    return list;
  }, [allAlerts, severityFilter, searchQuery]);

  const tabs = [
    { key: "alerts", label: "🚨 Alerts", count: filteredAlerts.length },
    { key: "scans", label: "🔍 Scans", count: scans.length },
    { key: "config", label: "⚙️ Config", count: 0 },
  ] as const;

  return (
    <div
      style={{
        fontFamily: "'Inter', -apple-system, system-ui, sans-serif",
        background: "#0d0d1a",
        color: "#e2e8f0",
        minHeight: "100vh",
        padding: 24,
      }}
    >
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ fontSize: 26, fontWeight: 800, margin: 0 }}>
          🚨 Anomaly Detection & Alert Dashboard
        </h1>
        <p style={{ color: "#6b7280", fontSize: 14, marginTop: 6 }}>
          Monitor plagiarism anomalies, collusion patterns, and suspicious activity across submissions
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 24 }}>
        <KpiCard icon="🚨" label="Total Alerts" value={summary.total_alerts} sub={`${summary.unacknowledged} unacknowledged`} />
        <KpiCard icon="⚠️" label="Unresolved" value={summary.unresolved} accent="#f97316" />
        <KpiCard icon="🔴" label="Critical" value={summary.critical_unresolved} accent="#ef4444" sub="Unresolved critical alerts" />
        <KpiCard icon="🎯" label="Avg Confidence" value={`${(summary.avg_confidence * 100).toFixed(1)}%`} sub={`${summary.completed_scans} scans completed`} />
      </div>

      {/* Severity Distribution */}
      <div
        style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 12,
          padding: 18,
          marginBottom: 24,
          display: "flex",
          alignItems: "center",
          gap: 24,
        }}
      >
        <div>
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 8 }}>Severity Distribution</div>
          <SeverityBarChart distribution={severityDistribution} width={360} height={120} />
        </div>
        <div style={{ flex: 1 }}>
          <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 12 }}>Quick Actions</div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {(["critical", "high", "medium", "low", "info"] as AnomalySeverity[]).map((sev) => (
              <button
                key={sev}
                onClick={() => setSeverityFilter(severityFilter === sev ? "all" : sev)}
                style={{
                  padding: "6px 14px",
                  borderRadius: 8,
                  border: `1px solid ${SEVERITY_COLORS[sev]}44`,
                  background: severityFilter === sev ? `${SEVERITY_COLORS[sev]}33` : "transparent",
                  color: SEVERITY_COLORS[sev],
                  fontWeight: 600,
                  fontSize: 12,
                  cursor: "pointer",
                }}
              >
                {SEVERITY_ICONS[sev]} {sev} ({severityDistribution[sev] || 0})
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Search */}
      <div style={{ marginBottom: 16 }}>
        <SearchBar value={searchQuery} onChange={setSearchQuery} />
      </div>

      {/* Tabs */}
      <div style={{ display: "flex", gap: 4, marginBottom: 20, borderBottom: "1px solid rgba(255,255,255,0.06)", paddingBottom: 4 }}>
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              padding: "8px 18px",
              borderRadius: 8,
              border: "none",
              background: activeTab === tab.key ? "rgba(99,102,241,0.15)" : "transparent",
              color: activeTab === tab.key ? "#818cf8" : "#6b7280",
              fontWeight: activeTab === tab.key ? 700 : 500,
              fontSize: 13,
              cursor: "pointer",
            }}
          >
            {tab.label}
            {tab.count > 0 && <span style={{ marginLeft: 6, opacity: 0.6 }}>({tab.count})</span>}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div
        style={{
          background: "rgba(255,255,255,0.02)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 12,
          padding: 20,
          minHeight: 300,
        }}
      >
        {activeTab === "alerts" && (
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
              Anomaly Alerts ({filteredAlerts.length})
            </h3>
            <AlertTable alerts={filteredAlerts.slice(0, 15)} />
          </div>
        )}

        {activeTab === "scans" && (
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
              Scan History ({scans.length})
            </h3>
            <ScanTimeline scans={scans} />
          </div>
        )}

        {activeTab === "config" && <ConfigPanel />}
      </div>

      {/* Footer */}
      <div style={{ marginTop: 24, textAlign: "center", color: "#4b5563", fontSize: 12 }}>
        Anomaly Detection Dashboard • Powered by Semantic Plagiarism Detector
      </div>
    </div>
  );
}
