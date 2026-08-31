/**
 * DocumentVersionDashboard.tsx
 *
 * Full-featured React dashboard for visualising document version history,
 * similarity trends across drafts, diff summaries, and drift analytics.
 *
 * Features:
 *   • KPI cards (total versions, lineages, avg similarity, unique users)
 *   • Similarity trend sparkline (SVG)
 *   • Version lineage timeline
 *   • Diff summary cards
 *   • Most-revised documents table
 *   • Highest-drift documents table
 *   • Search & filter bar
 *   • Drift-grade color coding
 */

import React, { useState, useMemo, useCallback } from "react";
import type {
  DocumentVersionSnapshot,
  VersionDiff,
  VersionTrendPoint,
  MostRevisedDoc,
  VersionSummary,
} from "./versionTypes";
import { driftGrade, DRIFT_GRADE_COLORS, DRIFT_GRADE_LABELS } from "./versionTypes";
import {
  mockSnapshots,
  mockDiffs,
  mockTrend,
  mockMostRevised,
  mockSummary,
} from "./versionMockData";

// ---------------------------------------------------------------------------
// SVG Mini-Chart
// ---------------------------------------------------------------------------

function TrendSparkline({ points, width = 200, height = 48 }: {
  points: number[];
  width?: number;
  height?: number;
}) {
  if (points.length < 2) return <div style={{ width, height, background: "#1e1e2e", borderRadius: 6 }} />;

  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const step = width / (points.length - 1);

  const pathD = points
    .map((v, i) => {
      const x = i * step;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");

  const areaD = `${pathD} L ${width} ${height} L 0 ${height} Z`;

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <defs>
        <linearGradient id="sparkGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
          <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={areaD} fill="url(#sparkGrad)" />
      <path d={pathD} fill="none" stroke="#6366f1" strokeWidth={2} />
      {points.map((v, i) => {
        const x = i * step;
        const y = height - ((v - min) / range) * (height - 4) - 2;
        return (
          <circle
            key={i}
            cx={x}
            cy={y}
            r={3}
            fill="#6366f1"
            stroke="#0d0d1a"
            strokeWidth={1.5}
          />
        );
      })}
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Drift Badge
// ---------------------------------------------------------------------------

function DriftBadge({ similarity }: { similarity: number }) {
  const grade = driftGrade(similarity);
  const color = DRIFT_GRADE_COLORS[grade];
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
        fontSize: 13,
      }}
    >
      {grade}
      <span style={{ fontWeight: 400, fontSize: 11, opacity: 0.8 }}>
        {DRIFT_GRADE_LABELS[grade]}
      </span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// KPI Card
// ---------------------------------------------------------------------------

function KpiCard({ label, value, sub, icon }: {
  label: string;
  value: string | number;
  sub?: string;
  icon: string;
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
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
        <span style={{ fontSize: 20 }}>{icon}</span>
        <span style={{ color: "#9ca3af", fontSize: 13, fontWeight: 500 }}>{label}</span>
      </div>
      <div style={{ fontSize: 28, fontWeight: 800, color: "#f1f5f9" }}>{value}</div>
      {sub && <div style={{ color: "#6b7280", fontSize: 12, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Search Bar
// ---------------------------------------------------------------------------

function SearchBar({ value, onChange, placeholder }: {
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
}) {
  return (
    <input
      type="text"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder || "Search versions, users, assignments..."}
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
// Version Timeline
// ---------------------------------------------------------------------------

function VersionTimeline({ versions }: { versions: DocumentVersionSnapshot[] }) {
  if (versions.length === 0) return <div style={{ color: "#6b7280" }}>No versions found.</div>;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 0, position: "relative" }}>
      <div
        style={{
          position: "absolute",
          left: 18,
          top: 8,
          bottom: 8,
          width: 2,
          background: "rgba(99,102,241,0.3)",
        }}
      />
      {versions.map((v, i) => (
        <div
          key={v.document_hash}
          style={{
            display: "flex",
            alignItems: "flex-start",
            gap: 14,
            padding: "12px 0",
            position: "relative",
          }}
        >
          <div
            style={{
              width: 12,
              height: 12,
              borderRadius: "50%",
              background: v.similarity_to_parent !== null
                ? DRIFT_GRADE_COLORS[driftGrade(v.similarity_to_parent)]
                : "#6366f1",
              border: "2px solid #0d0d1a",
              flexShrink: 0,
              marginTop: 4,
              marginLeft: 13,
              zIndex: 1,
            }}
          />
          <div style={{ flex: 1 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap" }}>
              <span style={{ fontWeight: 700, color: "#f1f5f9", fontSize: 14 }}>
                v{v.version_number}
              </span>
              <span style={{ color: "#9ca3af", fontSize: 12 }}>{v.filename}</span>
              <span style={{ color: "#6b7280", fontSize: 11 }}>
                {v.word_count.toLocaleString()} words
              </span>
            </div>
            <div style={{ color: "#6b7280", fontSize: 11, marginTop: 2 }}>
              {new Date(v.created_at).toLocaleDateString("en-US", {
                year: "numeric",
                month: "short",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </div>
            {v.similarity_to_parent !== null && (
              <div style={{ marginTop: 4 }}>
                <DriftBadge similarity={v.similarity_to_parent} />
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Diff Card
// ---------------------------------------------------------------------------

function DiffCard({ diff, index }: { diff: VersionDiff; index: number }) {
  const barMax = 800;
  const addWidth = Math.min(100, (diff.added_words / barMax) * 100);
  const removeWidth = Math.min(100, (diff.removed_words / barMax) * 100);

  return (
    <div
      style={{
        background: "rgba(255,255,255,0.03)",
        border: "1px solid rgba(255,255,255,0.06)",
        borderRadius: 10,
        padding: 16,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
        <span style={{ fontWeight: 700, color: "#e2e8f0", fontSize: 14 }}>
          Diff #{index + 1}
        </span>
        <DriftBadge similarity={diff.similarity} />
      </div>

      <div style={{ display: "flex", gap: 16, fontSize: 12, color: "#9ca3af", marginBottom: 10 }}>
        <span>Added: <b style={{ color: "#22c55e" }}>+{diff.added_words}</b></span>
        <span>Removed: <b style={{ color: "#ef4444" }}>-{diff.removed_words}</b></span>
        <span>Changed: <b style={{ color: "#eab308" }}>{diff.changed_words}</b></span>
        <span>Jaccard: <b style={{ color: "#6366f1" }}>{(diff.jaccard_index * 100).toFixed(1)}%</b></span>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 10, color: "#22c55e", width: 50 }}>+ Add</span>
          <div style={{ flex: 1, height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3 }}>
            <div style={{ width: `${addWidth}%`, height: "100%", background: "#22c55e", borderRadius: 3 }} />
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 10, color: "#ef4444", width: 50 }}>- Rem</span>
          <div style={{ flex: 1, height: 6, background: "rgba(255,255,255,0.05)", borderRadius: 3 }}>
            <div style={{ width: `${removeWidth}%`, height: "100%", background: "#ef4444", borderRadius: 3 }} />
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Most-Revised Table
// ---------------------------------------------------------------------------

function MostRevisedTable({ docs }: { docs: MostRevisedDoc[] }) {
  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
        <thead>
          <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.08)" }}>
            {["User", "Assignment", "Versions", "Avg Similarity", "Grade", "Last Updated"].map(
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
          {docs.map((d, i) => (
            <tr
              key={`${d.user_id}-${d.assignment_id}-${i}`}
              style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}
            >
              <td style={{ padding: "10px 12px", color: "#e2e8f0", fontWeight: 500 }}>
                {d.user_id}
              </td>
              <td style={{ padding: "10px 12px", color: "#e2e8f0" }}>{d.assignment_id}</td>
              <td style={{ padding: "10px 12px", color: "#e2e8f0", fontWeight: 700 }}>
                {d.total_versions}
              </td>
              <td style={{ padding: "10px 12px", color: "#9ca3af" }}>
                {(d.avg_similarity * 100).toFixed(1)}%
              </td>
              <td style={{ padding: "10px 12px" }}>
                <DriftBadge similarity={d.avg_similarity} />
              </td>
              <td style={{ padding: "10px 12px", color: "#6b7280", fontSize: 12 }}>
                {new Date(d.last_created).toLocaleDateString()}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main Dashboard Component
// ---------------------------------------------------------------------------

export default function DocumentVersionDashboard() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<"timeline" | "diffs" | "trends" | "analytics">("timeline");

  // Load mock data (in production, replace with API calls)
  const summary: VersionSummary = useMemo(() => mockSummary(), []);
  const snapshots: DocumentVersionSnapshot[] = useMemo(() => mockSnapshots(30), []);
  const diffs: VersionDiff[] = useMemo(() => mockDiffs(12), []);
  const trend: VersionTrendPoint[] = useMemo(() => mockTrend(6), []);
  const mostRevised: MostRevisedDoc[] = useMemo(() => mockMostRevised(8), []);

  // Filter
  const filteredSnapshots = useMemo(() => {
    if (!searchQuery.trim()) return snapshots;
    const q = searchQuery.toLowerCase();
    return snapshots.filter(
      (s) =>
        s.user_id.toLowerCase().includes(q) ||
        s.assignment_id.toLowerCase().includes(q) ||
        s.filename.toLowerCase().includes(q)
    );
  }, [snapshots, searchQuery]);

  const trendSimValues = useMemo(
    () => trend.map((t) => t.similarity ?? 0.5),
    [trend]
  );

  const tabs = [
    { key: "timeline", label: "📋 Timeline", count: filteredSnapshots.length },
    { key: "diffs", label: "🔀 Diffs", count: diffs.length },
    { key: "trends", label: "📈 Trends", count: trend.length },
    { key: "analytics", label: "📊 Analytics", count: mostRevised.length },
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
          📚 Document Versioning & Diff Dashboard
        </h1>
        <p style={{ color: "#6b7280", fontSize: 14, marginTop: 6 }}>
          Track document evolution, similarity drift, and draft lineage across assignments
        </p>
      </div>

      {/* KPI Cards */}
      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 24 }}>
        <KpiCard icon="📄" label="Total Versions" value={summary.total_versions} sub={`Across ${summary.total_lineages} lineages`} />
        <KpiCard icon="🔗" label="Lineages" value={summary.total_lineages} sub={`${summary.avg_versions_per_document} avg versions/doc`} />
        <KpiCard icon="🎯" label="Avg Similarity" value={`${(summary.avg_similarity * 100).toFixed(1)}%`} sub="Across all diffs" />
        <KpiCard icon="👥" label="Unique Users" value={summary.unique_users} sub={`${summary.total_diffs} diffs computed`} />
      </div>

      {/* Trend Sparkline */}
      <div
        style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.06)",
          borderRadius: 12,
          padding: 18,
          marginBottom: 24,
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <span style={{ fontWeight: 700, fontSize: 14 }}>Similarity Trend (Recent Lineage)</span>
          <span style={{ color: "#6b7280", fontSize: 12 }}>{trend.length} version transitions</span>
        </div>
        <TrendSparkline points={trendSimValues} width={600} height={56} />
        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 6, fontSize: 11, color: "#6b7280" }}>
          <span>v1 → v2</span>
          <span>Latest transition</span>
        </div>
      </div>

      {/* Search Bar */}
      <div style={{ marginBottom: 16 }}>
        <SearchBar value={searchQuery} onChange={setSearchQuery} placeholder="Filter by user, assignment, or filename..." />
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
              transition: "all 0.15s",
            }}
          >
            {tab.label}
            <span style={{ marginLeft: 6, opacity: 0.6 }}>({tab.count})</span>
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
        {activeTab === "timeline" && (
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
              Version Timeline ({filteredSnapshots.length} versions)
            </h3>
            <VersionTimeline versions={filteredSnapshots.slice(0, 15)} />
          </div>
        )}

        {activeTab === "diffs" && (
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
              Pairwise Diffs ({diffs.length} computed)
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(340px, 1fr))", gap: 14 }}>
              {diffs.slice(0, 9).map((d, i) => (
                <DiffCard key={d.child_hash} diff={d} index={i} />
              ))}
            </div>
          </div>
        )}

        {activeTab === "trends" && (
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
              Similarity Trend ({trend.length} transitions)
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 12 }}>
              {trend.map((t, i) => (
                <div
                  key={i}
                  style={{
                    background: "rgba(255,255,255,0.03)",
                    border: "1px solid rgba(255,255,255,0.06)",
                    borderRadius: 10,
                    padding: 14,
                  }}
                >
                  <div style={{ fontWeight: 700, fontSize: 14, marginBottom: 6 }}>
                    v{t.from_version} → v{t.to_version}
                  </div>
                  <div style={{ fontSize: 24, fontWeight: 800, color: "#6366f1" }}>
                    {t.similarity !== null ? `${(t.similarity * 100).toFixed(1)}%` : "N/A"}
                  </div>
                  <div style={{ fontSize: 11, color: "#6b7280", marginTop: 6 }}>
                    +{t.added_words ?? 0} / -{t.removed_words ?? 0} words
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === "analytics" && (
          <div>
            <h3 style={{ fontSize: 16, fontWeight: 700, marginBottom: 16 }}>
              Most Revised Documents
            </h3>
            <MostRevisedTable docs={mostRevised} />
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ marginTop: 24, textAlign: "center", color: "#4b5563", fontSize: 12 }}>
        Document Versioning Dashboard • Powered by Semantic Plagiarism Detector
      </div>
    </div>
  );
}
