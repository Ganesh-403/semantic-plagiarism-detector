/**
 * SimilarityHeatmapDashboard.tsx
 * ------------------------------
 * Interactive dashboard for visualising pairwise document similarity
 * heatmaps, hierarchical clustering, and similarity hotspot alerts.
 *
 * Features:
 *  - Interactive SVG heatmap with hover tooltips
 *  - Color scale legend (white → amber → red)
 *  - Clustering panel with cluster cards and silhouette score
 *  - Hotspot alerts list with severity badges and resolve action
 *  - Snapshot history list
 *  - Filter bar for hotspot severity and similarity range
 *
 * Styling: Tailwind CSS, glassmorphism cards, Lucide icons, dark-mode aware.
 */

import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  Grid3X3,
  Layers,
  AlertTriangle,
  CheckCircle2,
  Search,
  RotateCcw,
  Loader2,
  Eye,
  Clock,
  Target,
  Flame,
  Shield,
  ChevronDown,
  ChevronRight,
  X,
  History,
} from "lucide-react";

import type {
  HeatmapSnapshot,
  ClusteringResult,
  ClusterInfo,
  SimilarityHotspot,
  HotspotSummary,
  HeatmapFilters,
} from "./heatmapTypes";
import { CLUSTER_COLORS } from "./heatmapTypes";

// ============================================================================
// Helpers
// ============================================================================

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function shortLabel(label: string, maxLen = 14): string {
  return label.length > maxLen ? label.slice(0, maxLen - 1) + "…" : label;
}

// ============================================================================
// Similarity Color Scale
// ============================================================================

function similarityColor(val: number): string {
  if (val >= 0.9) return "#dc2626";
  if (val >= 0.7) return "#ea580c";
  if (val >= 0.5) return "#f59e0b";
  if (val >= 0.3) return "#fbbf24";
  if (val >= 0.1) return "#fef3c7";
  return "#ffffff";
}

function similarityTextColor(val: number): string {
  return val >= 0.5 ? "#ffffff" : "#374151";
}

// ============================================================================
// Severity Badge
// ============================================================================

const SEVERITY_STYLES: Record<string, string> = {
  critical: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
  warning: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20",
  info: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
};

function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold uppercase border ${SEVERITY_STYLES[severity] ?? SEVERITY_STYLES.info}`}>
      {severity === "critical" && <Flame className="w-2.5 h-2.5 mr-1" />}
      {severity}
    </span>
  );
}

// ============================================================================
// KPI Card
// ============================================================================

interface KpiProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  accent?: string;
}

function KpiCard({ title, value, subtitle, icon, accent = "amber" }: KpiProps) {
  const accentMap: Record<string, string> = {
    amber: "from-amber-500 to-orange-500",
    red: "from-red-500 to-rose-500",
    blue: "from-blue-500 to-indigo-500",
    emerald: "from-emerald-500 to-teal-500",
    violet: "from-violet-500 to-purple-500",
  };
  return (
    <div className="relative overflow-hidden bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl p-5 shadow-sm hover:shadow-md transition-all duration-300 group">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-[11px] font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">{title}</p>
          <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">{value}</p>
          {subtitle && <p className="text-[11px] text-neutral-400">{subtitle}</p>}
        </div>
        <div className={`flex items-center justify-center w-10 h-10 rounded-2xl bg-gradient-to-br ${accentMap[accent] ?? accentMap.amber} text-white shadow-lg group-hover:scale-110 transition-transform duration-300`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// SVG Heatmap Component
// ============================================================================

interface HeatmapSvgProps {
  labels: string[];
  matrix: number[][];
  cellSize?: number;
}

function HeatmapSvg({ labels, matrix, cellSize = 24 }: HeatmapSvgProps) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; row: string; col: string; val: number } | null>(null);
  const n = labels.length;
  const labelOffset = 130;

  if (n === 0) return <p className="text-xs text-neutral-400 text-center py-8">No heatmap data.</p>;

  const width = n * cellSize + labelOffset + 20;
  const height = n * cellSize + labelOffset + 20;

  return (
    <div className="relative overflow-x-auto">
      <svg
        width={width}
        height={height}
        className="select-none"
        onMouseLeave={() => setTooltip(null)}
      >
        {/* Column labels */}
        {labels.map((label, j) => (
          <text
            key={`col-${j}`}
            x={labelOffset + j * cellSize + cellSize / 2}
            y={labelOffset - 6}
            textAnchor="end"
            fontSize="8"
            fontWeight="600"
            fill="currentColor"
            className="text-neutral-500 dark:text-neutral-400"
            transform={`rotate(-45, ${labelOffset + j * cellSize + cellSize / 2}, ${labelOffset - 6})`}
          >
            {shortLabel(label)}
          </text>
        ))}

        {/* Row labels */}
        {labels.map((label, i) => (
          <text
            key={`row-${i}`}
            x={labelOffset - 6}
            y={labelOffset + i * cellSize + cellSize / 2 + 3}
            textAnchor="end"
            fontSize="8"
            fontWeight="600"
            fill="currentColor"
            className="text-neutral-500 dark:text-neutral-400"
          >
            {shortLabel(label)}
          </text>
        ))}

        {/* Cells */}
        {labels.map((rowLabel, i) =>
          labels.map((colLabel, j) => {
            const val = matrix[i]?.[j] ?? 0;
            const x = labelOffset + j * cellSize;
            const y = labelOffset + i * cellSize;
            const isDiagonal = i === j;

            return (
              <g key={`cell-${i}-${j}`}>
                <rect
                  x={x}
                  y={y}
                  width={cellSize - 1}
                  height={cellSize - 1}
                  rx={3}
                  fill={isDiagonal ? "#f59e0b" : similarityColor(val)}
                  stroke="rgba(0,0,0,0.05)"
                  strokeWidth={0.5}
                  className="cursor-pointer transition-opacity hover:opacity-80"
                  onMouseEnter={(e) =>
                    setTooltip({
                      x: e.clientX,
                      y: e.clientY,
                      row: rowLabel,
                      col: colLabel,
                      val,
                    })
                  }
                />
                {cellSize >= 28 && (
                  <text
                    x={x + cellSize / 2 - 1}
                    y={y + cellSize / 2 + 3}
                    textAnchor="middle"
                    fontSize="7"
                    fontWeight="bold"
                    fill={similarityTextColor(val)}
                    className="pointer-events-none"
                  >
                    {isDiagonal ? "1.0" : val.toFixed(2)}
                  </text>
                )}
              </g>
            );
          })
        )}
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 px-3 py-2 bg-neutral-900 dark:bg-neutral-100 text-white dark:text-neutral-900 text-[11px] font-medium rounded-xl shadow-lg pointer-events-none"
          style={{ left: tooltip.x + 12, top: tooltip.y - 30 }}
        >
          <span className="font-mono">{shortLabel(tooltip.row, 20)}</span>
          {" ↔ "}
          <span className="font-mono">{shortLabel(tooltip.col, 20)}</span>
          {": "}
          <span className="font-bold" style={{ color: similarityColor(tooltip.val) }}>
            {(tooltip.val * 100).toFixed(1)}%
          </span>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Color Legend
// ============================================================================

function ColorLegend() {
  const stops = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
  return (
    <div className="flex items-center gap-2 text-[10px] text-neutral-500">
      <span>Low</span>
      <div className="flex h-3 rounded overflow-hidden">
        {stops.map((s, i) => (
          <div
            key={i}
            className="w-6 h-full"
            style={{ backgroundColor: similarityColor(s) }}
          />
        ))}
      </div>
      <span>High</span>
    </div>
  );
}

// ============================================================================
// Filter Bar
// ============================================================================

interface FilterBarProps {
  filters: HeatmapFilters;
  setFilters: React.Dispatch<React.SetStateAction<HeatmapFilters>>;
}

function HeatmapFilterBar({ filters, setFilters }: FilterBarProps) {
  return (
    <div className="flex flex-wrap items-center gap-3 p-4 bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm">
      <span className="text-[11px] font-semibold text-neutral-500 uppercase">Filters:</span>

      {/* Min similarity */}
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] text-neutral-400">Min sim:</span>
        <input
          type="range"
          min={0}
          max={100}
          value={filters.min_similarity * 100}
          onChange={(e) => setFilters((p) => ({ ...p, min_similarity: Number(e.target.value) / 100 }))}
          className="w-20 accent-amber-500"
        />
        <span className="text-[10px] font-mono text-neutral-500 w-8">{(filters.min_similarity * 100).toFixed(0)}%</span>
      </div>

      {/* Severity */}
      <select
        value={filters.severity}
        onChange={(e) => setFilters((p) => ({ ...p, severity: e.target.value as any }))}
        className="px-2 py-1.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-xl text-[11px] focus:outline-none focus:ring-2 focus:ring-amber-500/20 font-medium"
      >
        <option value="">All Severity</option>
        <option value="critical">Critical</option>
        <option value="warning">Warning</option>
        <option value="info">Info</option>
      </select>

      {/* Unresolved toggle */}
      <label className="flex items-center gap-1.5 text-[11px] cursor-pointer">
        <input
          type="checkbox"
          checked={filters.unresolved_only}
          onChange={(e) => setFilters((p) => ({ ...p, unresolved_only: e.target.checked }))}
          className="rounded border-neutral-300 text-amber-500 w-3 h-3 accent-amber-500"
        />
        Unresolved only
      </label>

      {/* Reset */}
      <button
        onClick={() => setFilters({ min_similarity: 0, max_similarity: 1, unresolved_only: false, severity: "" })}
        className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
      >
        <RotateCcw className="w-3.5 h-3.5 text-neutral-400" />
      </button>
    </div>
  );
}

// ============================================================================
// Clustering Panel
// ============================================================================

interface ClusterPanelProps {
  result: ClusteringResult | null;
}

function ClusteringPanel({ result }: ClusterPanelProps) {
  const [expandedCluster, setExpandedCluster] = useState<number | null>(null);

  if (!result) {
    return (
      <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm p-5">
        <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-2 mb-4">
          <Layers className="w-4 h-4 text-amber-500" />
          Document Clusters
        </h3>
        <p className="text-xs text-neutral-400 text-center py-8">No clustering computed yet.</p>
      </div>
    );
  }

  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm p-5">
      <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-2 mb-4">
        <Layers className="w-4 h-4 text-amber-500" />
        Document Clusters
      </h3>

      {/* Summary bar */}
      <div className="flex items-center gap-4 mb-4 p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-800/50">
        <div className="text-center">
          <p className="text-lg font-bold text-amber-500">{result.num_clusters}</p>
          <p className="text-[9px] text-neutral-500 uppercase">Clusters</p>
        </div>
        <div className="text-center">
          <p className={`text-lg font-bold ${result.silhouette_score >= 0.5 ? "text-emerald-500" : result.silhouette_score >= 0.25 ? "text-amber-500" : "text-red-500"}`}>
            {result.silhouette_score.toFixed(3)}
          </p>
          <p className="text-[9px] text-neutral-500 uppercase">Silhouette</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-bold text-blue-500">{result.linkage_method}</p>
          <p className="text-[9px] text-neutral-500 uppercase">Linkage</p>
        </div>
      </div>

      {/* Cluster cards */}
      <div className="space-y-2 max-h-[350px] overflow-y-auto">
        {result.clusters.map((cluster, idx) => {
          const color = CLUSTER_COLORS[idx % CLUSTER_COLORS.length];
          const isExpanded = expandedCluster === cluster.cluster_id;
          return (
            <div key={cluster.cluster_id} className="rounded-2xl border border-neutral-200 dark:border-neutral-800 overflow-hidden">
              <button
                onClick={() => setExpandedCluster(isExpanded ? null : cluster.cluster_id)}
                className="w-full flex items-center gap-3 p-3 hover:bg-neutral-50 dark:hover:bg-neutral-800/50 transition text-left"
              >
                <div className="w-3 h-3 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                <span className="text-xs font-bold text-neutral-800 dark:text-neutral-200 flex-1">
                  Cluster {cluster.cluster_id}
                </span>
                <span className="text-[10px] text-neutral-400">
                  {cluster.size} doc{cluster.size !== 1 ? "s" : ""}
                </span>
                <span className="text-[10px] font-mono text-neutral-500">
                  sim: {cluster.centroid_score.toFixed(2)}
                </span>
                {isExpanded ? <ChevronDown className="w-3.5 h-3.5 text-neutral-400" /> : <ChevronRight className="w-3.5 h-3.5 text-neutral-400" />}
              </button>
              {isExpanded && (
                <div className="px-3 pb-3 space-y-1">
                  {cluster.documents.map((doc) => (
                    <div key={doc} className="flex items-center gap-2 pl-6 py-1">
                      <div className="w-1.5 h-1.5 rounded-full bg-neutral-300 dark:bg-neutral-600" />
                      <span className="text-[11px] font-mono text-neutral-600 dark:text-neutral-400 truncate">
                        {doc}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================================
// Hotspot Alerts Panel
// ============================================================================

interface HotspotPanelProps {
  hotspots: SimilarityHotspot[];
  summary: HotspotSummary | null;
  onResolve: (id: number) => void;
}

function HotspotPanel({ hotspots, summary, onResolve }: HotspotPanelProps) {
  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm p-5">
      <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-2 mb-4">
        <AlertTriangle className="w-4 h-4 text-red-500" />
        Similarity Hotspots
        {summary && summary.critical_unresolved > 0 && (
          <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-red-500 text-white text-[10px] font-bold">
            {summary.critical_unresolved}
          </span>
        )}
      </h3>

      {/* Summary stats */}
      {summary && (
        <div className="grid grid-cols-3 gap-2 mb-4">
          <div className="p-2 rounded-xl bg-neutral-50 dark:bg-neutral-800/50 text-center">
            <p className="text-sm font-bold text-neutral-900 dark:text-neutral-100">{summary.total_hotspots}</p>
            <p className="text-[9px] text-neutral-500 uppercase">Total</p>
          </div>
          <div className="p-2 rounded-xl bg-red-50 dark:bg-red-950/20 text-center">
            <p className="text-sm font-bold text-red-500">{summary.unresolved}</p>
            <p className="text-[9px] text-neutral-500 uppercase">Open</p>
          </div>
          <div className="p-2 rounded-xl bg-amber-50 dark:bg-amber-950/20 text-center">
            <p className="text-sm font-bold text-amber-500">{(summary.avg_similarity * 100).toFixed(0)}%</p>
            <p className="text-[9px] text-neutral-500 uppercase">Avg Sim</p>
          </div>
        </div>
      )}

      {/* Hotspot list */}
      <div className="space-y-2 max-h-[400px] overflow-y-auto">
        {hotspots.length === 0 && (
          <p className="text-xs text-neutral-400 text-center py-6">No hotspots found.</p>
        )}
        {hotspots.map((h) => (
          <div
            key={h.hotspot_id}
            className={`p-3 rounded-2xl border transition ${
              h.is_resolved
                ? "bg-neutral-50 dark:bg-neutral-900 border-neutral-200 dark:border-neutral-800 opacity-50"
                : h.severity === "critical"
                  ? "bg-red-50/50 dark:bg-red-950/10 border-red-200/50 dark:border-red-900/30"
                  : "bg-neutral-50 dark:bg-neutral-800/50 border-neutral-200 dark:border-neutral-800"
            }`}
          >
            <div className="flex items-start justify-between gap-2">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <SeverityBadge severity={h.severity} />
                  <span className="text-[10px] font-mono font-bold text-amber-500">
                    {(h.similarity * 100).toFixed(1)}%
                  </span>
                </div>
                <p className="text-[11px] font-mono text-neutral-700 dark:text-neutral-300 truncate">
                  {h.doc_a}
                </p>
                <p className="text-[10px] text-neutral-400">↔</p>
                <p className="text-[11px] font-mono text-neutral-700 dark:text-neutral-300 truncate">
                  {h.doc_b}
                </p>
                <p className="text-[9px] text-neutral-400 mt-1">{formatDate(h.created_at)}</p>
              </div>
              {!h.is_resolved && (
                <button
                  onClick={() => onResolve(h.hotspot_id)}
                  className="p-1.5 rounded-lg hover:bg-emerald-50 dark:hover:bg-emerald-950/20 transition flex-shrink-0"
                  title="Mark resolved"
                >
                  <CheckCircle2 className="w-3.5 h-3.5 text-neutral-400 hover:text-emerald-500" />
                </button>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Snapshot History Panel
// ============================================================================

interface SnapshotHistoryProps {
  snapshots: HeatmapSnapshot[];
  onSelect: (snap: HeatmapSnapshot) => void;
}

function SnapshotHistory({ snapshots, onSelect }: SnapshotHistoryProps) {
  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm p-5">
      <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-2 mb-4">
        <History className="w-4 h-4 text-amber-500" />
        Snapshot History
      </h3>
      <div className="space-y-2 max-h-[300px] overflow-y-auto">
        {snapshots.length === 0 && (
          <p className="text-xs text-neutral-400 text-center py-6">No snapshots yet.</p>
        )}
        {snapshots.map((snap) => (
          <button
            key={snap.snapshot_id}
            onClick={() => onSelect(snap)}
            className="w-full text-left p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-800/50 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
          >
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-bold text-neutral-800 dark:text-neutral-200">
                Snapshot #{snap.snapshot_id}
              </span>
              <span className="text-[10px] text-neutral-400">
                {formatDate(snap.computed_at)}
              </span>
            </div>
            <div className="flex items-center gap-3 mt-1">
              <span className="text-[10px] text-neutral-500">{snap.document_count} docs</span>
              <span className="text-[10px] font-mono text-amber-500">
                sim: {(snap.mean_similarity * 100).toFixed(0)}% avg
              </span>
              {snap.hotspots_found !== undefined && snap.hotspots_found > 0 && (
                <span className="text-[10px] text-red-500">
                  {snap.hotspots_found} hotspot{snap.hotspots_found !== 1 ? "s" : ""}
                </span>
              )}
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Main Dashboard
// ============================================================================

export default function SimilarityHeatmapDashboard() {
  const [filters, setFilters] = useState<HeatmapFilters>({
    min_similarity: 0,
    max_similarity: 1,
    unresolved_only: false,
    severity: "",
  });
  const [snapshot, setSnapshot] = useState<HeatmapSnapshot | null>(null);
  const [clustering, setClustering] = useState<ClusteringResult | null>(null);
  const [hotspots, setHotspots] = useState<SimilarityHotspot[]>([]);
  const [hotspotSummary, setHotspotSummary] = useState<HotspotSummary | null>(null);
  const [snapshots, setSnapshots] = useState<HeatmapSnapshot[]>([]);
  const [loading, setLoading] = useState(true);

  const loadDashboard = useCallback(async () => {
    setLoading(true);
    try {
      const mod = await import("./heatmapMockData");
      const snap = mod.generateMockSnapshot();
      setSnapshot(snap);
      setClustering(mod.generateMockClustering());
      setHotspots(mod.generateMockHotspots());
      setHotspotSummary(mod.generateMockHotspotSummary());
      setSnapshots(mod.generateMockSnapshotHistory());
    } catch {
      // Silent fail
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { loadDashboard(); }, [loadDashboard]);

  const handleResolve = (id: number) => {
    setHotspots((prev) => prev.map((h) => h.hotspot_id === id ? { ...h, is_resolved: 1 } : h));
    setHotspotSummary((prev) => prev ? { ...prev, unresolved: Math.max(0, prev.unresolved - 1) } : prev);
  };

  const handleSelectSnapshot = (snap: HeatmapSnapshot) => {
    setSnapshot(snap);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-amber-500 animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-neutral-50 via-white to-amber-50/30 dark:from-neutral-950 dark:via-neutral-900 dark:to-neutral-950 p-6 space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-3">
          <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-amber-500 to-red-500 flex items-center justify-center text-white shadow-lg">
            <Grid3X3 className="w-5 h-5" />
          </div>
          Similarity Heatmap & Clustering
        </h1>
        <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1 ml-13">
          Visualise document similarity, discover clusters, and track hotspot alerts
        </p>
      </div>

      {/* KPI Cards */}
      {snapshot && hotspotSummary && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <KpiCard title="Documents" value={snapshot.document_count} icon={<Grid3X3 className="w-5 h-5" />} accent="amber" />
          <KpiCard
            title="Mean Similarity"
            value={`${(snapshot.mean_similarity * 100).toFixed(1)}%`}
            subtitle={`Range: ${(snapshot.min_similarity * 100).toFixed(0)}–${(snapshot.max_similarity * 100).toFixed(0)}%`}
            icon={<Target className="w-5 h-5" />}
            accent="blue"
          />
          <KpiCard
            title="Clusters"
            value={clustering?.num_clusters ?? 0}
            subtitle={`Silhouette: ${clustering?.silhouette_score.toFixed(2) ?? "—"}`}
            icon={<Layers className="w-5 h-5" />}
            accent="emerald"
          />
          <KpiCard
            title="Hotspots"
            value={hotspotSummary.unresolved}
            subtitle={`${hotspotSummary.critical_unresolved} critical`}
            icon={<AlertTriangle className="w-5 h-5" />}
            accent="red"
          />
        </div>
      )}

      {/* Filter Bar */}
      <HeatmapFilterBar filters={filters} setFilters={setFilters} />

      {/* Main Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Heatmap (2/3) */}
        <div className="xl:col-span-2 space-y-6">
          <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm p-5 overflow-hidden">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
                <Grid3X3 className="w-4 h-4 text-amber-500" />
                Similarity Matrix
                {snapshot && (
                  <span className="text-[10px] font-normal text-neutral-400 ml-2">
                    Snapshot #{snapshot.snapshot_id}
                  </span>
                )}
              </h3>
              <ColorLegend />
            </div>
            {snapshot?.labels && snapshot?.matrix ? (
              <HeatmapSvg labels={snapshot.labels} matrix={snapshot.matrix} cellSize={22} />
            ) : (
              <p className="text-xs text-neutral-400 text-center py-12">No heatmap data available.</p>
            )}
          </div>

          {/* Hotspots */}
          <HotspotPanel
            hotspots={hotspots.filter((h) => {
              if (filters.unresolved_only && h.is_resolved) return false;
              if (filters.severity && h.severity !== filters.severity) return false;
              if (h.similarity < filters.min_similarity) return false;
              return true;
            })}
            summary={hotspotSummary}
            onResolve={handleResolve}
          />
        </div>

        {/* Sidebar (1/3) */}
        <div className="space-y-6">
          <ClusteringPanel result={clustering} />
          <SnapshotHistory snapshots={snapshots} onSelect={handleSelectSnapshot} />
        </div>
      </div>
    </div>
  );
}
