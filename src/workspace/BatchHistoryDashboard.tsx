/**
 * BatchHistoryDashboard.tsx
 * -------------------------
 * Interactive dashboard for monitoring and reviewing batch analysis runs.
 *
 * Features:
 *  - KPI summary cards (total runs, success rate, documents scanned, flagged)
 *  - Live search + filter bar (status, trigger, date range, similarity range)
 *  - Paginated batch run history table with inline status badges
 *  - Drill-down run detail modal with document results and severity distribution
 *  - Timeline audit trail with event type icons
 *  - Alerts panel with unread indicator and mark-all-read
 *  - Daily trend mini-chart (SVG sparkline)
 *
 * Styling: Tailwind CSS, glassmorphism cards, Lucide icons, dark-mode aware.
 */

import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  Search,
  Filter,
  RotateCcw,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  Clock,
  TrendingUp,
  FileText,
  Shield,
  Bell,
  ChevronLeft,
  ChevronRight,
  Eye,
  Trash2,
  Play,
  Pause,
  Loader2,
  Activity,
  BarChart3,
  Zap,
  Filter as FilterIcon,
} from "lucide-react";

import type {
  BatchRunSummary,
  BatchHistoryFilters,
  BatchRunDetailResponse,
  BatchTimelineEvent,
  BatchAlert,
  BatchHistorySummary,
  BatchTrendDataPoint,
  SeverityDistribution,
} from "./batchHistoryTypes";

// ============================================================================
// Helpers
// ============================================================================

function formatDuration(ms: number | null): string {
  if (ms === null || ms === undefined) return "—";
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  return `${(ms / 60_000).toFixed(1)}m`;
}

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

// ============================================================================
// Status Badge
// ============================================================================

const STATUS_CONFIG: Record<string, { color: string; icon: React.ReactNode; label: string }> = {
  running: {
    color: "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20",
    icon: <Loader2 className="w-3 h-3 animate-spin" />,
    label: "Running",
  },
  completed: {
    color: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
    icon: <CheckCircle2 className="w-3 h-3" />,
    label: "Completed",
  },
  failed: {
    color: "bg-red-500/10 text-red-600 dark:text-red-400 border-red-500/20",
    icon: <XCircle className="w-3 h-3" />,
    label: "Failed",
  },
  cancelled: {
    color: "bg-neutral-500/10 text-neutral-600 dark:text-neutral-400 border-neutral-500/20",
    icon: <Pause className="w-3 h-3" />,
    label: "Cancelled",
  },
};

function StatusBadge({ status }: { status: string }) {
  const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.cancelled;
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-semibold border ${config.color}`}
    >
      {config.icon}
      {config.label}
    </span>
  );
}

// ============================================================================
// Severity Badge
// ============================================================================

const SEVERITY_COLORS: Record<string, string> = {
  high: "bg-red-500/10 text-red-600 dark:text-red-400",
  medium: "bg-amber-500/10 text-amber-600 dark:text-amber-400",
  low: "bg-blue-500/10 text-blue-600 dark:text-blue-400",
  none: "bg-neutral-100 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400",
};

function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-md text-[10px] font-bold uppercase tracking-wider ${SEVERITY_COLORS[severity] ?? SEVERITY_COLORS.none}`}
    >
      {severity}
    </span>
  );
}

// ============================================================================
// KPI Card
// ============================================================================

interface KpiCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  accent?: string;
}

function KpiCard({ title, value, subtitle, icon, accent = "amber" }: KpiCardProps) {
  const accentMap: Record<string, string> = {
    amber: "from-amber-500 to-orange-500",
    emerald: "from-emerald-500 to-teal-500",
    red: "from-red-500 to-rose-500",
    blue: "from-blue-500 to-indigo-500",
    violet: "from-violet-500 to-purple-500",
  };

  return (
    <div className="relative overflow-hidden bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl p-5 shadow-sm hover:shadow-md transition-all duration-300 group">
      <div className="flex items-start justify-between">
        <div className="space-y-1">
          <p className="text-[11px] font-medium text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
            {title}
          </p>
          <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">{value}</p>
          {subtitle && (
            <p className="text-[11px] text-neutral-400 dark:text-neutral-500">{subtitle}</p>
          )}
        </div>
        <div
          className={`flex items-center justify-center w-10 h-10 rounded-2xl bg-gradient-to-br ${accentMap[accent] ?? accentMap.amber} text-white shadow-lg group-hover:scale-110 transition-transform duration-300`}
        >
          {icon}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Filter Bar
// ============================================================================

interface FilterBarProps {
  filters: BatchHistoryFilters;
  setFilters: React.Dispatch<React.SetStateAction<BatchHistoryFilters>>;
}

function BatchFilterBar({ filters, setFilters }: FilterBarProps) {
  const handleReset = () => {
    setFilters({
      search: "",
      status: "",
      trigger_source: "",
      start_date: "",
      end_date: "",
      min_similarity: 0,
      max_similarity: 1,
    });
  };

  return (
    <div className="flex flex-col lg:flex-row gap-3 p-4 bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm transition-all duration-300">
      {/* Search */}
      <div className="relative flex-1">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400" />
        <input
          type="text"
          placeholder="Search runs by ID, creator, or error..."
          value={filters.search}
          onChange={(e) => setFilters((p) => ({ ...p, search: e.target.value }))}
          className="w-full pl-11 pr-4 py-3 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs placeholder:text-neutral-400 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 transition-all"
        />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {/* Status Filter */}
        <select
          value={filters.status}
          onChange={(e) => setFilters((p) => ({ ...p, status: e.target.value as any }))}
          className="px-3 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs appearance-none focus:outline-none focus:ring-2 focus:ring-amber-500/20 font-medium min-w-[110px]"
        >
          <option value="">All Status</option>
          <option value="running">Running</option>
          <option value="completed">Completed</option>
          <option value="failed">Failed</option>
          <option value="cancelled">Cancelled</option>
        </select>

        {/* Trigger Filter */}
        <select
          value={filters.trigger_source}
          onChange={(e) => setFilters((p) => ({ ...p, trigger_source: e.target.value as any }))}
          className="px-3 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs appearance-none focus:outline-none focus:ring-2 focus:ring-amber-500/20 font-medium min-w-[110px]"
        >
          <option value="">All Triggers</option>
          <option value="manual">Manual</option>
          <option value="scheduled">Scheduled</option>
          <option value="api">API</option>
          <option value="webhook">Webhook</option>
        </select>

        {/* Date Range */}
        <input
          type="date"
          value={filters.start_date}
          onChange={(e) => setFilters((p) => ({ ...p, start_date: e.target.value }))}
          className="px-3 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs focus:outline-none focus:ring-2 focus:ring-amber-500/20"
        />
        <span className="text-neutral-400 text-xs">→</span>
        <input
          type="date"
          value={filters.end_date}
          onChange={(e) => setFilters((p) => ({ ...p, end_date: e.target.value }))}
          className="px-3 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs focus:outline-none focus:ring-2 focus:ring-amber-500/20"
        />

        {/* Reset */}
        <button
          onClick={handleReset}
          className="flex items-center justify-center p-2.5 text-neutral-500 hover:text-amber-500 bg-neutral-50 dark:bg-neutral-950 hover:bg-amber-50 dark:hover:bg-amber-950/20 border border-neutral-200 dark:border-neutral-800 rounded-2xl transition"
          title="Reset filters"
        >
          <RotateCcw className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

// ============================================================================
// Sparkline (mini SVG chart)
// ============================================================================

interface SparklineProps {
  data: number[];
  width?: number;
  height?: number;
  color?: string;
}

function Sparkline({ data, width = 120, height = 32, color = "#f59e0b" }: SparklineProps) {
  if (data.length < 2) return null;

  const max = Math.max(...data, 0.001);
  const min = Math.min(...data, 0);
  const range = max - min || 1;

  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((v - min) / range) * (height - 4) - 2;
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <svg width={width} height={height} className="opacity-60">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

// ============================================================================
// Batch Run History Table
// ============================================================================

interface RunTableProps {
  runs: BatchRunSummary[];
  onSelectRun: (runId: number) => void;
  onDeleteRun: (runId: number) => void;
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

function RunHistoryTable({ runs, onSelectRun, onDeleteRun, page, totalPages, onPageChange }: RunTableProps) {
  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-neutral-200 dark:border-neutral-800">
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                Run ID
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                Status
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                Started
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                Trigger
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider text-right">
                Scanned
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider text-right">
                Flagged
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider text-right">
                Avg Sim
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider text-right">
                Peak
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider text-right">
                Duration
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider text-right">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800/50">
            {runs.length === 0 && (
              <tr>
                <td colSpan={10} className="px-5 py-12 text-center text-neutral-400 text-sm">
                  <div className="flex flex-col items-center gap-2">
                    <BarChart3 className="w-8 h-8 opacity-30" />
                    <span>No batch runs found matching your filters.</span>
                  </div>
                </td>
              </tr>
            )}
            {runs.map((run) => (
              <tr
                key={run.run_id}
                className="hover:bg-amber-50/30 dark:hover:bg-amber-950/10 transition-colors cursor-pointer"
                onClick={() => onSelectRun(run.run_id)}
              >
                <td className="px-5 py-3.5 text-sm font-mono font-bold text-neutral-900 dark:text-neutral-100">
                  #{run.run_id}
                </td>
                <td className="px-5 py-3.5">
                  <StatusBadge status={run.status} />
                </td>
                <td className="px-5 py-3.5 text-xs text-neutral-600 dark:text-neutral-400">
                  {formatDate(run.started_at)}
                </td>
                <td className="px-5 py-3.5">
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-lg bg-neutral-100 dark:bg-neutral-800 text-[10px] font-medium text-neutral-600 dark:text-neutral-400">
                    <Zap className="w-2.5 h-2.5" />
                    {run.trigger_source}
                  </span>
                </td>
                <td className="px-5 py-3.5 text-xs text-right font-medium text-neutral-700 dark:text-neutral-300">
                  {run.documents_scanned.toLocaleString()}
                </td>
                <td className="px-5 py-3.5 text-xs text-right">
                  <span
                    className={`font-bold ${run.documents_flagged > 0 ? "text-red-500" : "text-neutral-500"}`}
                  >
                    {run.documents_flagged.toLocaleString()}
                  </span>
                </td>
                <td className="px-5 py-3.5 text-xs text-right font-mono text-neutral-600 dark:text-neutral-400">
                  {formatPercent(run.avg_similarity)}
                </td>
                <td className="px-5 py-3.5 text-xs text-right font-mono">
                  <span
                    className={`font-bold ${run.max_similarity > 0.8 ? "text-red-500" : run.max_similarity > 0.5 ? "text-amber-500" : "text-emerald-500"}`}
                  >
                    {formatPercent(run.max_similarity)}
                  </span>
                </td>
                <td className="px-5 py-3.5 text-xs text-right text-neutral-500 dark:text-neutral-400">
                  {formatDuration(run.duration_ms)}
                </td>
                <td className="px-5 py-3.5 text-right" onClick={(e) => e.stopPropagation()}>
                  <div className="flex items-center justify-end gap-1">
                    <button
                      onClick={() => onSelectRun(run.run_id)}
                      className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
                      title="View details"
                    >
                      <Eye className="w-3.5 h-3.5 text-neutral-400" />
                    </button>
                    <button
                      onClick={() => onDeleteRun(run.run_id)}
                      className="p-1.5 rounded-lg hover:bg-red-50 dark:hover:bg-red-950/20 transition"
                      title="Delete run"
                    >
                      <Trash2 className="w-3.5 h-3.5 text-neutral-400 hover:text-red-500" />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-5 py-3 border-t border-neutral-200 dark:border-neutral-800">
          <span className="text-[11px] text-neutral-400">
            Page {page} of {totalPages}
          </span>
          <div className="flex items-center gap-1">
            <button
              onClick={() => onPageChange(page - 1)}
              disabled={page <= 1}
              className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 disabled:opacity-30 transition"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <button
              onClick={() => onPageChange(page + 1)}
              disabled={page >= totalPages}
              className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 disabled:opacity-30 transition"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Timeline Panel
// ============================================================================

interface TimelinePanelProps {
  events: BatchTimelineEvent[];
}

const EVENT_ICONS: Record<string, React.ReactNode> = {
  batch_started: <Play className="w-3 h-3 text-blue-500" />,
  batch_completed: <CheckCircle2 className="w-3 h-3 text-emerald-500" />,
  batch_failed: <XCircle className="w-3 h-3 text-red-500" />,
  batch_cancelled: <Pause className="w-3 h-3 text-neutral-500" />,
  document_uploaded: <FileText className="w-3 h-3 text-violet-500" />,
  document_scanned: <Activity className="w-3 h-3 text-cyan-500" />,
  threshold_changed: <Sliders className="w-3 h-3 text-amber-500" />,
  system_maintenance: <Shield className="w-3 h-3 text-indigo-500" />,
  alert_triggered: <AlertTriangle className="w-3 h-3 text-red-500" />,
};

function Sliders(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <line x1="4" x2="4" y1="21" y2="14" />
      <line x1="4" x2="4" y1="10" y2="3" />
      <line x1="12" x2="12" y1="21" y2="12" />
      <line x1="12" x2="12" y1="8" y2="3" />
      <line x1="20" x2="20" y1="21" y2="16" />
      <line x1="20" x2="20" y1="12" y2="3" />
      <line x1="2" x2="6" y1="14" y2="14" />
      <line x1="10" x2="14" y1="8" y2="8" />
      <line x1="18" x2="22" y1="16" y2="16" />
    </svg>
  );
}

function TimelinePanel({ events }: TimelinePanelProps) {
  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm p-5">
      <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center gap-2">
        <Activity className="w-4 h-4 text-amber-500" />
        Audit Timeline
      </h3>

      <div className="space-y-3 max-h-[400px] overflow-y-auto pr-1">
        {events.length === 0 && (
          <p className="text-xs text-neutral-400 text-center py-8">No timeline events yet.</p>
        )}
        {events.map((event) => (
          <div
            key={event.event_id}
            className="flex gap-3 group"
          >
            {/* Timeline line + dot */}
            <div className="flex flex-col items-center">
              <div className="w-7 h-7 rounded-full bg-neutral-100 dark:bg-neutral-800 flex items-center justify-center border border-neutral-200 dark:border-neutral-700 group-hover:border-amber-500/40 transition">
                {EVENT_ICONS[event.event_type] ?? <Clock className="w-3 h-3 text-neutral-400" />}
              </div>
              <div className="w-px flex-1 bg-neutral-200 dark:bg-neutral-800 mt-1" />
            </div>

            {/* Event content */}
            <div className="pb-4 flex-1 min-w-0">
              <p className="text-xs font-medium text-neutral-800 dark:text-neutral-200 leading-relaxed">
                {event.message}
              </p>
              <p className="text-[10px] text-neutral-400 mt-0.5 flex items-center gap-2">
                <span>{formatDate(event.created_at)}</span>
                {event.run_id && (
                  <span className="font-mono text-amber-500">#{event.run_id}</span>
                )}
                <span
                  className={`px-1.5 py-0.5 rounded text-[9px] font-bold uppercase ${
                    event.severity === "error"
                      ? "bg-red-100 text-red-600 dark:bg-red-900/30 dark:text-red-400"
                      : event.severity === "warning"
                        ? "bg-amber-100 text-amber-600 dark:bg-amber-900/30 dark:text-amber-400"
                        : event.severity === "success"
                          ? "bg-emerald-100 text-emerald-600 dark:bg-emerald-900/30 dark:text-emerald-400"
                          : "bg-neutral-100 text-neutral-500 dark:bg-neutral-800 dark:text-neutral-400"
                  }`}
                >
                  {event.severity}
                </span>
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Alerts Panel
// ============================================================================

interface AlertsPanelProps {
  alerts: BatchAlert[];
  unreadCount: number;
  onMarkAllRead: () => void;
}

function AlertsPanel({ alerts, unreadCount, onMarkAllRead }: AlertsPanelProps) {
  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-2">
          <Bell className="w-4 h-4 text-red-500" />
          Alerts
          {unreadCount > 0 && (
            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-red-500 text-white text-[10px] font-bold">
              {unreadCount}
            </span>
          )}
        </h3>
        {unreadCount > 0 && (
          <button
            onClick={onMarkAllRead}
            className="text-[10px] font-medium text-amber-500 hover:text-amber-600 transition"
          >
            Mark all read
          </button>
        )}
      </div>

      <div className="space-y-2 max-h-[300px] overflow-y-auto">
        {alerts.length === 0 && (
          <p className="text-xs text-neutral-400 text-center py-6">No alerts.</p>
        )}
        {alerts.map((alert) => (
          <div
            key={alert.alert_id}
            className={`p-3 rounded-2xl border transition ${
              alert.is_read
                ? "bg-neutral-50 dark:bg-neutral-900 border-neutral-200 dark:border-neutral-800 opacity-60"
                : "bg-red-50/50 dark:bg-red-950/10 border-red-200/50 dark:border-red-900/30"
            }`}
          >
            <div className="flex items-start gap-2">
              <AlertTriangle
                className={`w-3.5 h-3.5 mt-0.5 flex-shrink-0 ${
                  alert.alert_type === "high_plagiarism" ? "text-red-500" : "text-amber-500"
                }`}
              />
              <div className="flex-1 min-w-0">
                <p className="text-xs font-semibold text-neutral-900 dark:text-neutral-100 truncate">
                  {alert.title}
                </p>
                <p className="text-[11px] text-neutral-500 dark:text-neutral-400 mt-0.5 line-clamp-2">
                  {alert.message}
                </p>
                <p className="text-[10px] text-neutral-400 mt-1">{formatDate(alert.created_at)}</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Severity Distribution Bar
// ============================================================================

function SeverityDistributionBar({ distribution }: { distribution: SeverityDistribution }) {
  const total = distribution.high + distribution.medium + distribution.low + distribution.none;
  if (total === 0) return null;

  const segments = [
    { label: "High", count: distribution.high, color: "bg-red-500" },
    { label: "Med", count: distribution.medium, color: "bg-amber-500" },
    { label: "Low", count: distribution.low, color: "bg-blue-500" },
    { label: "None", count: distribution.none, color: "bg-neutral-300 dark:bg-neutral-600" },
  ];

  return (
    <div className="space-y-2">
      <div className="flex rounded-full overflow-hidden h-2.5 bg-neutral-100 dark:bg-neutral-800">
        {segments.map((seg) =>
          seg.count > 0 ? (
            <div
              key={seg.label}
              className={`${seg.color} transition-all duration-500`}
              style={{ width: `${(seg.count / total) * 100}%` }}
              title={`${seg.label}: ${seg.count}`}
            />
          ) : null
        )}
      </div>
      <div className="flex gap-3">
        {segments.map((seg) => (
          <span key={seg.label} className="flex items-center gap-1 text-[10px] text-neutral-500">
            <span className={`w-2 h-2 rounded-full ${seg.color}`} />
            {seg.label}: {seg.count}
          </span>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Main Dashboard Component
// ============================================================================

export default function BatchHistoryDashboard() {
  // -- State --
  const [filters, setFilters] = useState<BatchHistoryFilters>({
    search: "",
    status: "",
    trigger_source: "",
    start_date: "",
    end_date: "",
    min_similarity: 0,
    max_similarity: 1,
  });
  const [runs, setRuns] = useState<BatchRunSummary[]>([]);
  const [summary, setSummary] = useState<BatchHistorySummary | null>(null);
  const [trends, setTrends] = useState<BatchTrendDataPoint[]>([]);
  const [timeline, setTimeline] = useState<BatchTimelineEvent[]>([]);
  const [alerts, setAlerts] = useState<BatchAlert[]>([]);
  const [unreadAlertCount, setUnreadAlertCount] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [selectedRunId, setSelectedRunId] = useState<number | null>(null);
  const [selectedRunDetail, setSelectedRunDetail] = useState<BatchRunDetailResponse | null>(null);
  const [loading, setLoading] = useState(true);

  // -- Mock data loader (replace with API calls) --
  const loadDashboard = useCallback(async () => {
    setLoading(true);
    try {
      // In production, replace with actual API calls:
      // const runsRes = await fetch(`/api/v1/batch/runs?page=${page}&status=${filters.status}`);
      // For now, use the existing mock generator pattern from the project
      const { generateMockBatchRuns, generateMockSummary, generateMockTrends, generateMockTimeline, generateMockAlerts } = await import("./batchHistoryMockData");

      setRuns(generateMockBatchRuns());
      setSummary(generateMockSummary());
      setTrends(generateMockTrends());
      setTimeline(generateMockTimeline());
      setAlerts(generateMockAlerts());
      setUnreadAlertCount(3);
      setTotalPages(5);
    } catch {
      // Silent fail for mock data
    } finally {
      setLoading(false);
    }
  }, [page, filters]);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  // -- Trend sparkline data --
  const similarityTrend = useMemo(
    () => trends.map((t) => t.avg_similarity),
    [trends]
  );
  const flaggedTrend = useMemo(
    () => trends.map((t) => t.total_docs_flagged ?? 0),
    [trends]
  );

  // -- Handlers --
  const handleSelectRun = (runId: number) => setSelectedRunId(runId);
  const handleDeleteRun = async (runId: number) => {
    if (window.confirm(`Delete batch run #${runId}?`)) {
      // await fetch(`/api/v1/batch/runs/${runId}`, { method: "DELETE" });
      setRuns((prev) => prev.filter((r) => r.run_id !== runId));
    }
  };
  const handleMarkAllRead = () => {
    setAlerts((prev) => prev.map((a) => ({ ...a, is_read: 1 })));
    setUnreadAlertCount(0);
  };

  // -- Loading state --
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
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-3">
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-amber-500 to-orange-500 flex items-center justify-center text-white shadow-lg">
              <BarChart3 className="w-5 h-5" />
            </div>
            Batch Analysis History
          </h1>
          <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1 ml-13">
            Monitor, review, and audit all batch plagiarism scan runs
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
          <KpiCard
            title="Total Runs"
            value={summary.total_runs}
            subtitle={`${summary.completed_runs} completed`}
            icon={<BarChart3 className="w-5 h-5" />}
            accent="amber"
          />
          <KpiCard
            title="Success Rate"
            value={`${summary.success_rate}%`}
            subtitle={`${summary.failed_runs} failed`}
            icon={<CheckCircle2 className="w-5 h-5" />}
            accent="emerald"
          />
          <KpiCard
            title="Documents Scanned"
            value={summary.total_documents_scanned.toLocaleString()}
            icon={<FileText className="w-5 h-5" />}
            accent="blue"
          />
          <KpiCard
            title="Documents Flagged"
            value={summary.total_documents_flagged.toLocaleString()}
            icon={<AlertTriangle className="w-5 h-5" />}
            accent="red"
          />
          <KpiCard
            title="Avg Duration"
            value={formatDuration(summary.avg_duration_ms)}
            subtitle={`Last run: ${summary.last_run_at ? formatDate(summary.last_run_at) : "Never"}`}
            icon={<Clock className="w-5 h-5" />}
            accent="violet"
          />
        </div>
      )}

      {/* Filter Bar */}
      <BatchFilterBar filters={filters} setFilters={setFilters} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Run History Table (2/3 width) */}
        <div className="xl:col-span-2 space-y-6">
          <RunHistoryTable
            runs={runs}
            onSelectRun={handleSelectRun}
            onDeleteRun={handleDeleteRun}
            page={page}
            totalPages={totalPages}
            onPageChange={setPage}
          />

          {/* Trend Mini-Charts */}
          {trends.length > 0 && (
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-wider">
                    Avg Similarity Trend
                  </span>
                  <TrendingUp className="w-4 h-4 text-amber-500" />
                </div>
                <Sparkline data={similarityTrend} width={280} height={48} color="#f59e0b" />
              </div>
              <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl p-5">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-[11px] font-semibold text-neutral-500 uppercase tracking-wider">
                    Flagged Documents Trend
                  </span>
                  <AlertTriangle className="w-4 h-4 text-red-500" />
                </div>
                <Sparkline data={flaggedTrend} width={280} height={48} color="#ef4444" />
              </div>
            </div>
          )}

          {/* Severity Distribution (if run selected) */}
          {selectedRunDetail && (
            <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl p-5">
              <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 mb-3 flex items-center gap-2">
                <Shield className="w-4 h-4 text-amber-500" />
                Run #{selectedRunDetail.run.run_id} — Severity Distribution
              </h3>
              <SeverityDistributionBar distribution={selectedRunDetail.severity_distribution} />
              <div className="mt-4 max-h-[200px] overflow-y-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-neutral-200 dark:border-neutral-800">
                      <th className="pb-2 font-medium text-neutral-500">Document</th>
                      <th className="pb-2 font-medium text-neutral-500 text-right">Score</th>
                      <th className="pb-2 font-medium text-neutral-500">Severity</th>
                      <th className="pb-2 font-medium text-neutral-500 text-center">Flagged</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800/50">
                    {selectedRunDetail.documents.map((doc) => (
                      <tr key={doc.id}>
                        <td className="py-2 font-mono text-neutral-800 dark:text-neutral-200 truncate max-w-[200px]">
                          {doc.document_name}
                        </td>
                        <td className="py-2 text-right font-mono">{formatPercent(doc.similarity_score)}</td>
                        <td className="py-2">
                          <SeverityBadge severity={doc.severity} />
                        </td>
                        <td className="py-2 text-center">
                          {doc.flagged ? (
                            <AlertTriangle className="w-3.5 h-3.5 text-red-500 mx-auto" />
                          ) : (
                            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 mx-auto" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Sidebar: Timeline + Alerts (1/3 width) */}
        <div className="space-y-6">
          <AlertsPanel
            alerts={alerts}
            unreadCount={unreadAlertCount}
            onMarkAllRead={handleMarkAllRead}
          />
          <TimelinePanel events={timeline} />
        </div>
      </div>

      {/* Run Detail Modal */}
      {selectedRunId && (
        <RunDetailModal
          runId={selectedRunId}
          onClose={() => {
            setSelectedRunId(null);
            setSelectedRunDetail(null);
          }}
          onLoaded={(detail) => setSelectedRunDetail(detail)}
        />
      )}
    </div>
  );
}

// ============================================================================
// Run Detail Modal
// ============================================================================

interface RunDetailModalProps {
  runId: number;
  onClose: () => void;
  onLoaded: (detail: BatchRunDetailResponse) => void;
}

function RunDetailModal({ runId, onClose, onLoaded }: RunDetailModalProps) {
  const [detail, setDetail] = useState<BatchRunDetailResponse | null>(null);

  useEffect(() => {
    // In production: fetch(`/api/v1/batch/runs/${runId}`)
    import("./batchHistoryMockData").then((mod) => {
      const d = mod.generateMockRunDetail(runId);
      setDetail(d);
      onLoaded(d);
    });
  }, [runId, onLoaded]);

  if (!detail) {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm">
        <Loader2 className="w-8 h-8 text-white animate-spin" />
      </div>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4" onClick={onClose}>
      <div
        className="w-full max-w-2xl bg-white dark:bg-neutral-900 rounded-3xl shadow-2xl p-6 max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-5">
          <h2 className="text-lg font-bold text-neutral-900 dark:text-neutral-100">
            Batch Run #{detail.run.run_id}
          </h2>
          <button
            onClick={onClose}
            className="p-2 rounded-xl hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
          >
            <XCircle className="w-5 h-5 text-neutral-400" />
          </button>
        </div>

        <div className="grid grid-cols-2 gap-3 mb-5">
          <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-800/50">
            <span className="text-[10px] font-medium text-neutral-500 uppercase">Status</span>
            <div className="mt-1"><StatusBadge status={detail.run.status} /></div>
          </div>
          <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-800/50">
            <span className="text-[10px] font-medium text-neutral-500 uppercase">Trigger</span>
            <p className="mt-1 text-xs font-medium text-neutral-800 dark:text-neutral-200">{detail.run.trigger_source}</p>
          </div>
          <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-800/50">
            <span className="text-[10px] font-medium text-neutral-500 uppercase">Scanned / Flagged</span>
            <p className="mt-1 text-xs font-bold text-neutral-800 dark:text-neutral-200">
              {detail.run.documents_scanned} / <span className="text-red-500">{detail.run.documents_flagged}</span>
            </p>
          </div>
          <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-800/50">
            <span className="text-[10px] font-medium text-neutral-500 uppercase">Threshold</span>
            <p className="mt-1 text-xs font-mono text-neutral-800 dark:text-neutral-200">
              {formatPercent(detail.run.threshold_used)}
            </p>
          </div>
          <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-800/50">
            <span className="text-[10px] font-medium text-neutral-500 uppercase">Avg Similarity</span>
            <p className="mt-1 text-xs font-mono font-bold text-amber-500">{formatPercent(detail.run.avg_similarity)}</p>
          </div>
          <div className="p-3 rounded-2xl bg-neutral-50 dark:bg-neutral-800/50">
            <span className="text-[10px] font-medium text-neutral-500 uppercase">Peak Similarity</span>
            <p className="mt-1 text-xs font-mono font-bold text-red-500">{formatPercent(detail.run.max_similarity)}</p>
          </div>
        </div>

        <SeverityDistributionBar distribution={detail.severity_distribution} />

        <div className="mt-5">
          <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-2">
            Document Results ({detail.documents.length})
          </h4>
          <div className="space-y-1.5 max-h-[250px] overflow-y-auto">
            {detail.documents.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center justify-between p-2.5 rounded-xl bg-neutral-50 dark:bg-neutral-800/50 text-xs"
              >
                <span className="font-mono text-neutral-800 dark:text-neutral-200 truncate max-w-[250px]">
                  {doc.document_name}
                </span>
                <div className="flex items-center gap-3">
                  <span className="font-mono font-bold text-neutral-600 dark:text-neutral-400">
                    {formatPercent(doc.similarity_score)}
                  </span>
                  <SeverityBadge severity={doc.severity} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
