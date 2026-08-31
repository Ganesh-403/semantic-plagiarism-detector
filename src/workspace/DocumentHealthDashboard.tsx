/**
 * DocumentHealthDashboard.tsx
 * ---------------------------
 * Interactive dashboard for monitoring document health scores and quality gates.
 *
 * Features:
 *  - KPI summary cards (total scored, avg score, pass rate, worst grade)
 *  - Grade distribution bar chart (SVG)
 *  - Dimension average radar/spider mini-visualisation
 *  - Live search + filter bar (score range, grade, gate status, sort)
 *  - Paginated scored document table with grade badges and sparklines
 *  - Quality gate configuration panel
 *  - Best / Worst document lists
 *  - Score detail modal with dimension breakdown
 *
 * Styling: Tailwind CSS, glassmorphism cards, Lucide icons, dark-mode aware.
 */

import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
  Search,
  Filter,
  RotateCcw,
  Shield,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Heart,
  TrendingUp,
  TrendingDown,
  FileText,
  Award,
  AlertOctagon,
  ChevronLeft,
  ChevronRight,
  Eye,
  Settings,
  Loader2,
  Activity,
  BarChart3,
  Zap,
  Target,
} from "lucide-react";

import type {
  DocumentHealthScore,
  HealthScoreFilters,
  HealthScoreSummary,
  HealthGateConfig,
  HealthDimensionScore,
  HealthGrade,
} from "./healthScoreTypes";
import { GRADE_COLORS } from "./healthScoreTypes";

// ============================================================================
// Helpers
// ============================================================================

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

function scoreColor(score: number): string {
  if (score >= 90) return "text-emerald-500";
  if (score >= 80) return "text-blue-500";
  if (score >= 70) return "text-amber-500";
  if (score >= 60) return "text-orange-500";
  return "text-red-500";
}

function scoreBgBar(score: number): string {
  if (score >= 90) return "bg-emerald-500";
  if (score >= 80) return "bg-blue-500";
  if (score >= 70) return "bg-amber-500";
  if (score >= 60) return "bg-orange-500";
  return "bg-red-500";
}

// ============================================================================
// Grade Badge
// ============================================================================

function GradeBadge({ grade }: { grade: HealthGrade }) {
  const colors = GRADE_COLORS[grade] ?? GRADE_COLORS["F"];
  return (
    <span
      className={`inline-flex items-center justify-center w-9 h-9 rounded-xl text-sm font-black border ${colors.bg} ${colors.text} ${colors.border}`}
    >
      {grade}
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
// Score Bar (inline)
// ============================================================================

function ScoreBar({ score, size = "md" }: { score: number; size?: "sm" | "md" }) {
  const h = size === "sm" ? "h-1.5" : "h-2.5";
  return (
    <div className={`w-full ${h} rounded-full bg-neutral-100 dark:bg-neutral-800 overflow-hidden`}>
      <div
        className={`${h} rounded-full ${scoreBgBar(score)} transition-all duration-500`}
        style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
      />
    </div>
  );
}

// ============================================================================
// Grade Distribution Chart (SVG bar chart)
// ============================================================================

interface GradeDistChartProps {
  distribution: Record<string, number>;
}

function GradeDistributionChart({ distribution }: GradeDistChartProps) {
  const grades: HealthGrade[] = ["A+", "A", "A-", "B+", "B", "B-", "C", "D", "F"];
  const counts = grades.map((g) => distribution[g] ?? 0);
  const maxCount = Math.max(...counts, 1);
  const barWidth = 28;
  const gap = 6;
  const chartWidth = grades.length * (barWidth + gap);
  const chartHeight = 100;

  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl p-5">
      <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center gap-2">
        <BarChart3 className="w-4 h-4 text-amber-500" />
        Grade Distribution
      </h3>
      <svg width="100%" viewBox={`0 0 ${chartWidth + 20} ${chartHeight + 30}`} className="overflow-visible">
        {grades.map((grade, i) => {
          const count = counts[i];
          const barHeight = (count / maxCount) * chartHeight;
          const x = i * (barWidth + gap) + 10;
          const y = chartHeight - barHeight + 10;
          const colors = GRADE_COLORS[grade];
          const fill =
            grade.startsWith("A")
              ? "#10b981"
              : grade.startsWith("B")
                ? "#3b82f6"
                : grade === "C"
                  ? "#f59e0b"
                  : grade === "D"
                    ? "#f97316"
                    : "#ef4444";

          return (
            <g key={grade}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                rx={6}
                fill={fill}
                opacity={0.85}
              />
              <text
                x={x + barWidth / 2}
                y={y - 4}
                textAnchor="middle"
                fontSize="10"
                fontWeight="bold"
                fill={fill}
              >
                {count > 0 ? count : ""}
              </text>
              <text
                x={x + barWidth / 2}
                y={chartHeight + 24}
                textAnchor="middle"
                fontSize="10"
                fontWeight="600"
                fill="currentColor"
                className="text-neutral-600 dark:text-neutral-400"
              >
                {grade}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

// ============================================================================
// Dimension Radar (simplified SVG spider)
// ============================================================================

interface DimensionRadarProps {
  dimensions: Record<string, number>;
}

function DimensionRadar({ dimensions }: DimensionRadarProps) {
  const names = Object.keys(dimensions);
  const values = Object.values(dimensions);
  const n = names.length;
  if (n === 0) return null;

  const cx = 100, cy = 100, maxR = 80;
  const angleStep = (2 * Math.PI) / n;

  const getPoint = (i: number, r: number) => ({
    x: cx + r * Math.sin(angleStep * i),
    y: cy - r * Math.cos(angleStep * i),
  });

  // Build grid rings
  const rings = [0.25, 0.5, 0.75, 1.0];
  const gridPaths = rings.map((frac) => {
    const pts = Array.from({ length: n }, (_, i) => {
      const p = getPoint(i, maxR * frac);
      return `${p.x},${p.y}`;
    }).join(" ");
    return pts;
  });

  // Data polygon
  const dataPoints = values.map((v, i) => {
    const p = getPoint(i, maxR * (v / 100));
    return `${p.x},${p.y}`;
  }).join(" ");

  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl p-5">
      <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center gap-2">
        <Target className="w-4 h-4 text-amber-500" />
        Dimension Averages
      </h3>
      <svg width="100%" viewBox="0 0 200 220" className="overflow-visible">
        {/* Grid rings */}
        {gridPaths.map((pts, i) => (
          <polygon
            key={i}
            points={pts}
            fill="none"
            stroke="currentColor"
            className="text-neutral-200 dark:text-neutral-700"
            strokeWidth="0.5"
          />
        ))}
        {/* Axis lines */}
        {names.map((_, i) => {
          const p = getPoint(i, maxR);
          return (
            <line
              key={i}
              x1={cx}
              y1={cy}
              x2={p.x}
              y2={p.y}
              stroke="currentColor"
              className="text-neutral-200 dark:text-neutral-700"
              strokeWidth="0.5"
            />
          );
        })}
        {/* Data polygon */}
        <polygon
          points={dataPoints}
          fill="#f59e0b"
          fillOpacity="0.2"
          stroke="#f59e0b"
          strokeWidth="2"
        />
        {/* Data points */}
        {values.map((v, i) => {
          const p = getPoint(i, maxR * (v / 100));
          return (
            <circle key={i} cx={p.x} cy={p.y} r="3" fill="#f59e0b" />
          );
        })}
        {/* Labels */}
        {names.map((name, i) => {
          const p = getPoint(i, maxR + 16);
          const shortName = name.replace("_", " ").split(" ").map(w => w[0].toUpperCase()).join("");
          return (
            <text
              key={i}
              x={p.x}
              y={p.y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize="8"
              fontWeight="600"
              fill="currentColor"
              className="text-neutral-600 dark:text-neutral-400"
            >
              {shortName}
            </text>
          );
        })}
        {/* Value labels */}
        {values.map((v, i) => {
          const p = getPoint(i, maxR * (v / 100) + 10);
          return (
            <text
              key={`v${i}`}
              x={p.x}
              y={p.y}
              textAnchor="middle"
              fontSize="7"
              fontWeight="bold"
              fill="#f59e0b"
            >
              {Math.round(v)}
            </text>
          );
        })}
      </svg>
      {/* Legend below */}
      <div className="flex flex-wrap gap-2 mt-2 justify-center">
        {names.map((name, i) => (
          <span key={i} className="text-[10px] text-neutral-500 bg-neutral-50 dark:bg-neutral-800 px-2 py-0.5 rounded-full">
            {name.replace(/_/g, " ")}: {Math.round(values[i])}
          </span>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Filter Bar
// ============================================================================

interface FilterBarProps {
  filters: HealthScoreFilters;
  setFilters: React.Dispatch<React.SetStateAction<HealthScoreFilters>>;
}

function HealthFilterBar({ filters, setFilters }: FilterBarProps) {
  const handleReset = () => {
    setFilters({
      search: "",
      min_score: 0,
      max_score: 100,
      grade: "",
      gate_passed: "all",
      sort_by: "overall_score",
      sort_order: "DESC",
    });
  };

  return (
    <div className="flex flex-col lg:flex-row gap-3 p-4 bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm transition-all duration-300">
      {/* Search */}
      <div className="relative flex-1">
        <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-neutral-400" />
        <input
          type="text"
          placeholder="Search by document name..."
          value={filters.search}
          onChange={(e) => setFilters((p) => ({ ...p, search: e.target.value }))}
          className="w-full pl-11 pr-4 py-3 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs placeholder:text-neutral-400 focus:outline-none focus:ring-2 focus:ring-amber-500/20 focus:border-amber-500 transition-all"
        />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {/* Score Range */}
        <input
          type="number"
          min={0}
          max={100}
          value={filters.min_score}
          onChange={(e) => setFilters((p) => ({ ...p, min_score: Number(e.target.value) }))}
          className="w-16 px-2 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs text-center focus:outline-none focus:ring-2 focus:ring-amber-500/20"
          placeholder="Min"
        />
        <span className="text-neutral-400 text-xs">—</span>
        <input
          type="number"
          min={0}
          max={100}
          value={filters.max_score}
          onChange={(e) => setFilters((p) => ({ ...p, max_score: Number(e.target.value) }))}
          className="w-16 px-2 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs text-center focus:outline-none focus:ring-2 focus:ring-amber-500/20"
          placeholder="Max"
        />

        {/* Grade Filter */}
        <select
          value={filters.grade}
          onChange={(e) => setFilters((p) => ({ ...p, grade: e.target.value as any }))}
          className="px-3 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs appearance-none focus:outline-none focus:ring-2 focus:ring-amber-500/20 font-medium min-w-[80px]"
        >
          <option value="">All Grades</option>
          {["A+", "A", "A-", "B+", "B", "B-", "C", "D", "F"].map((g) => (
            <option key={g} value={g}>{g}</option>
          ))}
        </select>

        {/* Gate Filter */}
        <select
          value={filters.gate_passed}
          onChange={(e) => setFilters((p) => ({ ...p, gate_passed: e.target.value as any }))}
          className="px-3 py-2.5 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs appearance-none focus:outline-none focus:ring-2 focus:ring-amber-500/20 font-medium min-w-[100px]"
        >
          <option value="all">All Gates</option>
          <option value="passed">✓ Passed</option>
          <option value="failed">✗ Failed</option>
        </select>

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
// Score Table
// ============================================================================

interface ScoreTableProps {
  scores: DocumentHealthScore[];
  onSelectScore: (score: DocumentHealthScore) => void;
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

function ScoreTable({ scores, onSelectScore, page, totalPages, onPageChange }: ScoreTableProps) {
  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-neutral-200 dark:border-neutral-800">
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                Document
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider text-center">
                Grade
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                Score
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                Gate
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider">
                Checked
              </th>
              <th className="px-5 py-3.5 text-[11px] font-semibold text-neutral-500 dark:text-neutral-400 uppercase tracking-wider text-right">
                Actions
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-neutral-100 dark:divide-neutral-800/50">
            {scores.length === 0 && (
              <tr>
                <td colSpan={6} className="px-5 py-12 text-center text-neutral-400 text-sm">
                  <div className="flex flex-col items-center gap-2">
                    <Heart className="w-8 h-8 opacity-30" />
                    <span>No scored documents found.</span>
                  </div>
                </td>
              </tr>
            )}
            {scores.map((s) => (
              <tr
                key={s.id}
                className="hover:bg-amber-50/30 dark:hover:bg-amber-950/10 transition-colors cursor-pointer"
                onClick={() => onSelectScore(s)}
              >
                <td className="px-5 py-3.5 text-xs font-mono font-medium text-neutral-900 dark:text-neutral-100 max-w-[280px] truncate">
                  {s.filename}
                </td>
                <td className="px-5 py-3.5 text-center">
                  <GradeBadge grade={s.grade} />
                </td>
                <td className="px-5 py-3.5">
                  <div className="flex items-center gap-2 min-w-[120px]">
                    <span className={`text-sm font-bold font-mono ${scoreColor(s.overall_score)}`}>
                      {s.overall_score.toFixed(1)}
                    </span>
                    <div className="flex-1">
                      <ScoreBar score={s.overall_score} size="sm" />
                    </div>
                  </div>
                </td>
                <td className="px-5 py-3.5">
                  {s.gate_passed ? (
                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-emerald-600 dark:text-emerald-400">
                      <CheckCircle2 className="w-3 h-3" /> Pass
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1 text-[11px] font-medium text-red-600 dark:text-red-400">
                      <XCircle className="w-3 h-3" /> Fail
                    </span>
                  )}
                </td>
                <td className="px-5 py-3.5 text-[11px] text-neutral-500">
                  {formatDate(s.checked_at)}
                </td>
                <td className="px-5 py-3.5 text-right" onClick={(e) => e.stopPropagation()}>
                  <button
                    onClick={() => onSelectScore(s)}
                    className="p-1.5 rounded-lg hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
                    title="View details"
                  >
                    <Eye className="w-3.5 h-3.5 text-neutral-400" />
                  </button>
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
// Best / Worst List
// ============================================================================

interface BestWorstProps {
  title: string;
  icon: React.ReactNode;
  items: DocumentHealthScore[];
  variant: "best" | "worst";
}

function BestWorstList({ title, icon, items, variant }: BestWorstProps) {
  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm p-5">
      <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center gap-2">
        {icon}
        {title}
      </h3>
      <div className="space-y-2">
        {items.length === 0 && (
          <p className="text-xs text-neutral-400 text-center py-4">No data yet.</p>
        )}
        {items.map((item, i) => (
          <div
            key={item.id}
            className="flex items-center gap-3 p-2.5 rounded-xl bg-neutral-50 dark:bg-neutral-800/50 hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
          >
            <span className="text-[10px] font-bold text-neutral-400 w-4 text-center">
              {i + 1}
            </span>
            <GradeBadge grade={item.grade} />
            <div className="flex-1 min-w-0">
              <p className="text-[11px] font-mono font-medium text-neutral-800 dark:text-neutral-200 truncate">
                {item.filename}
              </p>
              <ScoreBar score={item.overall_score} size="sm" />
            </div>
            <span className={`text-xs font-bold font-mono ${scoreColor(item.overall_score)}`}>
              {item.overall_score.toFixed(1)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ============================================================================
// Quality Gate Config Panel
// ============================================================================

interface GateConfigProps {
  config: HealthGateConfig;
  onUpdate: (config: HealthGateConfig) => void;
}

function QualityGateConfig({ config, onUpdate }: GateConfigProps) {
  const [localConfig, setLocalConfig] = useState(config);

  return (
    <div className="bg-white/60 dark:bg-neutral-900/60 backdrop-blur-md border border-neutral-200 dark:border-neutral-800 rounded-3xl shadow-sm p-5">
      <h3 className="text-sm font-bold text-neutral-900 dark:text-neutral-100 mb-4 flex items-center gap-2">
        <Settings className="w-4 h-4 text-amber-500" />
        Quality Gate Config
      </h3>
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-xs text-neutral-500">Enabled</span>
          <button
            onClick={() => {
              const updated = { ...localConfig, enabled: !localConfig.enabled };
              setLocalConfig(updated);
              onUpdate(updated);
            }}
            className={`w-10 h-5 rounded-full transition-colors ${
              localConfig.enabled ? "bg-emerald-500" : "bg-neutral-300 dark:bg-neutral-600"
            }`}
          >
            <div
              className={`w-4 h-4 rounded-full bg-white shadow transition-transform ${
                localConfig.enabled ? "translate-x-5" : "translate-x-0.5"
              }`}
            />
          </button>
        </div>
        <div>
          <label className="text-[10px] font-medium text-neutral-500 uppercase">Min Score</label>
          <input
            type="range"
            min={0}
            max={100}
            value={localConfig.min_score}
            onChange={(e) => {
              const updated = { ...localConfig, min_score: Number(e.target.value) };
              setLocalConfig(updated);
              onUpdate(updated);
            }}
            className="w-full mt-1 accent-amber-500"
          />
          <span className="text-xs font-mono font-bold text-amber-500">{localConfig.min_score}</span>
        </div>
        <div>
          <label className="text-[10px] font-medium text-neutral-500 uppercase">Min Grade</label>
          <select
            value={localConfig.min_grade}
            onChange={(e) => {
              const updated = { ...localConfig, min_grade: e.target.value };
              setLocalConfig(updated);
              onUpdate(updated);
            }}
            className="w-full mt-1 px-3 py-2 bg-neutral-50 dark:bg-neutral-950 border border-neutral-200 dark:border-neutral-800 rounded-2xl text-xs focus:outline-none focus:ring-2 focus:ring-amber-500/20"
          >
            {["F", "D", "C", "B-", "B", "B+", "A-", "A", "A+"].map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Score Detail Modal
// ============================================================================

interface ScoreDetailModalProps {
  score: DocumentHealthScore;
  onClose: () => void;
}

function ScoreDetailModal({ score, onClose }: ScoreDetailModalProps) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-sm p-4" onClick={onClose}>
      <div
        className="w-full max-w-2xl bg-white dark:bg-neutral-900 rounded-3xl shadow-2xl p-6 max-h-[80vh] overflow-y-auto"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-5">
          <div className="flex items-center gap-3">
            <GradeBadge grade={score.grade} />
            <div>
              <h2 className="text-lg font-bold text-neutral-900 dark:text-neutral-100 font-mono">
                {score.filename}
              </h2>
              <p className="text-xs text-neutral-500">Health Score Report</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-xl hover:bg-neutral-100 dark:hover:bg-neutral-800 transition"
          >
            <XCircle className="w-5 h-5 text-neutral-400" />
          </button>
        </div>

        {/* Overall Score */}
        <div className="mb-5 text-center">
          <p className={`text-5xl font-black ${scoreColor(score.overall_score)}`}>
            {score.overall_score.toFixed(1)}
          </p>
          <p className="text-xs text-neutral-500 mt-1">Overall Health Score</p>
          <div className="mt-2 max-w-xs mx-auto">
            <ScoreBar score={score.overall_score} />
          </div>
        </div>

        {/* Gate Result */}
        <div className={`flex items-center gap-2 p-3 rounded-2xl mb-5 ${
          score.gate_passed
            ? "bg-emerald-50 dark:bg-emerald-950/20 border border-emerald-200 dark:border-emerald-800"
            : "bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800"
        }`}>
          {score.gate_passed ? (
            <CheckCircle2 className="w-4 h-4 text-emerald-500" />
          ) : (
            <XCircle className="w-4 h-4 text-red-500" />
          )}
          <span className="text-xs font-medium">
            {score.gate_passed ? "Passed" : "Failed"} Quality Gate
          </span>
          <span className="text-[10px] text-neutral-500 ml-auto">{score.gate_reason}</span>
        </div>

        {/* Dimension Breakdown */}
        <h4 className="text-xs font-bold text-neutral-500 uppercase tracking-wider mb-3">
          Dimension Breakdown
        </h4>
        <div className="space-y-3">
          {score.dimensions.map((dim) => (
            <div key={dim.name} className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-neutral-700 dark:text-neutral-300">
                  {dim.name.replace(/_/g, " ")}
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-[10px] text-neutral-400">
                    ×{(dim.weight * 100).toFixed(0)}%
                  </span>
                  <span className={`text-xs font-bold font-mono ${scoreColor(dim.score)}`}>
                    {dim.score.toFixed(1)}
                  </span>
                </div>
              </div>
              <ScoreBar score={dim.score} size="sm" />
              <p className="text-[10px] text-neutral-400">{dim.details}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ============================================================================
// Main Dashboard Component
// ============================================================================

export default function DocumentHealthDashboard() {
  // -- State --
  const [filters, setFilters] = useState<HealthScoreFilters>({
    search: "",
    min_score: 0,
    max_score: 100,
    grade: "",
    gate_passed: "all",
    sort_by: "overall_score",
    sort_order: "DESC",
  });
  const [scores, setScores] = useState<DocumentHealthScore[]>([]);
  const [summary, setSummary] = useState<HealthScoreSummary | null>(null);
  const [dimensionAvgs, setDimensionAvgs] = useState<Record<string, number>>({});
  const [gateConfig, setGateConfig] = useState<HealthGateConfig>({
    min_score: 60,
    min_grade: "D",
    enabled: true,
  });
  const [bestDocs, setBestDocs] = useState<DocumentHealthScore[]>([]);
  const [worstDocs, setWorstDocs] = useState<DocumentHealthScore[]>([]);
  const [selectedScore, setSelectedScore] = useState<DocumentHealthScore | null>(null);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);

  // -- Mock data loader --
  const loadDashboard = useCallback(async () => {
    setLoading(true);
    try {
      const { generateMockHealthScores, generateMockHealthSummary, generateMockDimensionAvgs } = await import("./healthScoreMockData");
      setScores(generateMockHealthScores());
      setSummary(generateMockHealthSummary());
      setDimensionAvgs(generateMockDimensionAvgs());
      setBestDocs(generateMockHealthScores().filter((s) => s.overall_score >= 85).slice(0, 5));
      setWorstDocs(generateMockHealthScores().filter((s) => s.overall_score < 65).slice(0, 5));
      setTotalPages(3);
    } catch {
      // Silent fail for mock data
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadDashboard();
  }, [loadDashboard]);

  // -- Handlers --
  const handleSelectScore = (score: DocumentHealthScore) => setSelectedScore(score);
  const handleGateConfigUpdate = (config: HealthGateConfig) => {
    setGateConfig(config);
    // In production: PUT /api/v1/health/gate/config
  };

  // -- Loading --
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
            <div className="w-10 h-10 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center text-white shadow-lg">
              <Heart className="w-5 h-5" />
            </div>
            Document Health Scoring
          </h1>
          <p className="text-xs text-neutral-500 dark:text-neutral-400 mt-1 ml-13">
            Monitor document quality, metadata completeness, and embedding coverage
          </p>
        </div>
      </div>

      {/* KPI Cards */}
      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
          <KpiCard
            title="Total Scored"
            value={summary.total_scored}
            icon={<FileText className="w-5 h-5" />}
            accent="amber"
          />
          <KpiCard
            title="Avg Score"
            value={summary.avg_score.toFixed(1)}
            subtitle={`Range: ${summary.min_score.toFixed(0)}–${summary.max_score.toFixed(0)}`}
            icon={<Activity className="w-5 h-5" />}
            accent="emerald"
          />
          <KpiCard
            title="Pass Rate"
            value={`${summary.pass_rate}%`}
            subtitle={`${summary.passed_gate} pass / ${summary.failed_gate} fail`}
            icon={<CheckCircle2 className="w-5 h-5" />}
            accent="blue"
          />
          <KpiCard
            title="Failed Gate"
            value={summary.failed_gate}
            icon={<AlertOctagon className="w-5 h-5" />}
            accent="red"
          />
          <KpiCard
            title="Last Check"
            value={summary.last_checked_at ? formatDate(summary.last_checked_at) : "Never"}
            icon={<Shield className="w-5 h-5" />}
            accent="violet"
          />
        </div>
      )}

      {/* Filter Bar */}
      <HealthFilterBar filters={filters} setFilters={setFilters} />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Left: Table + Charts (2/3) */}
        <div className="xl:col-span-2 space-y-6">
          <ScoreTable
            scores={scores}
            onSelectScore={handleSelectScore}
            page={page}
            totalPages={totalPages}
            onPageChange={setPage}
          />

          {/* Charts Row */}
          <div className="grid grid-cols-2 gap-4">
            {summary && <GradeDistributionChart distribution={summary.grade_distribution} />}
            <DimensionRadar dimensions={dimensionAvgs} />
          </div>
        </div>

        {/* Right: Sidebar (1/3) */}
        <div className="space-y-6">
          <QualityGateConfig config={gateConfig} onUpdate={handleGateConfigUpdate} />
          <BestWorstList
            title="Top Healthiest"
            icon={<Award className="w-4 h-4 text-emerald-500" />}
            items={bestDocs}
            variant="best"
          />
          <BestWorstList
            title="Needs Attention"
            icon={<AlertTriangle className="w-4 h-4 text-red-500" />}
            items={worstDocs}
            variant="worst"
          />
        </div>
      </div>

      {/* Score Detail Modal */}
      {selectedScore && (
        <ScoreDetailModal
          score={selectedScore}
          onClose={() => setSelectedScore(null)}
        />
      )}
    </div>
  );
}
