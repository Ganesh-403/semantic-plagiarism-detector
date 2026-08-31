/**
 * healthScoreTypes.ts
 * -------------------
 * TypeScript interfaces for the Document Health Scoring system.
 */

// ---------------------------------------------------------------------------
// Health Dimension
// ---------------------------------------------------------------------------

export interface HealthDimensionScore {
  name: string;
  score: number;
  weight: number;
  weighted: number;
  details: string;
}

export type HealthGrade = "A+" | "A" | "A-" | "B+" | "B" | "B-" | "C" | "D" | "F";

// ---------------------------------------------------------------------------
// Document Health Score
// ---------------------------------------------------------------------------

export interface DocumentHealthScore {
  id: number;
  filename: string;
  overall_score: number;
  grade: HealthGrade;
  dimensions: HealthDimensionScore[];
  checked_at: string;
  gate_passed: boolean;
  gate_reason: string;
}

export interface DocumentHealthListResponse {
  scores: DocumentHealthScore[];
  page: number;
  per_page: number;
  total_items: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

// ---------------------------------------------------------------------------
// Summary & Analytics
// ---------------------------------------------------------------------------

export interface HealthScoreSummary {
  total_scored: number;
  avg_score: number;
  min_score: number;
  max_score: number;
  passed_gate: number;
  failed_gate: number;
  pass_rate: number;
  grade_distribution: Record<HealthGrade, number>;
  last_checked_at: string | null;
}

// ---------------------------------------------------------------------------
// Quality Gate
// ---------------------------------------------------------------------------

export interface HealthGateConfig {
  min_score: number;
  min_grade: string;
  enabled: boolean;
}

export interface HealthGateCheckResult {
  filename: string;
  passed: boolean;
  reason: string;
  overall_score: number;
  grade: HealthGrade;
}

// ---------------------------------------------------------------------------
// Dashboard Filters
// ---------------------------------------------------------------------------

export interface HealthScoreFilters {
  search: string;
  min_score: number;
  max_score: number;
  grade: HealthGrade | "";
  gate_passed: "all" | "passed" | "failed";
  sort_by: "overall_score" | "checked_at" | "grade" | "filename";
  sort_order: "DESC" | "ASC";
}

// ---------------------------------------------------------------------------
// Grade Color Mapping
// ---------------------------------------------------------------------------

export const GRADE_COLORS: Record<HealthGrade, { bg: string; text: string; border: string }> = {
  "A+": { bg: "bg-emerald-500/10", text: "text-emerald-600 dark:text-emerald-400", border: "border-emerald-500/20" },
  "A":  { bg: "bg-emerald-500/10", text: "text-emerald-600 dark:text-emerald-400", border: "border-emerald-500/20" },
  "A-": { bg: "bg-teal-500/10", text: "text-teal-600 dark:text-teal-400", border: "border-teal-500/20" },
  "B+": { bg: "bg-blue-500/10", text: "text-blue-600 dark:text-blue-400", border: "border-blue-500/20" },
  "B":  { bg: "bg-blue-500/10", text: "text-blue-600 dark:text-blue-400", border: "border-blue-500/20" },
  "B-": { bg: "bg-indigo-500/10", text: "text-indigo-600 dark:text-indigo-400", border: "border-indigo-500/20" },
  "C":  { bg: "bg-amber-500/10", text: "text-amber-600 dark:text-amber-400", border: "border-amber-500/20" },
  "D":  { bg: "bg-orange-500/10", text: "text-orange-600 dark:text-orange-400", border: "border-orange-500/20" },
  "F":  { bg: "bg-red-500/10", text: "text-red-600 dark:text-red-400", border: "border-red-500/20" },
};
