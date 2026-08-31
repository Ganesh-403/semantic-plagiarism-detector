/**
 * anomalyTypes.ts
 * TypeScript interfaces for the Anomaly Detection & Alert Dashboard.
 */

export type AnomalySeverity = "info" | "low" | "medium" | "high" | "critical";
export type AnomalyType =
  | "cluster"
  | "outlier"
  | "pattern"
  | "collusion"
  | "template"
  | "statistical"
  | "temporal";

export interface AnomalyScan {
  id: number;
  scan_type: string;
  status: string;
  documents_scanned: number;
  anomalies_found: number;
  started_at: string;
  completed_at: string | null;
  triggered_by: string;
  error_message: string | null;
}

export interface AnomalyAlert {
  id: number;
  scan_id: number | null;
  anomaly_type: AnomalyType;
  severity: AnomalySeverity;
  title: string;
  description: string;
  confidence: number;
  affected_docs: string[];
  evidence: Record<string, unknown>;
  is_acknowledged: boolean;
  is_resolved: boolean;
  notes: string;
  detected_at: string;
  acknowledged_at: string | null;
  resolved_at: string | null;
}

export interface PaginatedResponse<T> {
  items: T[];
  page: number;
  per_page: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface AnomalySummary {
  total_alerts: number;
  unacknowledged: number;
  unresolved: number;
  critical_unresolved: number;
  avg_confidence: number;
  total_scans: number;
  completed_scans: number;
}

export interface AnomalyConfig {
  z_score_threshold: number;
  cluster_min_size: number;
  cluster_similarity: number;
  outlier_percentile: number;
  collusion_threshold: number;
  template_threshold: number;
  enable_statistical: number;
  enable_cluster: number;
  enable_pattern: number;
  enable_collusion: number;
  updated_at: string | null;
}

// -- Helpers -----------------------------------------------------------------

export const SEVERITY_COLORS: Record<AnomalySeverity, string> = {
  info: "#6b7280",
  low: "#3b82f6",
  medium: "#eab308",
  high: "#f97316",
  critical: "#ef4444",
};

export const SEVERITY_ICONS: Record<AnomalySeverity, string> = {
  info: "ℹ️",
  low: "🔵",
  medium: "🟡",
  high: "🟠",
  critical: "🔴",
};

export const TYPE_LABELS: Record<AnomalyType, string> = {
  cluster: "Cluster Anomaly",
  outlier: "Statistical Outlier",
  pattern: "Pattern Match",
  collusion: "Collusion Detection",
  template: "Template Reuse",
  statistical: "Statistical Anomaly",
  temporal: "Temporal Anomaly",
};

export const TYPE_ICONS: Record<AnomalyType, string> = {
  cluster: "📊",
  outlier: "📈",
  pattern: "🔍",
  collusion: "🤝",
  template: "📋",
  statistical: "📉",
  temporal: "⏰",
};
