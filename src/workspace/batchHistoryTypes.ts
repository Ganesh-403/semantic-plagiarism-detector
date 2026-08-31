/**
 * batchHistoryTypes.ts
 * --------------------
 * TypeScript interfaces for the Batch Analysis History dashboard.
 */

// ---------------------------------------------------------------------------
// Batch Run
// ---------------------------------------------------------------------------

export type BatchRunStatus = "running" | "completed" | "failed" | "cancelled";
export type BatchTriggerSource = "manual" | "scheduled" | "api" | "webhook";

export interface BatchRunSummary {
  run_id: number;
  started_at: string;
  completed_at: string | null;
  status: BatchRunStatus;
  trigger_source: BatchTriggerSource;
  documents_scanned: number;
  documents_flagged: number;
  avg_similarity: number;
  max_similarity: number;
  threshold_used: number;
  duration_ms: number | null;
  error_message: string | null;
  created_by: string | null;
}

export interface BatchRunListResponse {
  runs: BatchRunSummary[];
  page: number;
  per_page: number;
  total_items: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

// ---------------------------------------------------------------------------
// Batch Document Result
// ---------------------------------------------------------------------------

export type SeverityLevel = "high" | "medium" | "low" | "none";

export interface BatchDocumentResult {
  id: number;
  run_id: number;
  document_name: string;
  similarity_score: number;
  severity: SeverityLevel;
  flagged: number;
  matched_docs: string[];
  processing_ms: number | null;
}

export interface BatchRunDetailResponse {
  run: BatchRunSummary;
  documents: BatchDocumentResult[];
  severity_distribution: Record<SeverityLevel, number>;
}

// ---------------------------------------------------------------------------
// Timeline
// ---------------------------------------------------------------------------

export type TimelineEventType =
  | "batch_started"
  | "batch_completed"
  | "batch_failed"
  | "batch_cancelled"
  | "document_uploaded"
  | "document_scanned"
  | "threshold_changed"
  | "system_maintenance"
  | "alert_triggered";

export type TimelineSeverity = "info" | "warning" | "error" | "success";

export interface BatchTimelineEvent {
  event_id: number;
  run_id: number | null;
  event_type: TimelineEventType;
  severity: TimelineSeverity;
  message: string;
  metadata: Record<string, unknown> | null;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Alerts
// ---------------------------------------------------------------------------

export type AlertType =
  | "high_plagiarism"
  | "threshold_exceeded"
  | "batch_failure"
  | "anomaly_detected";

export interface BatchAlert {
  alert_id: number;
  run_id: number | null;
  alert_type: AlertType;
  title: string;
  message: string;
  is_read: number;
  created_at: string;
}

// ---------------------------------------------------------------------------
// Analytics
// ---------------------------------------------------------------------------

export interface BatchHistorySummary {
  total_runs: number;
  completed_runs: number;
  failed_runs: number;
  success_rate: number;
  total_documents_scanned: number;
  total_documents_flagged: number;
  avg_similarity: number;
  avg_duration_ms: number;
  last_run_at: string | null;
}

export interface BatchTrendDataPoint {
  scan_date: string;
  total_runs: number;
  total_docs_scanned: number;
  total_docs_flagged: number;
  avg_similarity: number;
  peak_similarity: number;
  avg_duration_ms: number;
}

// ---------------------------------------------------------------------------
// Dashboard Filters
// ---------------------------------------------------------------------------

export interface BatchHistoryFilters {
  search: string;
  status: BatchRunStatus | "";
  trigger_source: BatchTriggerSource | "";
  start_date: string;
  end_date: string;
  min_similarity: number;
  max_similarity: number;
}

// ---------------------------------------------------------------------------
// Severity Distribution for charts
// ---------------------------------------------------------------------------

export interface SeverityDistribution {
  high: number;
  medium: number;
  low: number;
  none: number;
}
