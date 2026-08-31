/**
 * batchHistoryMockData.ts
 * ------------------------
 * Mock data generators for the Batch Analysis History dashboard.
 * In production, replace these with actual API calls.
 */

import type {
  BatchRunSummary,
  BatchRunDetailResponse,
  BatchDocumentResult,
  BatchHistorySummary,
  BatchTrendDataPoint,
  BatchTimelineEvent,
  BatchAlert,
  SeverityDistribution,
} from "./batchHistoryTypes";

// ---------------------------------------------------------------------------
// Mock Batch Runs
// ---------------------------------------------------------------------------

export function generateMockBatchRuns(): BatchRunSummary[] {
  const now = new Date();
  const runs: BatchRunSummary[] = [
    {
      run_id: 47,
      started_at: new Date(now.getTime() - 2 * 3600_000).toISOString(),
      completed_at: new Date(now.getTime() - 1.5 * 3600_000).toISOString(),
      status: "completed",
      trigger_source: "manual",
      documents_scanned: 156,
      documents_flagged: 12,
      avg_similarity: 0.23,
      max_similarity: 0.87,
      threshold_used: 0.75,
      duration_ms: 184_300,
      error_message: null,
      created_by: "prof_jackson",
    },
    {
      run_id: 46,
      started_at: new Date(now.getTime() - 8 * 3600_000).toISOString(),
      completed_at: new Date(now.getTime() - 7.5 * 3600_000).toISOString(),
      status: "completed",
      trigger_source: "scheduled",
      documents_scanned: 142,
      documents_flagged: 8,
      avg_similarity: 0.19,
      max_similarity: 0.82,
      threshold_used: 0.70,
      duration_ms: 156_200,
      error_message: null,
      created_by: "system",
    },
    {
      run_id: 45,
      started_at: new Date(now.getTime() - 24 * 3600_000).toISOString(),
      completed_at: null,
      status: "failed",
      trigger_source: "api",
      documents_scanned: 45,
      documents_flagged: 0,
      avg_similarity: 0.0,
      max_similarity: 0.0,
      threshold_used: 0.75,
      duration_ms: null,
      error_message: "Embedding service timeout after 30s",
      created_by: "automation_bot",
    },
    {
      run_id: 44,
      started_at: new Date(now.getTime() - 36 * 3600_000).toISOString(),
      completed_at: new Date(now.getTime() - 35 * 3600_000).toISOString(),
      status: "completed",
      trigger_source: "webhook",
      documents_scanned: 89,
      documents_flagged: 3,
      avg_similarity: 0.14,
      max_similarity: 0.71,
      threshold_used: 0.65,
      duration_ms: 98_100,
      error_message: null,
      created_by: "github_ci",
    },
    {
      run_id: 43,
      started_at: new Date(now.getTime() - 48 * 3600_000).toISOString(),
      completed_at: new Date(now.getTime() - 47.8 * 3600_000).toISOString(),
      status: "completed",
      trigger_source: "manual",
      documents_scanned: 201,
      documents_flagged: 19,
      avg_similarity: 0.28,
      max_similarity: 0.93,
      threshold_used: 0.75,
      duration_ms: 221_500,
      error_message: null,
      created_by: "dr_dupont",
    },
    {
      run_id: 42,
      started_at: new Date(now.getTime() - 72 * 3600_000).toISOString(),
      completed_at: new Date(now.getTime() - 71.9 * 3600_000).toISOString(),
      status: "cancelled",
      trigger_source: "manual",
      documents_scanned: 10,
      documents_flagged: 0,
      avg_similarity: 0.0,
      max_similarity: 0.0,
      threshold_used: 0.80,
      duration_ms: 12_000,
      error_message: null,
      created_by: "admin_user",
    },
    {
      run_id: 41,
      started_at: new Date(now.getTime() - 96 * 3600_000).toISOString(),
      completed_at: new Date(now.getTime() - 95 * 3600_000).toISOString(),
      status: "completed",
      trigger_source: "scheduled",
      documents_scanned: 178,
      documents_flagged: 14,
      avg_similarity: 0.21,
      max_similarity: 0.79,
      threshold_used: 0.70,
      duration_ms: 195_400,
      error_message: null,
      created_by: "system",
    },
  ];
  return runs;
}

// ---------------------------------------------------------------------------
// Mock Summary
// ---------------------------------------------------------------------------

export function generateMockSummary(): BatchHistorySummary {
  return {
    total_runs: 47,
    completed_runs: 41,
    failed_runs: 4,
    success_rate: 87.2,
    total_documents_scanned: 6_842,
    total_documents_flagged: 312,
    avg_similarity: 0.21,
    avg_duration_ms: 165_000,
    last_run_at: new Date(Date.now() - 2 * 3600_000).toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Mock Trends
// ---------------------------------------------------------------------------

export function generateMockTrends(): BatchTrendDataPoint[] {
  const points: BatchTrendDataPoint[] = [];
  const now = new Date();
  for (let i = 29; i >= 0; i--) {
    const date = new Date(now.getTime() - i * 86400_000);
    const hasRun = Math.random() > 0.3;
    points.push({
      scan_date: date.toISOString().slice(0, 10),
      total_runs: hasRun ? Math.floor(Math.random() * 3) + 1 : 0,
      total_docs_scanned: hasRun ? Math.floor(Math.random() * 200) + 50 : 0,
      total_docs_flagged: hasRun ? Math.floor(Math.random() * 20) : 0,
      avg_similarity: hasRun ? 0.15 + Math.random() * 0.15 : 0,
      peak_similarity: hasRun ? 0.6 + Math.random() * 0.35 : 0,
      avg_duration_ms: hasRun ? 120_000 + Math.random() * 100_000 : 0,
    });
  }
  return points;
}

// ---------------------------------------------------------------------------
// Mock Timeline Events
// ---------------------------------------------------------------------------

export function generateMockTimeline(): BatchTimelineEvent[] {
  const now = Date.now();
  return [
    {
      event_id: 201,
      run_id: 47,
      event_type: "batch_completed",
      severity: "success",
      message: "Batch run #47 completed: 156 scanned, 12 flagged",
      metadata: { documents_scanned: 156, documents_flagged: 12 },
      created_at: new Date(now - 2 * 3600_000).toISOString(),
    },
    {
      event_id: 200,
      run_id: 47,
      event_type: "alert_triggered",
      severity: "warning",
      message: "High plagiarism alert: 2 documents scored above 85% similarity",
      metadata: null,
      created_at: new Date(now - 2.1 * 3600_000).toISOString(),
    },
    {
      event_id: 199,
      run_id: 47,
      event_type: "batch_started",
      severity: "info",
      message: "Batch run #47 started (threshold=0.75, trigger=manual)",
      metadata: { threshold: 0.75, trigger: "manual" },
      created_at: new Date(now - 2.5 * 3600_000).toISOString(),
    },
    {
      event_id: 198,
      run_id: 45,
      event_type: "batch_failed",
      severity: "error",
      message: "Batch run #45 failed: Embedding service timeout after 30s",
      metadata: { error_message: "Embedding service timeout after 30s" },
      created_at: new Date(now - 24 * 3600_000).toISOString(),
    },
    {
      event_id: 197,
      run_id: null,
      event_type: "system_maintenance",
      severity: "info",
      message: "Scheduled purge removed 12 batch runs older than 90 days",
      metadata: { deleted_count: 12 },
      created_at: new Date(now - 48 * 3600_000).toISOString(),
    },
    {
      event_id: 196,
      run_id: 43,
      event_type: "batch_completed",
      severity: "success",
      message: "Batch run #43 completed: 201 scanned, 19 flagged (peak 93%)",
      metadata: { documents_scanned: 201, documents_flagged: 19 },
      created_at: new Date(now - 48 * 3600_000).toISOString(),
    },
    {
      event_id: 195,
      run_id: null,
      event_type: "document_uploaded",
      severity: "info",
      message: "New document 'thesis_final_v3.pdf' uploaded by student_42",
      metadata: { filename: "thesis_final_v3.pdf" },
      created_at: new Date(now - 52 * 3600_000).toISOString(),
    },
    {
      event_id: 194,
      run_id: null,
      event_type: "threshold_changed",
      severity: "warning",
      message: "Similarity threshold changed from 0.80 to 0.75 by admin_user",
      metadata: { old_value: 0.80, new_value: 0.75 },
      created_at: new Date(now - 72 * 3600_000).toISOString(),
    },
  ];
}

// ---------------------------------------------------------------------------
// Mock Alerts
// ---------------------------------------------------------------------------

export function generateMockAlerts(): BatchAlert[] {
  const now = Date.now();
  return [
    {
      alert_id: 15,
      run_id: 47,
      alert_type: "high_plagiarism",
      title: "High plagiarism detected in run #47",
      message: "2 documents flagged with peak similarity 87%",
      is_read: 0,
      created_at: new Date(now - 2 * 3600_000).toISOString(),
    },
    {
      alert_id: 14,
      run_id: 45,
      alert_type: "batch_failure",
      title: "Batch run #45 failed",
      message: "Embedding service timeout after 30s",
      is_read: 0,
      created_at: new Date(now - 24 * 3600_000).toISOString(),
    },
    {
      alert_id: 13,
      run_id: 43,
      alert_type: "high_plagiarism",
      title: "High plagiarism detected in run #43",
      message: "19 documents flagged with peak similarity 93%",
      is_read: 0,
      created_at: new Date(now - 48 * 3600_000).toISOString(),
    },
    {
      alert_id: 12,
      run_id: null,
      alert_type: "anomaly_detected",
      title: "Unusual upload pattern detected",
      message: "15 documents uploaded within 5 minutes from the same IP",
      is_read: 1,
      created_at: new Date(now - 72 * 3600_000).toISOString(),
    },
    {
      alert_id: 11,
      run_id: 40,
      alert_type: "threshold_exceeded",
      title: "Threshold exceeded in run #40",
      message: "Average similarity (0.42) exceeded configured threshold",
      is_read: 1,
      created_at: new Date(now - 96 * 3600_000).toISOString(),
    },
  ];
}

// ---------------------------------------------------------------------------
// Mock Run Detail
// ---------------------------------------------------------------------------

export function generateMockRunDetail(runId: number): BatchRunDetailResponse {
  const severityDist: SeverityDistribution = {
    high: 3,
    medium: 5,
    low: 4,
    none: 144,
  };

  const documents: BatchDocumentResult[] = [
    {
      id: 1,
      run_id: runId,
      document_name: "suspected_copy_paper_a.pdf",
      similarity_score: 0.93,
      severity: "high",
      flagged: 1,
      matched_docs: ["original_paper_x.pdf"],
      processing_ms: 1_200,
    },
    {
      id: 2,
      run_id: runId,
      document_name: "paraphrased_essay.docx",
      similarity_score: 0.81,
      severity: "high",
      flagged: 1,
      matched_docs: ["reference_essay.pdf", "online_article.txt"],
      processing_ms: 980,
    },
    {
      id: 3,
      run_id: runId,
      document_name: "shared_section_report.pdf",
      similarity_score: 0.74,
      severity: "medium",
      flagged: 1,
      matched_docs: ["peer_submission.pdf"],
      processing_ms: 1_100,
    },
    {
      id: 4,
      run_id: runId,
      document_name: "original_research_v2.pdf",
      similarity_score: 0.32,
      severity: "low",
      flagged: 0,
      matched_docs: [],
      processing_ms: 850,
    },
    {
      id: 5,
      run_id: runId,
      document_name: "thesis_draft_final.pdf",
      similarity_score: 0.12,
      severity: "none",
      flagged: 0,
      matched_docs: [],
      processing_ms: 720,
    },
  ];

  return {
    run: {
      run_id: runId,
      started_at: new Date(Date.now() - 2 * 3600_000).toISOString(),
      completed_at: new Date(Date.now() - 1.5 * 3600_000).toISOString(),
      status: "completed",
      trigger_source: "manual",
      documents_scanned: 156,
      documents_flagged: 12,
      avg_similarity: 0.23,
      max_similarity: 0.87,
      threshold_used: 0.75,
      duration_ms: 184_300,
      error_message: null,
      created_by: "prof_jackson",
    },
    documents,
    severity_distribution: severityDist,
  };
}
