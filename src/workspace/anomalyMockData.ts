/**
 * anomalyMockData.ts
 * Mock data generators for the Anomaly Detection Dashboard.
 */

import type {
  AnomalyScan,
  AnomalyAlert,
  AnomalySummary,
  AnomalyType,
  AnomalySeverity,
} from "./anomalyTypes";

function randInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randFloat(min: number, max: number): number {
  return Math.round((Math.random() * (max - min) + min) * 10000) / 10000;
}

function isoDate(daysAgo: number): string {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return d.toISOString();
}

const ANOMALY_TYPES: AnomalyType[] = [
  "cluster", "outlier", "pattern", "collusion", "template", "statistical", "temporal",
];
const SEVERITIES: AnomalySeverity[] = ["info", "low", "medium", "high", "critical"];
const USERS = ["alice", "bob", "carol", "dave", "eve", "frank", "grace"];
const DOCS = [
  "essay_intro.docx", "lab_report.docx", "thesis_ch3.docx", "midterm.docx",
  "capstone_draft.docx", "review_paper.docx", "case_study.docx", "proposal.docx",
];

const TITLES: Record<AnomalyType, string[]> = {
  cluster: ["High document similarity cluster", "Suspicious grouping detected"],
  outlier: ["Abnormal similarity score", "Statistical outlier in submission"],
  pattern: ["Repeated phrase pattern", "Matching paragraph structure"],
  collusion: ["Potential collusion between users", "Identical submissions detected"],
  template: ["Template reuse detected", "Shared boilerplate content"],
  statistical: ["Unusual score distribution", "Z-score threshold exceeded"],
  temporal: ["Suspicious submission timing", "Burst submission anomaly"],
};

// ---------------------------------------------------------------------------
// Mock generators
// ---------------------------------------------------------------------------

export function mockScans(count: number = 10): AnomalyScan[] {
  return Array.from({ length: count }, (_, i) => ({
    id: i + 1,
    scan_type: i % 3 === 0 ? "incremental" : "full",
    status: i === 0 ? "running" : i % 5 === 0 ? "failed" : "completed",
    documents_scanned: randInt(10, 200),
    anomalies_found: randInt(0, 15),
    started_at: isoDate(randInt(0, 14)),
    completed_at: i === 0 ? null : isoDate(randInt(0, 13)),
    triggered_by: USERS[randInt(0, USERS.length - 1)],
    error_message: i % 5 === 0 ? "Connection timeout" : null,
  }));
}

export function mockAlerts(count: number = 25): AnomalyAlert[] {
  return Array.from({ length: count }, (_, i) => {
    const type = ANOMALY_TYPES[randInt(0, ANOMALY_TYPES.length - 1)];
    const severity = SEVERITIES[randInt(0, SEVERITIES.length - 1)];
    const titleList = TITLES[type];
    const docCount = randInt(1, 4);

    return {
      id: i + 1,
      scan_id: randInt(1, 10),
      anomaly_type: type,
      severity,
      title: titleList[randInt(0, titleList.length - 1)],
      description: `Anomaly detected involving ${docCount} document(s). Confidence: ${(randFloat(0.5, 0.99) * 100).toFixed(0)}%`,
      confidence: randFloat(0.4, 0.99),
      affected_docs: Array.from({ length: docCount }, () =>
        DOCS[randInt(0, DOCS.length - 1)]
      ),
      evidence: { z_score: randFloat(1.5, 4.0), cluster_size: randInt(2, 8) },
      is_acknowledged: Math.random() > 0.5,
      is_resolved: Math.random() > 0.7,
      notes: "",
      detected_at: isoDate(randInt(0, 30)),
      acknowledged_at: Math.random() > 0.5 ? isoDate(randInt(0, 28)) : null,
      resolved_at: Math.random() > 0.7 ? isoDate(randInt(0, 25)) : null,
    };
  }).sort((a, b) => new Date(b.detected_at).getTime() - new Date(a.detected_at).getTime());
}

export function mockSummary(): AnomalySummary {
  return {
    total_alerts: randInt(30, 100),
    unacknowledged: randInt(5, 20),
    unresolved: randInt(10, 40),
    critical_unresolved: randInt(1, 5),
    avg_confidence: randFloat(0.6, 0.85),
    total_scans: randInt(15, 50),
    completed_scans: randInt(10, 45),
  };
}
