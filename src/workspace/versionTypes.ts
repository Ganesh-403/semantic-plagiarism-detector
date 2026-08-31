/**
 * versionTypes.ts
 * TypeScript interfaces for the Document Versioning & Diff Dashboard.
 */

export interface DocumentVersionSnapshot {
  document_hash: string;
  user_id: string;
  assignment_id: string;
  filename: string;
  content_length: number;
  word_count: number;
  version_number: number;
  parent_hash: string | null;
  similarity_to_parent: number | null;
  created_at: string;
}

export interface VersionListResponse {
  items: DocumentVersionSnapshot[];
  page: number;
  per_page: number;
  total: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

export interface VersionLineageEntry {
  document_hash: string;
  version_number: number;
  filename: string;
  word_count: number;
  similarity_to_parent: number | null;
  created_at: string;
}

export interface VersionLineageResponse {
  user_id: string;
  assignment_id: string;
  versions: VersionLineageEntry[];
  total: number;
}

export interface VersionDiff {
  parent_hash: string;
  child_hash: string;
  similarity: number;
  added_words: number;
  removed_words: number;
  changed_words: number;
  jaccard_index: number;
  computed_at: string;
}

export interface VersionSummary {
  total_versions: number;
  total_lineages: number;
  total_diffs: number;
  avg_similarity: number;
  avg_versions_per_document: number;
  unique_users: number;
}

export interface VersionTrendPoint {
  from_version: number;
  to_version: number;
  similarity: number | null;
  added_words: number | null;
  removed_words: number | null;
  created_at: string;
}

export interface VersionTrendResponse {
  user_id: string;
  assignment_id: string;
  trend: VersionTrendPoint[];
  total_points: number;
}

export interface MostRevisedDoc {
  assignment_id: string;
  user_id: string;
  total_versions: number;
  avg_similarity: number;
  last_created: string;
}

export interface MostRevisedResponse {
  documents: MostRevisedDoc[];
}

// -- Grade helpers -----------------------------------------------------------

export function driftGrade(avgSimilarity: number): string {
  if (avgSimilarity >= 0.9) return "A";
  if (avgSimilarity >= 0.75) return "B";
  if (avgSimilarity >= 0.5) return "C";
  if (avgSimilarity >= 0.3) return "D";
  return "F";
}

export const DRIFT_GRADE_COLORS: Record<string, string> = {
  A: "#22c55e",
  B: "#3b82f6",
  C: "#eab308",
  D: "#f97316",
  F: "#ef4444",
};

export const DRIFT_GRADE_LABELS: Record<string, string> = {
  A: "Minimal Drift",
  B: "Low Drift",
  C: "Moderate Drift",
  D: "High Drift",
  F: "Extreme Drift",
};
