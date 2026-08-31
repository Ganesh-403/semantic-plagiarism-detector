/**
 * heatmapTypes.ts
 * ----------------
 * TypeScript interfaces for the Similarity Heatmap & Clustering dashboard.
 */

// ---------------------------------------------------------------------------
// Heatmap
// ---------------------------------------------------------------------------

export interface HeatmapSnapshot {
  snapshot_id: number;
  labels?: string[];
  matrix?: number[][];
  document_count: number;
  min_similarity: number;
  max_similarity: number;
  mean_similarity: number;
  computed_at: string;
  computed_by?: string;
  notes?: string;
  hotspots_found?: number;
}

export interface HeatmapSnapshotListResponse {
  snapshots: HeatmapSnapshot[];
  page: number;
  per_page: number;
  total_items: number;
  total_pages: number;
  has_next: boolean;
  has_previous: boolean;
}

// ---------------------------------------------------------------------------
// Clustering
// ---------------------------------------------------------------------------

export interface ClusterInfo {
  cluster_id: number;
  documents: string[];
  centroid_score: number;
  size: number;
}

export interface ClusteringResult {
  result_id: number;
  num_clusters: number;
  silhouette_score: number;
  linkage_method: string;
  distance_threshold: number;
  clusters: ClusterInfo[];
  document_assignments: Record<string, number>;
  computed_at: string;
}

// ---------------------------------------------------------------------------
// Hotspots
// ---------------------------------------------------------------------------

export type HotspotSeverity = "info" | "warning" | "critical";

export interface SimilarityHotspot {
  hotspot_id: number;
  snapshot_id: number | null;
  doc_a: string;
  doc_b: string;
  similarity: number;
  severity: HotspotSeverity;
  created_at: string;
  is_resolved: number;
}

export interface HotspotSummary {
  total_hotspots: number;
  unresolved: number;
  critical_unresolved: number;
  avg_similarity: number;
}

// ---------------------------------------------------------------------------
// SVG Heatmap Data
// ---------------------------------------------------------------------------

export interface HeatmapCellData {
  x: number;
  y: number;
  color: string;
  row_label: string;
  col_label: string;
  value: number;
}

export interface HeatmapSvgData {
  cell_size: number;
  width: number;
  height: number;
  labels: string[];
  cells: HeatmapCellData[];
}

// ---------------------------------------------------------------------------
// Dashboard Filters
// ---------------------------------------------------------------------------

export interface HeatmapFilters {
  min_similarity: number;
  max_similarity: number;
  unresolved_only: boolean;
  severity: HotspotSeverity | "";
}

// Cluster color palette
export const CLUSTER_COLORS = [
  "#f59e0b", "#3b82f6", "#10b981", "#ef4444", "#8b5cf6",
  "#ec4899", "#06b6d4", "#f97316", "#84cc16", "#6366f1",
  "#14b8a6", "#e11d48", "#a855f7", "#0ea5e9", "#22c55e",
];
