/**
 * heatmapMockData.ts
 * -------------------
 * Mock data generators for the Similarity Heatmap & Clustering dashboard.
 */

import type {
  HeatmapSnapshot,
  ClusteringResult,
  SimilarityHotspot,
  HotspotSummary,
} from "./heatmapTypes";

// ---------------------------------------------------------------------------
// Mock Similarity Matrix (8 documents)
// ---------------------------------------------------------------------------

const DOC_NAMES = [
  "neural_nlp_overview.pdf",
  "plagiarism_heuristics.docx",
  "deep_learning_survey.pdf",
  "transformer_attention.pdf",
  "suspected_copy_a.pdf",
  "paraphrased_essay.docx",
  "original_research.pdf",
  "thesis_final.pdf",
];

const MOCK_MATRIX: number[][] = [
  [1.0,  0.23, 0.78, 0.82, 0.15, 0.31, 0.65, 0.55],
  [0.23, 1.0,  0.18, 0.12, 0.87, 0.74, 0.20, 0.10],
  [0.78, 0.18, 1.0,  0.91, 0.11, 0.22, 0.70, 0.62],
  [0.82, 0.12, 0.91, 1.0,  0.08, 0.19, 0.68, 0.58],
  [0.15, 0.87, 0.11, 0.08, 1.0,  0.72, 0.13, 0.05],
  [0.31, 0.74, 0.22, 0.19, 0.72, 1.0,  0.25, 0.18],
  [0.65, 0.20, 0.70, 0.68, 0.13, 0.25, 1.0,  0.88],
  [0.55, 0.10, 0.62, 0.58, 0.05, 0.18, 0.88, 1.0],
];

export function generateMockSnapshot(): HeatmapSnapshot {
  return {
    snapshot_id: 12,
    labels: DOC_NAMES,
    matrix: MOCK_MATRIX,
    document_count: DOC_NAMES.length,
    min_similarity: 0.05,
    max_similarity: 0.91,
    mean_similarity: 0.38,
    computed_at: new Date(Date.now() - 3600_000).toISOString(),
    computed_by: "prof_jackson",
    notes: "Weekly corpus similarity scan",
    hotspots_found: 5,
  };
}

// ---------------------------------------------------------------------------
// Mock Clustering
// ---------------------------------------------------------------------------

export function generateMockClustering(): ClusteringResult {
  return {
    result_id: 8,
    num_clusters: 3,
    silhouette_score: 0.612,
    linkage_method: "single",
    distance_threshold: 0.5,
    clusters: [
      {
        cluster_id: 0,
        documents: ["neural_nlp_overview.pdf", "deep_learning_survey.pdf", "transformer_attention.pdf"],
        centroid_score: 0.837,
        size: 3,
      },
      {
        cluster_id: 1,
        documents: ["plagiarism_heuristics.docx", "suspected_copy_a.pdf", "paraphrased_essay.docx"],
        centroid_score: 0.777,
        size: 3,
      },
      {
        cluster_id: 2,
        documents: ["original_research.pdf", "thesis_final.pdf"],
        centroid_score: 0.88,
        size: 2,
      },
    ],
    document_assignments: {
      "neural_nlp_overview.pdf": 0,
      "deep_learning_survey.pdf": 0,
      "transformer_attention.pdf": 0,
      "plagiarism_heuristics.docx": 1,
      "suspected_copy_a.pdf": 1,
      "paraphrased_essay.docx": 1,
      "original_research.pdf": 2,
      "thesis_final.pdf": 2,
    },
    computed_at: new Date(Date.now() - 3600_000).toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Mock Hotspots
// ---------------------------------------------------------------------------

export function generateMockHotspots(): SimilarityHotspot[] {
  const now = Date.now();
  return [
    {
      hotspot_id: 1,
      snapshot_id: 12,
      doc_a: "transformer_attention.pdf",
      doc_b: "deep_learning_survey.pdf",
      similarity: 0.91,
      severity: "critical",
      created_at: new Date(now - 3600_000).toISOString(),
      is_resolved: 0,
    },
    {
      hotspot_id: 2,
      snapshot_id: 12,
      doc_a: "thesis_final.pdf",
      doc_b: "original_research.pdf",
      similarity: 0.88,
      severity: "critical",
      created_at: new Date(now - 3600_000).toISOString(),
      is_resolved: 0,
    },
    {
      hotspot_id: 3,
      snapshot_id: 12,
      doc_a: "suspected_copy_a.pdf",
      doc_b: "plagiarism_heuristics.docx",
      similarity: 0.87,
      severity: "critical",
      created_at: new Date(now - 3600_000).toISOString(),
      is_resolved: 0,
    },
    {
      hotspot_id: 4,
      snapshot_id: 12,
      doc_a: "paraphrased_essay.docx",
      doc_b: "plagiarism_heuristics.docx",
      similarity: 0.74,
      severity: "warning",
      created_at: new Date(now - 3600_000).toISOString(),
      is_resolved: 0,
    },
    {
      hotspot_id: 5,
      snapshot_id: 12,
      doc_a: "suspected_copy_a.pdf",
      doc_b: "paraphrased_essay.docx",
      similarity: 0.72,
      severity: "warning",
      created_at: new Date(now - 3600_000).toISOString(),
      is_resolved: 1,
    },
  ];
}

// ---------------------------------------------------------------------------
// Mock Hotspot Summary
// ---------------------------------------------------------------------------

export function generateMockHotspotSummary(): HotspotSummary {
  return {
    total_hotspots: 5,
    unresolved: 4,
    critical_unresolved: 3,
    avg_similarity: 0.824,
  };
}

// ---------------------------------------------------------------------------
// Mock Snapshot History
// ---------------------------------------------------------------------------

export function generateMockSnapshotHistory(): HeatmapSnapshot[] {
  const now = Date.now();
  return [
    {
      snapshot_id: 12,
      document_count: 8,
      min_similarity: 0.05,
      max_similarity: 0.91,
      mean_similarity: 0.38,
      computed_at: new Date(now - 3600_000).toISOString(),
      hotspots_found: 5,
    },
    {
      snapshot_id: 11,
      document_count: 8,
      min_similarity: 0.03,
      max_similarity: 0.89,
      mean_similarity: 0.35,
      computed_at: new Date(now - 86400_000).toISOString(),
      hotspots_found: 4,
    },
    {
      snapshot_id: 10,
      document_count: 7,
      min_similarity: 0.04,
      max_similarity: 0.85,
      mean_similarity: 0.32,
      computed_at: new Date(now - 2 * 86400_000).toISOString(),
      hotspots_found: 3,
    },
    {
      snapshot_id: 9,
      document_count: 6,
      min_similarity: 0.02,
      max_similarity: 0.82,
      mean_similarity: 0.29,
      computed_at: new Date(now - 3 * 86400_000).toISOString(),
      hotspots_found: 2,
    },
  ];
}
