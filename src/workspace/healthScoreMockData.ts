/**
 * healthScoreMockData.ts
 * -----------------------
 * Mock data generators for the Document Health Scoring dashboard.
 * In production, replace these with actual API calls.
 */

import type {
  DocumentHealthScore,
  HealthScoreSummary,
  HealthGrade,
} from "./healthScoreTypes";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function randomBetween(min: number, max: number): number {
  return Math.round((Math.random() * (max - min) + min) * 10) / 10;
}

function randomGrade(score: number): HealthGrade {
  if (score >= 97) return "A+";
  if (score >= 93) return "A";
  if (score >= 90) return "A-";
  if (score >= 87) return "B+";
  if (score >= 83) return "B";
  if (score >= 80) return "B-";
  if (score >= 70) return "C";
  if (score >= 60) return "D";
  return "F";
}

// ---------------------------------------------------------------------------
// Mock Document Health Scores
// ---------------------------------------------------------------------------

export function generateMockHealthScores(): DocumentHealthScore[] {
  const docs = [
    { name: "neural_nlp_overview.pdf", metaScore: 100, chunkScore: 88, embedScore: 100, contentScore: 92, fpScore: 100 },
    { name: "plagiarism_detection_heuristics.docx", metaScore: 80, chunkScore: 75, embedScore: 100, contentScore: 85, fpScore: 100 },
    { name: "apprentissage_profond_notes.txt", metaScore: 60, chunkScore: 65, embedScore: 100, contentScore: 55, fpScore: 100 },
    { name: "transformer_attention_mechanism.pdf", metaScore: 100, chunkScore: 92, embedScore: 100, contentScore: 95, fpScore: 100 },
    { name: "deep_learning_survey_2026.pdf", metaScore: 100, chunkScore: 85, embedScore: 100, contentScore: 88, fpScore: 100 },
    { name: "suspected_copy_paper_a.pdf", metaScore: 40, chunkScore: 55, embedScore: 80, contentScore: 35, fpScore: 0 },
    { name: "paraphrased_essay.docx", metaScore: 60, chunkScore: 70, embedScore: 100, contentScore: 65, fpScore: 100 },
    { name: "shared_section_report.pdf", metaScore: 80, chunkScore: 60, embedScore: 100, contentScore: 72, fpScore: 50 },
    { name: "original_research_v2.pdf", metaScore: 100, chunkScore: 90, embedScore: 100, contentScore: 91, fpScore: 100 },
    { name: "thesis_draft_final.pdf", metaScore: 100, chunkScore: 88, embedScore: 100, contentScore: 94, fpScore: 100 },
    { name: "incomplete_submission.pdf", metaScore: 20, chunkScore: 30, embedScore: 50, contentScore: 25, fpScore: 100 },
    { name: "duplicate_content_alert.pdf", metaScore: 80, chunkScore: 75, embedScore: 100, contentScore: 70, fpScore: 0 },
    { name: "well_structured_essay.docx", metaScore: 100, chunkScore: 95, embedScore: 100, contentScore: 93, fpScore: 100 },
    { name: "minimal_metadata_doc.pdf", metaScore: 20, chunkScore: 70, embedScore: 100, contentScore: 60, fpScore: 100 },
    { name: "very_short_submission.txt", metaScore: 60, chunkScore: 25, embedScore: 100, contentScore: 20, fpScore: 100 },
  ];

  return docs.map((d, i) => {
    const meta = 0.25;
    const chunk = 0.20;
    const embed = 0.20;
    const content = 0.25;
    const fp = 0.10;
    const overall =
      d.metaScore * meta +
      d.chunkScore * chunk +
      d.embedScore * embed +
      d.contentScore * content +
      d.fpScore * fp;
    const score = Math.round(overall * 10) / 10;
    const grade = randomGrade(score);

    return {
      id: i + 1,
      filename: d.name,
      overall_score: score,
      grade,
      dimensions: [
        { name: "metadata_completeness", score: d.metaScore, weight: meta, weighted: d.metaScore * meta, details: "Fields populated" },
        { name: "chunk_balance", score: d.chunkScore, weight: chunk, weighted: d.chunkScore * chunk, details: "Chunk size distribution" },
        { name: "embedding_coverage", score: d.embedScore, weight: embed, weighted: d.embedScore * embed, details: "Embedding ratio" },
        { name: "content_quality", score: d.contentScore, weight: content, weighted: d.contentScore * content, details: "Content analysis" },
        { name: "fingerprint_uniqueness", score: d.fpScore, weight: fp, weighted: d.fpScore * fp, details: "Hash uniqueness" },
      ],
      checked_at: new Date(Date.now() - i * 3600_000).toISOString(),
      gate_passed: score >= 60,
      gate_reason: score >= 60 ? "All quality checks passed" : `Score ${score.toFixed(1)} below minimum 60.0`,
    };
  });
}

// ---------------------------------------------------------------------------
// Mock Summary
// ---------------------------------------------------------------------------

export function generateMockHealthSummary(): HealthScoreSummary {
  return {
    total_scored: 42,
    avg_score: 78.4,
    min_score: 24.5,
    max_score: 97.2,
    passed_gate: 35,
    failed_gate: 7,
    pass_rate: 83.3,
    grade_distribution: {
      "A+": 3,
      "A": 5,
      "A-": 4,
      "B+": 8,
      "B": 7,
      "B-": 4,
      "C": 5,
      "D": 3,
      "F": 3,
    },
    last_checked_at: new Date(Date.now() - 1800_000).toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Mock Dimension Averages
// ---------------------------------------------------------------------------

export function generateMockDimensionAvgs(): Record<string, number> {
  return {
    metadata_completeness: 72.5,
    chunk_balance: 74.8,
    embedding_coverage: 95.2,
    content_quality: 70.1,
    fingerprint_uniqueness: 88.6,
  };
}
