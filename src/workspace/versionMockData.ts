/**
 * versionMockData.ts
 * Mock data generators for the Document Versioning Dashboard.
 */

import type {
  DocumentVersionSnapshot,
  VersionDiff,
  VersionTrendPoint,
  MostRevisedDoc,
  VersionSummary,
} from "./versionTypes";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function randInt(min: number, max: number): number {
  return Math.floor(Math.random() * (max - min + 1)) + min;
}

function randFloat(min: number, max: number): number {
  return Math.round((Math.random() * (max - min) + min) * 10000) / 10000;
}

function randomHash(): string {
  return Array.from({ length: 16 }, () =>
    Math.floor(Math.random() * 16).toString(16)
  ).join("");
}

function isoDate(daysAgo: number): string {
  const d = new Date();
  d.setDate(d.getDate() - daysAgo);
  return d.toISOString();
}

// ---------------------------------------------------------------------------
// Mock generators
// ---------------------------------------------------------------------------

const USERS = ["alice", "bob", "carol", "dave", "eve"];
const ASSIGNMENTS = ["essay-01", "lab-report-03", "thesis-draft", "midterm-essay", "capstone-v2"];
const FILENAMES = [
  "introduction.docx",
  "methodology.docx",
  "analysis.docx",
  "conclusion.docx",
  "literature-review.docx",
  "discussion.docx",
  "abstract.docx",
  "appendix.docx",
];

export function mockSnapshots(count: number = 25): DocumentVersionSnapshot[] {
  const snapshots: DocumentVersionSnapshot[] = [];

  for (let i = 0; i < count; i++) {
    const userId = USERS[randInt(0, USERS.length - 1)];
    const assignmentId = ASSIGNMENTS[randInt(0, ASSIGNMENTS.length - 1)];
    const versionNum = randInt(1, 6);
    const wordCount = randInt(500, 5000);
    const contentLength = wordCount * randInt(4, 8);

    snapshots.push({
      document_hash: randomHash(),
      user_id: userId,
      assignment_id: assignmentId,
      filename: FILENAMES[randInt(0, FILENAMES.length - 1)],
      content_length: contentLength,
      word_count: wordCount,
      version_number: versionNum,
      parent_hash: versionNum > 1 ? randomHash() : null,
      similarity_to_parent: versionNum > 1 ? randFloat(0.4, 0.98) : null,
      created_at: isoDate(randInt(0, 30)),
    });
  }

  return snapshots.sort(
    (a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
  );
}

export function mockDiffs(count: number = 15): VersionDiff[] {
  const diffs: VersionDiff[] = [];

  for (let i = 0; i < count; i++) {
    const addedWords = randInt(20, 800);
    const removedWords = randInt(10, 400);
    const changedWords = randInt(5, 200);

    diffs.push({
      parent_hash: randomHash(),
      child_hash: randomHash(),
      similarity: randFloat(0.45, 0.95),
      added_words: addedWords,
      removed_words: removedWords,
      changed_words: changedWords,
      jaccard_index: randFloat(0.2, 0.85),
      computed_at: isoDate(randInt(0, 30)),
    });
  }

  return diffs;
}

export function mockTrend(versionCount: number = 5): VersionTrendPoint[] {
  const trend: VersionTrendPoint[] = [];
  let currentSim = randFloat(0.5, 0.85);

  for (let v = 1; v < versionCount; v++) {
    const added = randInt(30, 500);
    const removed = randInt(10, 300);
    currentSim = Math.min(1.0, Math.max(0.3, currentSim + randFloat(-0.15, 0.15)));

    trend.push({
      from_version: v,
      to_version: v + 1,
      similarity: currentSim,
      added_words: added,
      removed_words: removed,
      created_at: isoDate(versionCount - v),
    });
  }

  return trend;
}

export function mockMostRevised(count: number = 8): MostRevisedDoc[] {
  const docs: MostRevisedDoc[] = [];

  for (let i = 0; i < count; i++) {
    docs.push({
      assignment_id: ASSIGNMENTS[randInt(0, ASSIGNMENTS.length - 1)],
      user_id: USERS[randInt(0, USERS.length - 1)],
      total_versions: randInt(3, 12),
      avg_similarity: randFloat(0.35, 0.95),
      last_created: isoDate(randInt(0, 14)),
    });
  }

  return docs.sort((a, b) => b.total_versions - a.total_versions);
}

export function mockSummary(): VersionSummary {
  return {
    total_versions: randInt(40, 200),
    total_lineages: randInt(10, 50),
    total_diffs: randInt(20, 100),
    avg_similarity: randFloat(0.55, 0.85),
    avg_versions_per_document: randFloat(2.0, 5.5),
    unique_users: randInt(3, 15),
  };
}
