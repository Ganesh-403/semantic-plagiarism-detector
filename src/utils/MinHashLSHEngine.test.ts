import { describe, it, expect, beforeEach } from 'vitest';
import { MinHashLSHEngine } from './MinHashLSHEngine';

describe('MinHashLSHEngine', () => {
  let lshEngine: MinHashLSHEngine;

  beforeEach(() => {
    lshEngine = new MinHashLSHEngine({
      numPermutations: 64,
      shingleSize: 3,
      numBands: 8,
      rowsPerBand: 8
    });
  });

  it('should extract shingles correctly from text', () => {
    const text = "The quick brown fox jumps over the lazy dog";
    const shingles = lshEngine.extractShingles(text);
    expect(shingles.size).toBeGreaterThan(0);
  });

  it('should compute valid minhash signatures', () => {
    const text = "Semantic plagiarism detection using minhash and locality sensitive hashing algorithms.";
    const shingles = lshEngine.extractShingles(text);
    const signature = lshEngine.computeMinHashSignature(shingles);
    expect(signature.length).toBe(64);
    signature.forEach(val => expect(val).toBeGreaterThanOrEqual(0));
  });

  it('should detect identical documents with 1.0 estimated Jaccard similarity', () => {
    const text = "High performance text similarity analysis using LSH shingling pipeline.";
    lshEngine.indexDocument("doc1", text);
    lshEngine.indexDocument("doc2", text);

    const candidates = lshEngine.findCandidatePairs();
    expect(candidates.length).toBe(1);
    expect(candidates[0].docIdA).toBe("doc1");
    expect(candidates[0].docIdB).toBe("doc2");
    expect(candidates[0].estimatedJaccard).toBe(1.0);
  });

  it('should estimate lower similarity for distinct texts', () => {
    const textA = "Quantum computing leverages superposition and entanglement to perform complex operations.";
    const textB = "Baking sourdough bread requires flour water salt and wild yeast fermentation.";

    lshEngine.indexDocument("docA", textA);
    lshEngine.indexDocument("docB", textB);

    const sigA = lshEngine.getFingerprint("docA")!.signature;
    const sigB = lshEngine.getFingerprint("docB")!.signature;
    const similarity = lshEngine.estimateJaccard(sigA, sigB);
    expect(similarity).toBeLessThan(0.3);
  });
});
