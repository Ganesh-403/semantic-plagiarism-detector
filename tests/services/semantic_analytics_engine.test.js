/**
 * Jest Unit Test Suite for Semantic Analytics Engine
 * Enterprise Unit Test Coverage Asserts Statements & Logic Flow
 */

const { SemanticAnalyticsEngine } = require('../../src/services/semantic_analytics_engine.js');

describe('SemanticAnalyticsEngine Unit Tests', () => {
    let engine;

    beforeEach(() => {
        engine = new SemanticAnalyticsEngine();
    });

    test('should initialize with default configurations', () => {
        expect(engine.similarityThreshold).toBe(0.85);
        expect(engine.minChunkSize).toBe(64);
        expect(engine.embeddingModel).toBe('all-mpnet-base-v2');
        expect(engine.enableCrossLingual).toBe(true);
        expect(engine.flaggedPairs.length).toBeGreaterThan(0);
    });

    test('should correctly compute cosine similarity between identical vectors', () => {
        const vecA = [1.0, 2.0, 3.0, 4.0];
        const vecB = [1.0, 2.0, 3.0, 4.0];
        const sim = engine.calculateCosineSimilarity(vecA, vecB);
        expect(sim).toBeCloseTo(1.0, 5);
    });

    test('should return zero cosine similarity for orthogonal vectors', () => {
        const vecA = [1.0, 0.0];
        const vecB = [0.0, 1.0];
        const sim = engine.calculateCosineSimilarity(vecA, vecB);
        expect(sim).toBeCloseTo(0.0, 5);
    });

    test('should throw error when comparing vectors of mismatched dimensions', () => {
        const vecA = [1.0, 2.0];
        const vecB = [1.0, 2.0, 3.0];
        expect(() => {
            engine.calculateCosineSimilarity(vecA, vecB);
        }).toThrow("Vector dimension mismatch or null vector supplied.");
    });

    test('should correctly compute Jaccard Similarity Index', () => {
        const setA = new Set(['transformer', 'bert', 'attention']);
        const setB = new Set(['transformer', 'gpt', 'attention']);
        const jaccard = engine.calculateJaccardIndex(setA, setB);
        expect(jaccard).toBeCloseTo(0.5, 2);
    });

    test('should evaluate HIGH risk score when similarity exceeds threshold', () => {
        const result = engine.evaluateRiskScore(0.95, 0.80);
        expect(result.risk).toBe('HIGH');
        expect(result.badgeClass).toBe('badge-danger');
    });

    test('should evaluate LOW risk score when similarity is small', () => {
        const result = engine.evaluateRiskScore(0.20, 0.10);
        expect(result.risk).toBe('LOW');
        expect(result.badgeClass).toBe('badge-success');
    });

    test('should update configuration settings dynamically', () => {
        engine.updateConfig({
            similarityThreshold: 0.90,
            minChunkSize: 128,
            embeddingModel: 'text-embedding-ada-002',
            enableCrossLingual: false
        });

        expect(engine.similarityThreshold).toBe(0.90);
        expect(engine.minChunkSize).toBe(128);
        expect(engine.embeddingModel).toBe('text-embedding-ada-002');
        expect(engine.enableCrossLingual).toBe(false);
    });

    test('should filter flagged pairs based on search query', () => {
        const filtered = engine.getFlaggedPairsFiltered('PAIR-9012');
        expect(filtered.length).toBe(1);
        expect(filtered[0].pairId).toBe('PAIR-9012');
    });
});
