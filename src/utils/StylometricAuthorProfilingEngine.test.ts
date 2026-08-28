import { describe, it, expect } from 'vitest';
import {
  extractStylometricProfile,
  calculateStylometricDistance,
  verifyAuthorshipMatch,
} from './StylometricAuthorProfilingEngine';

describe('StylometricAuthorProfilingEngine', () => {
  const textSample1 = `The rapid advancement of artificial intelligence models has profoundly altered modern computing. Furthermore, neural network architectures continue to evolve at an unprecedented speed, enabling complex pattern recognition and semantic text analysis across diverse domain datasets.`;

  const textSample2 = `AI technology is moving fast and changing how software works! However, we must be careful about data quality, training efficiency, and model evaluation metrics to prevent unexpected errors.`;

  it('should extract comprehensive stylometric profile metrics', () => {
    const profile = extractStylometricProfile(textSample1);

    expect(profile).toBeDefined();
    expect(profile.totalWords).toBeGreaterThan(20);
    expect(profile.totalSentences).toBeGreaterThan(0);
    expect(profile.averageSentenceLength).toBeGreaterThan(5);
    expect(profile.typeTokenRatio).toBeGreaterThan(0);
    expect(profile.punctuationDensity).toBeGreaterThanOrEqual(0);
    expect(profile.hapaxLegomenaRatio).toBeGreaterThan(0);
  });

  it('should calculate stylometric distance between two text samples', () => {
    const profileA = extractStylometricProfile(textSample1);
    const profileB = extractStylometricProfile(textSample2);

    const distance = calculateStylometricDistance(profileA, profileB);

    expect(distance).toBeGreaterThanOrEqual(0);
    expect(distance).toBeLessThanOrEqual(1.0);
  });

  it('should verify authorship match and return similarity score', () => {
    const verification = verifyAuthorshipMatch(textSample1, textSample1);

    expect(verification.isSameAuthor).toBe(true);
    expect(verification.authorshipProbability).toBeGreaterThan(0.8);
    expect(verification.stylometricDistance).toBeLessThan(0.2);
  });
});
