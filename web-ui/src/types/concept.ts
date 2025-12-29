/**
 * Concept types matching Weaver/pkg/concepts/store.go
 *
 * Note: Field names use camelCase to match the API JSON responses.
 */

/**
 * Sample represents a single extracted sample for a concept.
 * Maps to Sample in store.go.
 */
export interface ConceptSample {
  id: string;
  content: string;
  extractedAt: string;
  model?: string;
  hiddenState?: {
    vector: number[];
    layer: number;
    tokenIndex: number;
  } | null;
}

/**
 * Concept holds all samples for a named concept.
 * Maps to Concept in store.go.
 */
export interface Concept {
  name: string;
  samples: ConceptSample[];
  createdAt: string;
  updatedAt: string;
}

/**
 * ConceptStats holds detailed statistics for a single concept.
 * Maps to ConceptStats in store.go.
 */
export interface ConceptStats {
  name: string;
  sampleCount: number;
  dimension: number;
  mismatchedIds?: string[];
  createdAt: string;
  updatedAt: string;
  models?: string[];
  oldestSampleAt?: string;
  newestSampleAt?: string;
}

/**
 * StoreStats holds aggregate statistics for the entire concept store.
 * Maps to StoreStats in store.go.
 */
export interface ConceptStoreStats {
  conceptCount: number;
  totalSamples: number;
  dimensions: Record<number, number>;
  models: Record<string, number>;
  healthyConcepts: number;
  conceptsWithIssues: number;
  oldestExtraction?: string;
  newestExtraction?: string;
  concepts: Record<string, ConceptStats>;
}

/**
 * ConceptListResponse is the JSON response for GET /api/concepts.
 */
export interface ConceptListResponse {
  concepts: ConceptStats[];
}

/**
 * ConceptDetailResponse is the JSON response for GET /api/concepts/:id.
 */
export interface ConceptDetailResponse {
  concept: Concept;
  stats: ConceptStats;
}

/**
 * ConceptStoreStatsResponse is the JSON response for GET /api/concepts/stats.
 */
export interface ConceptStoreStatsResponse {
  stats: ConceptStoreStats;
}

/**
 * Format dimension as human-readable string.
 */
export function formatDimension(dimension: number): string {
  if (dimension === 0) return 'N/A';
  if (dimension >= 1000) {
    return `${(dimension / 1000).toFixed(1)}k`;
  }
  return String(dimension);
}

/**
 * Get health status for a concept based on mismatched IDs.
 */
export function getConceptHealth(stats: ConceptStats): 'healthy' | 'warning' | 'error' {
  if (!stats.mismatchedIds || stats.mismatchedIds.length === 0) {
    return 'healthy';
  }
  if (stats.mismatchedIds.length < stats.sampleCount / 2) {
    return 'warning';
  }
  return 'error';
}

/**
 * Format relative time from a date string.
 */
export function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSec = Math.floor(diffMs / 1000);
  const diffMin = Math.floor(diffSec / 60);
  const diffHour = Math.floor(diffMin / 60);
  const diffDay = Math.floor(diffHour / 24);

  if (diffDay > 0) {
    return diffDay === 1 ? '1 day ago' : `${diffDay} days ago`;
  }
  if (diffHour > 0) {
    return diffHour === 1 ? '1 hour ago' : `${diffHour} hours ago`;
  }
  if (diffMin > 0) {
    return diffMin === 1 ? '1 minute ago' : `${diffMin} minutes ago`;
  }
  return 'just now';
}
