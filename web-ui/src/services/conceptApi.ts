/**
 * Concept API service for WeaverTools web-ui.
 * Handles concept management operations (list, get, delete).
 */

import { get, post, del, ApiError } from './api';
import type {
  ConceptStats,
  ConceptListResponse,
  ConceptDetailResponse,
  ConceptStoreStats,
  ConceptStoreStatsResponse,
  Concept,
} from '@/types/concept';

/** API endpoints for concept operations */
const ENDPOINTS = {
  concepts: '/api/concepts',
  concept: (name: string) => `/api/concepts/${encodeURIComponent(name)}`,
  stats: '/api/concepts/stats',
} as const;

/**
 * List all concepts with their statistics.
 * @returns Promise resolving to array of ConceptStats
 * @throws ApiError on failure
 */
export async function listConcepts(): Promise<ConceptStats[]> {
  const response = await get<ConceptListResponse>(ENDPOINTS.concepts);
  return response.data.concepts;
}

/**
 * Get a specific concept by name.
 * @param name - Concept name
 * @returns Promise resolving to Concept with stats
 * @throws ApiError on failure (404 if not found)
 */
export async function getConcept(name: string): Promise<{ concept: Concept; stats: ConceptStats }> {
  const response = await get<ConceptDetailResponse>(ENDPOINTS.concept(name));
  return {
    concept: response.data.concept,
    stats: response.data.stats,
  };
}

/**
 * Delete a concept.
 * @param name - Concept name
 * @throws ApiError on failure
 */
export async function deleteConcept(name: string): Promise<void> {
  await del<{ success: boolean }>(ENDPOINTS.concept(name));
}

/**
 * Get concept store statistics.
 * @returns Promise resolving to ConceptStoreStats
 * @throws ApiError on failure
 */
export async function getConceptStoreStats(): Promise<ConceptStoreStats> {
  const response = await get<ConceptStoreStatsResponse>(ENDPOINTS.stats);
  return response.data.stats;
}

/**
 * Check if a concept exists.
 * @param name - Concept name
 * @returns Promise resolving to existence status
 */
export async function conceptExists(name: string): Promise<boolean> {
  try {
    await getConcept(name);
    return true;
  } catch (error) {
    if (error instanceof ApiError && error.isNotFound()) {
      return false;
    }
    throw error;
  }
}

/** Concept API service object */
export const conceptApi = {
  listConcepts,
  getConcept,
  deleteConcept,
  getConceptStoreStats,
  conceptExists,
};

export default conceptApi;
