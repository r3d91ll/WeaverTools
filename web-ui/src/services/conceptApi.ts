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
  samples: (name: string) => `/api/concepts/${encodeURIComponent(name)}/samples`,
  stats: '/api/concepts/stats',
} as const;

/**
 * Request body for adding a sample to a concept.
 */
export interface AddSampleRequest {
  /** Sample text content */
  content: string;
  /** Optional model name used to generate the sample */
  model?: string;
  /** Optional hidden state data */
  hiddenState?: {
    vector: number[];
    layer: number;
    tokenIdx: number;
    dtype?: string;
  };
}

/**
 * Response from adding a sample to a concept.
 */
export interface AddSampleResponse {
  id: string;
  conceptName: string;
  content: string;
  model?: string;
  extractedAt: string;
}

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

/**
 * Add a sample to a concept.
 * Creates the concept if it doesn't exist.
 * @param name - Concept name
 * @param sample - Sample data to add
 * @returns Promise resolving to AddSampleResponse
 * @throws ApiError on failure
 */
export async function addSample(
  name: string,
  sample: AddSampleRequest
): Promise<AddSampleResponse> {
  const response = await post<AddSampleResponse>(
    ENDPOINTS.samples(name),
    sample
  );
  return response.data;
}

/**
 * Add multiple samples to a concept.
 * Creates the concept if it doesn't exist.
 * @param name - Concept name
 * @param samples - Array of sample data to add
 * @returns Promise resolving to array of AddSampleResponse
 * @throws ApiError on failure
 */
export async function addSamples(
  name: string,
  samples: AddSampleRequest[]
): Promise<AddSampleResponse[]> {
  const results: AddSampleResponse[] = [];
  for (const sample of samples) {
    const response = await addSample(name, sample);
    results.push(response);
  }
  return results;
}

/** Concept API service object */
export const conceptApi = {
  listConcepts,
  getConcept,
  deleteConcept,
  getConceptStoreStats,
  conceptExists,
  addSample,
  addSamples,
};

export default conceptApi;
