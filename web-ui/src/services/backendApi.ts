/**
 * Backend API service for WeaverTools web-ui.
 * Handles backend status and model operations.
 */

import { get, ApiError } from './api';
import type { BackendCapabilities, BackendTypeName } from '@/types';

/** API endpoints for backend operations */
const ENDPOINTS = {
  backends: '/api/backends',
  backend: (name: string) => `/api/backends/${encodeURIComponent(name)}`,
} as const;

/** Backend status information */
export interface BackendStatus {
  name: string;
  type: BackendTypeName | string;
  available: boolean;
  capabilities: BackendCapabilities;
  error?: string;
}

/** Backends list response */
interface BackendsResponse {
  backends: BackendStatus[];
}

/** Single backend response */
interface BackendResponse {
  backend: BackendStatus;
}

/**
 * List all available backends.
 * @returns Promise resolving to array of BackendStatus
 * @throws ApiError on failure
 */
export async function listBackends(): Promise<BackendStatus[]> {
  const response = await get<BackendsResponse>(ENDPOINTS.backends);
  return response.data.backends;
}

/**
 * Get a specific backend by name.
 * @param name - Backend name
 * @returns Promise resolving to BackendStatus
 * @throws ApiError on failure (404 if not found)
 */
export async function getBackend(name: string): Promise<BackendStatus> {
  const response = await get<BackendResponse>(ENDPOINTS.backend(name));
  return response.data.backend;
}

/**
 * Check if a backend is available.
 * @param name - Backend name
 * @returns Promise resolving to availability status
 */
export async function isBackendAvailable(name: string): Promise<boolean> {
  try {
    const backend = await getBackend(name);
    return backend.available;
  } catch (error) {
    if (error instanceof ApiError && error.isNotFound()) {
      return false;
    }
    throw error;
  }
}

/**
 * Get capabilities of a backend.
 * @param name - Backend name
 * @returns Promise resolving to BackendCapabilities
 * @throws ApiError on failure
 */
export async function getBackendCapabilities(name: string): Promise<BackendCapabilities> {
  const backend = await getBackend(name);
  return backend.capabilities;
}

/**
 * Check if streaming is supported by a backend.
 * @param name - Backend name
 * @returns Promise resolving to streaming support status
 */
export async function supportsStreaming(name: string): Promise<boolean> {
  const capabilities = await getBackendCapabilities(name);
  return capabilities.supportsStreaming;
}

/**
 * Check if hidden states are supported by a backend.
 * @param name - Backend name
 * @returns Promise resolving to hidden state support status
 */
export async function supportsHiddenStates(name: string): Promise<boolean> {
  const capabilities = await getBackendCapabilities(name);
  return capabilities.supportsHidden;
}

/** Backend API service object */
export const backendApi = {
  listBackends,
  getBackend,
  isBackendAvailable,
  getBackendCapabilities,
  supportsStreaming,
  supportsHiddenStates,
};

export default backendApi;
