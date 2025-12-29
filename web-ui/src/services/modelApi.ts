/**
 * Model API service for WeaverTools web-ui.
 * Handles model management operations (list, load, unload).
 */

import { get, post, ApiError, withRetry } from './api';

/** API endpoints for model operations */
const ENDPOINTS = {
  models: '/api/models',
  model: (name: string) => `/api/models/${encodeURIComponent(name)}`,
  load: (name: string) => `/api/models/${encodeURIComponent(name)}/load`,
  unload: (name: string) => `/api/models/${encodeURIComponent(name)}/unload`,
} as const;

/** Model information */
export interface ModelInfo {
  name: string;
  loaded: boolean;
  size: number; // in bytes
  memoryUsed: number; // in bytes
  backend: string;
  path?: string;
  parameters?: number; // model parameter count
  quantization?: string; // e.g., "Q4_K_M"
}

/** Model load options */
export interface ModelLoadOptions {
  /** GPU device to load on */
  device?: string;
  /** Number of GPU layers */
  gpuLayers?: number;
  /** Context size override */
  contextSize?: number;
}

/** Models list response */
interface ModelsResponse {
  models: ModelInfo[];
}

/** Single model response */
interface ModelResponse {
  model: ModelInfo;
}

/** Load operation response */
interface LoadResponse {
  success: boolean;
  model: ModelInfo;
  loadTime: number; // in milliseconds
}

/** Unload operation response */
interface UnloadResponse {
  success: boolean;
  memoryFreed: number; // in bytes
}

/**
 * List all available models.
 * @returns Promise resolving to array of ModelInfo
 * @throws ApiError on failure
 */
export async function listModels(): Promise<ModelInfo[]> {
  const response = await get<ModelsResponse>(ENDPOINTS.models);
  return response.data.models;
}

/**
 * Get a specific model by name.
 * @param name - Model name
 * @returns Promise resolving to ModelInfo
 * @throws ApiError on failure (404 if not found)
 */
export async function getModel(name: string): Promise<ModelInfo> {
  const response = await get<ModelResponse>(ENDPOINTS.model(name));
  return response.data.model;
}

/**
 * Load a model.
 * @param name - Model name
 * @param options - Load options
 * @returns Promise resolving to loaded ModelInfo
 * @throws ApiError on failure
 */
export async function loadModel(
  name: string,
  options?: ModelLoadOptions
): Promise<{ model: ModelInfo; loadTime: number }> {
  const response = await post<LoadResponse>(ENDPOINTS.load(name), options ?? {}, {
    timeout: 120000, // 2 minutes for model loading
  });
  return {
    model: response.data.model,
    loadTime: response.data.loadTime,
  };
}

/**
 * Unload a model.
 * @param name - Model name
 * @returns Promise resolving to memory freed in bytes
 * @throws ApiError on failure
 */
export async function unloadModel(name: string): Promise<number> {
  const response = await post<UnloadResponse>(ENDPOINTS.unload(name));
  return response.data.memoryFreed;
}

/**
 * Check if a model is loaded.
 * @param name - Model name
 * @returns Promise resolving to loaded status
 */
export async function isModelLoaded(name: string): Promise<boolean> {
  try {
    const model = await getModel(name);
    return model.loaded;
  } catch (error) {
    if (error instanceof ApiError && error.isNotFound()) {
      return false;
    }
    throw error;
  }
}

/**
 * Get loaded models.
 * @returns Promise resolving to array of loaded ModelInfo
 */
export async function getLoadedModels(): Promise<ModelInfo[]> {
  const models = await listModels();
  return models.filter((m) => m.loaded);
}

/**
 * Get available (not loaded) models.
 * @returns Promise resolving to array of available ModelInfo
 */
export async function getAvailableModels(): Promise<ModelInfo[]> {
  const models = await listModels();
  return models.filter((m) => !m.loaded);
}

/**
 * Format model size as human-readable string.
 * @param bytes - Size in bytes
 * @returns Formatted size string (e.g., "7.5 GB")
 */
export function formatModelSize(bytes: number): string {
  if (bytes === 0) return '0 B';

  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const k = 1024;
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  const value = bytes / Math.pow(k, i);

  return `${value.toFixed(i > 0 ? 1 : 0)} ${units[i]}`;
}

/**
 * Format parameter count as human-readable string.
 * @param params - Number of parameters
 * @returns Formatted parameter string (e.g., "7B", "13B")
 */
export function formatParameterCount(params: number): string {
  if (params >= 1e12) {
    return `${(params / 1e12).toFixed(1)}T`;
  }
  if (params >= 1e9) {
    return `${(params / 1e9).toFixed(1)}B`;
  }
  if (params >= 1e6) {
    return `${(params / 1e6).toFixed(1)}M`;
  }
  return String(params);
}

/**
 * Load a model with retry on failure.
 * @param name - Model name
 * @param options - Load options
 * @param retries - Number of retries (default: 2)
 * @returns Promise resolving to loaded ModelInfo
 */
export async function loadModelWithRetry(
  name: string,
  options?: ModelLoadOptions,
  retries = 2
): Promise<{ model: ModelInfo; loadTime: number }> {
  return withRetry(() => loadModel(name, options), retries);
}

/** Model API service object */
export const modelApi = {
  listModels,
  getModel,
  loadModel,
  unloadModel,
  isModelLoaded,
  getLoadedModels,
  getAvailableModels,
  formatModelSize,
  formatParameterCount,
  loadModelWithRetry,
};

export default modelApi;
