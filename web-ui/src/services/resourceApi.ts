/**
 * Resource API service for system resources like GPUs.
 */

import { api } from './api';
import type { GPUInfo } from '../types/config';

/**
 * Response structure for GET /api/resources/gpus
 */
export interface GPUListResponse {
  gpus: GPUInfo[];
  total: number;
}

/**
 * Fetches the list of available GPUs on the system.
 */
export async function getGPUs(): Promise<GPUListResponse> {
  return api.get<GPUListResponse>('/resources/gpus');
}

export const resourceApi = {
  getGPUs,
};
