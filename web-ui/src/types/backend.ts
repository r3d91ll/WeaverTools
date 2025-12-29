/**
 * Backend types matching Weaver/pkg/backend/backend.go
 * and Weaver/pkg/api/handlers_backends.go, handlers_models.go
 *
 * Note: Field names use camelCase to match the API JSON responses.
 */

import type { HiddenState } from './measurement';

/**
 * Backend type identifiers.
 * Maps to Type in backend.go.
 */
export type BackendTypeName = 'claudecode' | 'loom';

/**
 * BackendCapabilities describes what a backend can do.
 * Maps to Capabilities in backend.go and BackendCapabilities in handlers_backends.go.
 */
export interface BackendCapabilities {
  contextLimit: number;
  supportsTools: boolean;
  supportsStreaming: boolean;
  supportsHidden: boolean;
  maxTokens: number;
}

/**
 * BackendInfo is the JSON representation of a backend's information.
 * Maps to BackendInfo in handlers_backends.go.
 */
export interface BackendInfo {
  name: string;
  type: BackendTypeName | string;
  available: boolean;
  capabilities: BackendCapabilities;
}

/**
 * BackendListResponse is the JSON response for GET /api/backends.
 */
export interface BackendListResponse {
  backends: BackendInfo[];
}

/**
 * ModelInfo represents information about a model.
 * Maps to ModelInfo in handlers_models.go.
 */
export interface ModelInfo {
  name: string;
  loaded: boolean;
  /** Size in bytes */
  size?: number;
  /** Memory used when loaded */
  memoryUsed?: number;
  /** Which backend owns this model */
  backend?: string;
}

/**
 * ModelListResponse is the JSON response for GET /api/models.
 */
export interface ModelListResponse {
  models: ModelInfo[];
}

/**
 * ModelLoadRequest is the optional JSON request body for POST /api/models/:name/load.
 */
export interface ModelLoadRequest {
  /** Device to load the model on, e.g., "cuda:0", "cpu" */
  device?: string;
}

/**
 * ModelActionResponse is the JSON response for model load/unload operations.
 * Maps to ModelActionResponse in handlers_models.go.
 */
export interface ModelActionResponse {
  name: string;
  action: 'load' | 'unload';
  success: boolean;
  message?: string;
}

/**
 * ChatMessage represents a single message for backend chat requests.
 * Maps to ChatMessage in backend.go.
 */
export interface ChatMessage {
  role: string;
  content: string;
  name?: string;
}

/**
 * TokenUsage tracks token consumption.
 * Maps to TokenUsage in backend.go.
 */
export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

/**
 * ChatRequest contains parameters for a chat request.
 * Maps to ChatRequest in backend.go.
 */
export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  maxTokens?: number;
  temperature?: number;
  stream?: boolean;
  returnHiddenStates?: boolean;
  /** Layer selection: "all", number, or number[] */
  hiddenStateLayers?: 'all' | number | number[];
  /** GPU device: "auto", "cuda:0", "cuda:1", etc. */
  device?: string;
}

/**
 * ChatResponse contains the model's response.
 * Maps to ChatResponse in backend.go.
 */
export interface ChatResponse {
  content: string;
  usage: TokenUsage;
  /** Single layer (legacy) */
  hiddenState?: HiddenState | null;
  /** Multi-layer: layer index -> state */
  hiddenStates?: Record<number, HiddenState>;
  metadata?: Record<string, unknown>;
  latencyMs: number;
  model: string;
  finishReason: string;
}

/**
 * StreamChunk represents a chunk of streamed response.
 * Maps to StreamChunk in backend.go.
 */
export interface StreamChunk {
  content: string;
  done: boolean;
  finishReason?: string;
}

/**
 * BackendConfig holds common backend configuration.
 * Maps to Config in backend.go.
 */
export interface BackendConfig {
  name: string;
  type: BackendTypeName | string;
  url?: string;
  model?: string;
  /** Timeout in milliseconds */
  timeout?: number;
}

/**
 * ResourceStatus represents the status of system resources.
 */
export interface ResourceStatus {
  cpuPercent: number;
  memoryPercent: number;
  memoryUsedMB: number;
  memoryTotalMB: number;
  gpus?: GPUStatus[];
}

/**
 * GPUStatus represents the status of a GPU.
 */
export interface GPUStatus {
  index: number;
  name: string;
  memoryUsedMB: number;
  memoryTotalMB: number;
  memoryPercent: number;
  temperature?: number;
  utilization?: number;
}
