/**
 * Backend types matching Weaver/pkg/backend/backend.go
 * Provides unified interface types for model communication.
 */

/** Type identifies the backend type. */
export type BackendType = 'claudecode' | 'loom';

/** Capabilities describes what a backend can do. */
export interface Capabilities {
  contextLimit: number;
  supportsTools: boolean;
  supportsStreaming: boolean;
  supportsHidden: boolean;
  maxTokens: number;
}

/** ChatMessage represents a single message. */
export interface ChatMessage {
  role: string;
  content: string;
  name?: string;
}

/** TokenUsage tracks token consumption. */
export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

/** HiddenState represents the boundary object - semantic state before text projection. */
export interface HiddenState {
  /** Hidden state vector, typically 2048-8192 float32 values */
  vector: number[];
  /** Original tensor shape, e.g., [1, seq_len, hidden_dim] */
  shape: number[];
  /** Layer index this state was extracted from */
  layer: number;
  /** Data type, typically "float32" */
  dtype: string;
}

/** ChatRequest contains parameters for a chat request. */
export interface ChatRequest {
  model: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  return_hidden_states?: boolean;
  /** Layer selection: "all", number, or number[] */
  hidden_state_layers?: string | number | number[];
  /** GPU device: "auto", "cuda:0", "cuda:1", etc. */
  device?: string;
}

/** ChatResponse contains the model's response. */
export interface ChatResponse {
  content: string;
  usage: TokenUsage;
  /** Single layer (legacy) */
  hidden_state?: HiddenState;
  /** Multi-layer: layer index -> state */
  hidden_states?: Record<number, HiddenState>;
  metadata?: Record<string, unknown>;
  latency_ms: number;
  model: string;
  finish_reason: string;
}

/** StreamChunk represents a chunk of streamed response. */
export interface StreamChunk {
  content: string;
  done: boolean;
  finish_reason?: string;
}

/** BackendConfig holds common backend configuration. */
export interface BackendConfig {
  name: string;
  type: BackendType;
  url?: string;
  model?: string;
  /** Timeout in milliseconds */
  timeout?: number;
}
