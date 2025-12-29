/**
 * Central type exports for the Weaver Web UI.
 * Re-exports all types from individual modules for convenient importing.
 */

// Configuration types
export type {
  BackendType,
  AgentRole,
  MeasurementMode,
  ClaudeCodeConfig,
  LoomConfig,
  BackendsConfig,
  AgentConfig,
  SessionConfig,
  Config,
  ValidationResult,
} from './config';

export { INFERENCE_DEFAULTS, DEFAULT_CONFIG } from './config';

// Agent types
export type {
  AgentInfo,
  AgentListResponse,
  MessageRole,
  ChatHistoryMessage,
  ChatAgentRequest,
  ChatAgentResponse,
  Message,
  Participant,
  Conversation,
} from './agent';

export { isValidMessageRole } from './agent';

// Session types
export type {
  SessionAPIConfig,
  SessionStats,
  Session,
  SessionListResponse,
  CreateSessionRequest,
  UpdateSessionRequest,
  SessionMetricsSummary,
  SessionStartEvent,
  SessionEndEvent,
  SessionStatus,
} from './session';

export { isValidMeasurementMode, calculateSessionStats } from './session';

// Measurement types
export type {
  BetaStatus,
  HiddenState,
  Measurement,
  MeasurementEvent,
  MeasurementBatchEvent,
  BetaAlertEvent,
  ConversationTurnEvent,
  MeasurementData,
} from './measurement';

export {
  BETA_STATUS_RANGES,
  computeBetaStatus,
  isValidBetaStatus,
  getHiddenDimension,
  isBilateral,
} from './measurement';

// Backend types
export type {
  BackendTypeName,
  BackendCapabilities,
  BackendInfo,
  BackendListResponse,
  ModelInfo,
  ModelListResponse,
  ModelLoadRequest,
  ModelActionResponse,
  ChatMessage,
  TokenUsage,
  ChatRequest,
  ChatResponse,
  StreamChunk,
  BackendConfig,
  ResourceStatus,
  GPUStatus,
} from './backend';

// WebSocket message types
export type WebSocketMessageType =
  | 'measurement'
  | 'measurement_batch'
  | 'message'
  | 'backend_status'
  | 'model_status'
  | 'resource_update'
  | 'session_start'
  | 'session_end'
  | 'conversation_turn'
  | 'beta_alert'
  | 'pong'
  | 'error'
  | 'subscribe'
  | 'ping';

export type WebSocketChannel =
  | 'measurements'
  | 'messages'
  | 'status'
  | 'resources';

/**
 * WebSocketMessage is the base structure for all WebSocket messages.
 */
export interface WebSocketMessage<T = unknown> {
  type: WebSocketMessageType;
  data: T;
  channel?: WebSocketChannel;
  timestamp: string;
}

/**
 * API error response structure.
 */
export interface APIError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

/**
 * Generic API response wrapper.
 */
export interface APIResponse<T> {
  data?: T;
  error?: APIError;
}
