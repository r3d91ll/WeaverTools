/**
 * Type exports for WeaverTools web-ui.
 * Re-exports all types from domain-specific modules.
 */

// Backend types
export type {
  BackendType,
  Capabilities,
  ChatMessage,
  TokenUsage,
  HiddenState,
  ChatRequest,
  ChatResponse,
  StreamChunk,
  BackendConfig,
} from './backend';

// Agent types
export type { AgentRole, AgentConfig } from './agent';
export { INFERENCE_DEFAULTS, applyInferenceDefaults } from './agent';

// Config types
export type {
  MeasurementMode as ConfigMeasurementMode,
  ClaudeCodeConfig,
  LoomConfig,
  BackendsConfig,
  ConfigSessionSettings,
  Config,
} from './config';
export { DEFAULT_CONFIG } from './config';

// Measurement types
export type { BetaStatus, Measurement } from './measurement';
export {
  BETA_THRESHOLDS,
  computeBetaStatus,
  isValidBetaStatus,
  isBilateral,
} from './measurement';

// Session types
export type {
  MessageRole,
  Message,
  Participant,
  Conversation,
  MeasurementMode,
  SessionConfig,
  Session,
  SessionStats,
  SessionStatus,
} from './session';
export {
  isValidMessageRole,
  isValidMeasurementMode,
  calculateSessionStats,
} from './session';
