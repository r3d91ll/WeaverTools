/**
 * Configuration types matching Weaver/pkg/config/config.go
 * and API response types from Weaver/pkg/api/handlers_config.go
 *
 * Note: Field names use camelCase to match the API JSON responses.
 */

// Valid backend names
export type BackendType = 'claudecode' | 'loom';

// Valid agent roles
export type AgentRole = 'senior' | 'junior' | 'analyst' | 'architect' | 'reviewer';

// Valid measurement modes
export type MeasurementMode = 'active' | 'passive' | 'triggered' | 'disabled';

/**
 * ClaudeCodeConfig holds Claude Code backend settings.
 * Maps to ClaudeCodeResponse in handlers_config.go.
 */
export interface ClaudeCodeConfig {
  enabled: boolean;
}

/**
 * LoomConfig holds The Loom backend settings.
 * Maps to LoomResponse in handlers_config.go.
 */
export interface LoomConfig {
  enabled: boolean;
  url: string;
  /** Path to TheLoom directory (for auto-start) */
  path: string;
  /** Start TheLoom if not running */
  autoStart: boolean;
  /** Port for TheLoom server */
  port: number;
  /** GPU device IDs to use (e.g., [0, 1]). Empty = auto-detect all */
  gpus?: number[];
}

/**
 * GPUInfo represents information about a single GPU.
 * Maps to GPUInfo in handlers_resources.go.
 */
export interface GPUInfo {
  index: number;
  name: string;
  memoryTotal: number;
  memoryFree: number;
  memoryUsed: number;
  utilization: number;
  available: boolean;
}

/**
 * BackendsConfig holds backend settings.
 * Maps to BackendsResponse in handlers_config.go.
 */
export interface BackendsConfig {
  claudeCode: ClaudeCodeConfig;
  loom: LoomConfig;
}

/**
 * AgentConfig holds agent settings.
 * Maps to AgentResponse in handlers_config.go.
 */
export interface AgentConfig {
  role: AgentRole | string;
  backend: BackendType | string;
  model?: string;
  systemPrompt: string;
  tools?: string[];
  toolsEnabled: boolean;
  /** Whether agent is active for this session */
  active: boolean;
  maxTokens?: number;
  /** Temperature is optional to distinguish "not set" from "explicitly 0" */
  temperature?: number | null;
  contextLength?: number;
  /** TopP is optional to distinguish "not set" from "explicitly 0" */
  topP?: number | null;
  topK?: number;
  /**
   * GPU assignment (for Loom backend)
   * "auto" = let Loom decide, "0" = cuda:0, "1" = cuda:1, etc.
   */
  gpu?: string;
}

/**
 * SessionConfig holds session settings.
 * Maps to SessionResponse in handlers_config.go.
 */
export interface SessionConfig {
  measurementMode: MeasurementMode | string;
  autoExport: boolean;
  exportPath: string;
}

/**
 * Config is the root configuration structure.
 * Maps to ConfigResponse in handlers_config.go.
 */
export interface Config {
  backends: BackendsConfig;
  agents: Record<string, AgentConfig>;
  session: SessionConfig;
}

/**
 * ValidationResult is the response for config validation.
 * Maps to ValidationResult in handlers_config.go.
 */
export interface ValidationResult {
  valid: boolean;
  errors?: string[];
}

/** Default inference parameter values. */
export const INFERENCE_DEFAULTS = {
  maxTokens: 2048,
  temperature: 0.7,
  contextLength: 32768,
  topP: 0.9,
} as const;

/**
 * Default configuration values for creating new configs.
 */
export const DEFAULT_CONFIG: Config = {
  backends: {
    claudeCode: {
      enabled: true,
    },
    loom: {
      enabled: true,
      url: 'http://localhost:8080',
      path: '../TheLoom/the-loom',
      autoStart: true,
      port: 8080,
      gpus: [], // Empty = auto-detect all available GPUs
    },
  },
  agents: {
    senior: {
      role: 'senior',
      backend: 'claudecode',
      active: true,
      systemPrompt: `You are the Senior Engineer in a multi-agent AI research system.
Your role is to handle complex reasoning, architecture decisions, and orchestration.
You can interact with other agents using @agent <message>.`,
      toolsEnabled: true,
    },
    junior: {
      role: 'junior',
      backend: 'loom',
      model: 'Qwen/Qwen2.5-Coder-7B-Instruct',
      active: true,
      systemPrompt: `You are the Junior Engineer in a multi-agent AI research system.
Your role is to handle implementation tasks, file operations, and routine work.
You have access to tools for file manipulation and command execution.`,
      tools: ['read_file', 'write_file', 'list_directory', 'execute_command', 'search_files', 'context_read', 'context_write'],
      toolsEnabled: true,
      maxTokens: 2048,
      temperature: 0.7,
      contextLength: 32768,
      topP: 0.9,
    },
  },
  session: {
    measurementMode: 'active',
    autoExport: true,
    exportPath: './experiments',
  },
};
