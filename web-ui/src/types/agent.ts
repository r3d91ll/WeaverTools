/**
 * Agent types matching Weaver/pkg/config/config.go
 * Configuration for AI agents in the multi-agent system.
 */

import type { BackendType } from './backend';

/** Valid agent roles in the system. */
export type AgentRole = 'senior' | 'junior' | 'analyst' | 'architect' | 'reviewer';

/** AgentConfig holds agent settings. */
export interface AgentConfig {
  role: AgentRole;
  backend: BackendType | string;
  model: string;
  system_prompt: string;
  tools: string[];
  tools_enabled: boolean;
  /** Whether agent is active for this session */
  active: boolean;

  // Inference parameters (for Loom backend)
  max_tokens: number;
  /** Temperature is optional to distinguish "not set" from "explicitly 0" */
  temperature?: number;
  context_length: number;
  /** TopP is optional to distinguish "not set" from "explicitly 0" */
  top_p?: number;
  top_k: number;

  /**
   * GPU assignment (for Loom backend)
   * "auto" = let Loom decide, "0" = cuda:0, "1" = cuda:1, etc.
   */
  gpu: string;
}

/** Default inference parameter values. */
export const INFERENCE_DEFAULTS = {
  maxTokens: 2048,
  temperature: 0.7,
  contextLength: 32768,
  topP: 0.9,
} as const;

/** Apply default inference parameters to an agent config. */
export function applyInferenceDefaults(config: Partial<AgentConfig>): AgentConfig {
  return {
    role: 'junior',
    backend: 'loom',
    model: '',
    system_prompt: '',
    tools: [],
    tools_enabled: false,
    active: false,
    max_tokens: config.max_tokens || INFERENCE_DEFAULTS.maxTokens,
    temperature: config.temperature ?? INFERENCE_DEFAULTS.temperature,
    context_length: config.context_length || INFERENCE_DEFAULTS.contextLength,
    top_p: config.top_p ?? INFERENCE_DEFAULTS.topP,
    top_k: config.top_k || 0,
    gpu: config.gpu || 'auto',
    ...config,
  };
}
