/**
 * Config API service for WeaverTools web-ui.
 * Handles configuration management operations.
 */

import { get, put, post, ApiError } from './api';
import type { Config } from '@/types';

/** API endpoints for config operations */
const ENDPOINTS = {
  config: '/api/config',
  validate: '/api/config/validate',
} as const;

/** Config validation result */
export interface ConfigValidationResult {
  valid: boolean;
  errors: ConfigValidationError[];
}

/** Single validation error */
export interface ConfigValidationError {
  field: string;
  message: string;
}

/** Config API response structure */
interface ConfigResponse {
  config: Config;
}

/** Config validation API response structure */
interface ValidationResponse {
  valid: boolean;
  errors?: Array<{ field: string; message: string }>;
}

/**
 * Get the current configuration.
 * @returns Promise resolving to Config object
 * @throws ApiError on failure
 */
export async function getConfig(): Promise<Config> {
  const response = await get<ConfigResponse>(ENDPOINTS.config);
  return response.data.config;
}

/**
 * Update the configuration.
 * @param config - New configuration to save
 * @returns Promise resolving to updated Config
 * @throws ApiError on failure (including validation errors)
 */
export async function updateConfig(config: Config): Promise<Config> {
  const response = await put<ConfigResponse>(ENDPOINTS.config, { config });
  return response.data.config;
}

/**
 * Validate a configuration without saving it.
 * @param config - Configuration to validate
 * @returns Promise resolving to validation result
 */
export async function validateConfig(config: Config): Promise<ConfigValidationResult> {
  try {
    const response = await post<ValidationResponse>(ENDPOINTS.validate, { config });
    return {
      valid: response.data.valid,
      errors: response.data.errors ?? [],
    };
  } catch (error) {
    // Handle validation error responses (400)
    if (error instanceof ApiError && error.isValidationError()) {
      const body = error.body as ValidationResponse | undefined;
      return {
        valid: false,
        errors: body?.errors ?? [
          { field: 'config', message: error.message },
        ],
      };
    }
    throw error;
  }
}

/**
 * Get a specific agent configuration by name.
 * @param config - Full config object
 * @param agentName - Name of the agent
 * @returns Agent config or undefined if not found
 */
export function getAgentFromConfig(
  config: Config,
  agentName: string
): Config['agents'][string] | undefined {
  return config.agents[agentName];
}

/**
 * Update a specific agent in the configuration.
 * Does not persist - call updateConfig to save.
 * @param config - Current config object
 * @param agentName - Name of the agent to update
 * @param agentConfig - New agent configuration
 * @returns Updated config object (new reference)
 */
export function updateAgentInConfig(
  config: Config,
  agentName: string,
  agentConfig: Config['agents'][string]
): Config {
  return {
    ...config,
    agents: {
      ...config.agents,
      [agentName]: agentConfig,
    },
  };
}

/**
 * Add a new agent to the configuration.
 * Does not persist - call updateConfig to save.
 * @param config - Current config object
 * @param agentName - Name for the new agent
 * @param agentConfig - Agent configuration
 * @returns Updated config object (new reference)
 */
export function addAgentToConfig(
  config: Config,
  agentName: string,
  agentConfig: Config['agents'][string]
): Config {
  if (config.agents[agentName]) {
    throw new Error(`Agent "${agentName}" already exists`);
  }
  return updateAgentInConfig(config, agentName, agentConfig);
}

/**
 * Remove an agent from the configuration.
 * Does not persist - call updateConfig to save.
 * @param config - Current config object
 * @param agentName - Name of the agent to remove
 * @returns Updated config object (new reference)
 */
export function removeAgentFromConfig(config: Config, agentName: string): Config {
  const { [agentName]: removed, ...remainingAgents } = config.agents;
  if (!removed) {
    throw new Error(`Agent "${agentName}" not found`);
  }
  return {
    ...config,
    agents: remainingAgents,
  };
}

/**
 * Get list of agent names from config.
 * @param config - Config object
 * @returns Array of agent names
 */
export function getAgentNames(config: Config): string[] {
  return Object.keys(config.agents);
}

/**
 * Get list of active agents from config.
 * @param config - Config object
 * @returns Array of [name, config] tuples for active agents
 */
export function getActiveAgents(
  config: Config
): Array<[string, Config['agents'][string]]> {
  return Object.entries(config.agents).filter(([, agent]) => agent.active);
}

/** Config API service object */
export const configApi = {
  getConfig,
  updateConfig,
  validateConfig,
  getAgentFromConfig,
  updateAgentInConfig,
  addAgentToConfig,
  removeAgentFromConfig,
  getAgentNames,
  getActiveAgents,
};

export default configApi;
