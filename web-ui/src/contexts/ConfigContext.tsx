/**
 * ConfigContext provides global configuration state management.
 *
 * Features:
 * - Load configuration from backend on mount
 * - Save configuration changes to backend
 * - Validate configuration before saving
 * - Update individual agent configurations
 * - Track loading and error states
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useMemo,
  type ReactNode,
} from 'react';

import type { Config, AgentConfig } from '@/types';
import {
  getConfig,
  updateConfig,
  validateConfig,
  type ConfigValidationResult,
} from '@/services/configApi';

/**
 * ConfigContextValue provides access to configuration state and actions.
 */
export interface ConfigContextValue {
  /** Current configuration, null if not loaded */
  config: Config | null;
  /** Whether configuration is being loaded */
  isLoading: boolean;
  /** Whether configuration is being saved */
  isSaving: boolean;
  /** Current error message, if any */
  error: string | null;
  /** Last validation result */
  validationResult: ConfigValidationResult | null;
  /** Load configuration from backend */
  loadConfig: () => Promise<void>;
  /** Save configuration to backend */
  saveConfig: (newConfig?: Config) => Promise<boolean>;
  /** Validate configuration without saving */
  validateCurrentConfig: (configToValidate?: Config) => Promise<ConfigValidationResult>;
  /** Update a specific agent in the configuration (local state only) */
  updateAgent: (agentName: string, agentConfig: AgentConfig) => void;
  /** Update the entire configuration (local state only) */
  setConfig: (config: Config) => void;
  /** Add a new agent to the configuration (local state only) */
  addAgent: (agentName: string, agentConfig: AgentConfig) => void;
  /** Remove an agent from the configuration (local state only) */
  removeAgent: (agentName: string) => void;
  /** Reset configuration to last saved state */
  resetConfig: () => void;
  /** Check if configuration has unsaved changes */
  hasUnsavedChanges: boolean;
  /** Clear the current error */
  clearError: () => void;
}

/**
 * ConfigProviderProps defines the props for ConfigProvider.
 */
export interface ConfigProviderProps {
  children: ReactNode;
  /** Whether to auto-load configuration on mount (default: true) */
  autoLoad?: boolean;
}

/** Context for configuration state */
const ConfigContext = createContext<ConfigContextValue | undefined>(undefined);

/**
 * ConfigProvider wraps the application with configuration state management.
 *
 * @example
 * ```tsx
 * function App() {
 *   return (
 *     <ConfigProvider>
 *       <YourApp />
 *     </ConfigProvider>
 *   );
 * }
 *
 * function ConfigDisplay() {
 *   const { config, isLoading, error } = useConfig();
 *
 *   if (isLoading) return <Loading />;
 *   if (error) return <Error message={error} />;
 *   if (!config) return null;
 *
 *   return <div>{Object.keys(config.agents).length} agents</div>;
 * }
 * ```
 */
export function ConfigProvider({
  children,
  autoLoad = true,
}: ConfigProviderProps): React.ReactElement {
  const [config, setConfigState] = useState<Config | null>(null);
  const [savedConfig, setSavedConfig] = useState<Config | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [validationResult, setValidationResult] = useState<ConfigValidationResult | null>(null);

  /**
   * Load configuration from backend.
   */
  const loadConfig = useCallback(async (): Promise<void> => {
    setIsLoading(true);
    setError(null);

    try {
      const loadedConfig = await getConfig();
      setConfigState(loadedConfig);
      setSavedConfig(loadedConfig);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load configuration';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, []);

  /**
   * Save configuration to backend.
   */
  const saveConfig = useCallback(
    async (newConfig?: Config): Promise<boolean> => {
      const configToSave = newConfig ?? config;
      if (!configToSave) {
        setError('No configuration to save');
        return false;
      }

      setIsSaving(true);
      setError(null);

      try {
        const savedResult = await updateConfig(configToSave);
        setConfigState(savedResult);
        setSavedConfig(savedResult);
        setValidationResult({ valid: true, errors: [] });
        return true;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to save configuration';
        setError(message);
        return false;
      } finally {
        setIsSaving(false);
      }
    },
    [config]
  );

  /**
   * Validate configuration without saving.
   */
  const validateCurrentConfig = useCallback(
    async (configToValidate?: Config): Promise<ConfigValidationResult> => {
      const configForValidation = configToValidate ?? config;
      if (!configForValidation) {
        const result: ConfigValidationResult = { valid: false, errors: ['No configuration to validate'] };
        setValidationResult(result);
        return result;
      }

      try {
        const result = await validateConfig(configForValidation);
        setValidationResult(result);
        return result;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Validation failed';
        const result: ConfigValidationResult = { valid: false, errors: [message] };
        setValidationResult(result);
        return result;
      }
    },
    [config]
  );

  /**
   * Update a specific agent in the configuration (local state only).
   */
  const updateAgent = useCallback((agentName: string, agentConfig: AgentConfig): void => {
    setConfigState((prev) => {
      if (!prev) return prev;
      return {
        ...prev,
        agents: {
          ...prev.agents,
          [agentName]: agentConfig,
        },
      };
    });
  }, []);

  /**
   * Set the entire configuration (local state only).
   */
  const setConfig = useCallback((newConfig: Config): void => {
    setConfigState(newConfig);
  }, []);

  /**
   * Add a new agent to the configuration (local state only).
   */
  const addAgent = useCallback((agentName: string, agentConfig: AgentConfig): void => {
    setConfigState((prev) => {
      if (!prev) return prev;
      if (prev.agents[agentName]) {
        setError(`Agent "${agentName}" already exists`);
        return prev;
      }
      return {
        ...prev,
        agents: {
          ...prev.agents,
          [agentName]: agentConfig,
        },
      };
    });
  }, []);

  /**
   * Remove an agent from the configuration (local state only).
   */
  const removeAgent = useCallback((agentName: string): void => {
    setConfigState((prev) => {
      if (!prev) return prev;
      const { [agentName]: removed, ...remainingAgents } = prev.agents;
      if (!removed) {
        setError(`Agent "${agentName}" not found`);
        return prev;
      }
      return {
        ...prev,
        agents: remainingAgents,
      };
    });
  }, []);

  /**
   * Reset configuration to last saved state.
   */
  const resetConfig = useCallback((): void => {
    if (savedConfig) {
      setConfigState(savedConfig);
      setError(null);
      setValidationResult(null);
    }
  }, [savedConfig]);

  /**
   * Clear the current error.
   */
  const clearError = useCallback((): void => {
    setError(null);
  }, []);

  /**
   * Check if configuration has unsaved changes.
   */
  const hasUnsavedChanges = useMemo(() => {
    if (!config || !savedConfig) return false;
    return JSON.stringify(config) !== JSON.stringify(savedConfig);
  }, [config, savedConfig]);

  // Auto-load configuration on mount
  useEffect(() => {
    if (autoLoad) {
      loadConfig();
    }
  }, [autoLoad, loadConfig]);

  const value: ConfigContextValue = useMemo(
    () => ({
      config,
      isLoading,
      isSaving,
      error,
      validationResult,
      loadConfig,
      saveConfig,
      validateCurrentConfig,
      updateAgent,
      setConfig,
      addAgent,
      removeAgent,
      resetConfig,
      hasUnsavedChanges,
      clearError,
    }),
    [
      config,
      isLoading,
      isSaving,
      error,
      validationResult,
      loadConfig,
      saveConfig,
      validateCurrentConfig,
      updateAgent,
      setConfig,
      addAgent,
      removeAgent,
      resetConfig,
      hasUnsavedChanges,
      clearError,
    ]
  );

  return <ConfigContext.Provider value={value}>{children}</ConfigContext.Provider>;
}

/**
 * useConfig hook provides access to the configuration context.
 *
 * @throws Error if used outside of ConfigProvider
 * @returns Configuration context value
 *
 * @example
 * ```tsx
 * function AgentList() {
 *   const { config, isLoading, updateAgent, saveConfig } = useConfig();
 *
 *   if (isLoading || !config) return <Loading />;
 *
 *   return (
 *     <div>
 *       {Object.entries(config.agents).map(([name, agent]) => (
 *         <AgentCard
 *           key={name}
 *           name={name}
 *           agent={agent}
 *           onUpdate={(updated) => updateAgent(name, updated)}
 *         />
 *       ))}
 *       <button onClick={() => saveConfig()}>Save</button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useConfig(): ConfigContextValue {
  const context = useContext(ConfigContext);
  if (context === undefined) {
    throw new Error('useConfig must be used within a ConfigProvider');
  }
  return context;
}

/**
 * useAgentConfig hook provides access to a specific agent's configuration.
 *
 * @param agentName - Name of the agent
 * @returns Agent configuration and update function
 *
 * @example
 * ```tsx
 * function AgentEditor({ name }: { name: string }) {
 *   const { agent, updateAgent, isLoading } = useAgentConfig(name);
 *
 *   if (isLoading || !agent) return <Loading />;
 *
 *   return (
 *     <input
 *       value={agent.systemPrompt}
 *       onChange={(e) =>
 *         updateAgent({ ...agent, systemPrompt: e.target.value })
 *       }
 *     />
 *   );
 * }
 * ```
 */
export function useAgentConfig(agentName: string): {
  agent: AgentConfig | null;
  updateAgent: (agentConfig: AgentConfig) => void;
  isLoading: boolean;
  error: string | null;
} {
  const { config, isLoading, error, updateAgent: updateAgentInConfig } = useConfig();

  const agent = config?.agents[agentName] ?? null;

  const updateAgent = useCallback(
    (agentConfig: AgentConfig) => {
      updateAgentInConfig(agentName, agentConfig);
    },
    [agentName, updateAgentInConfig]
  );

  return {
    agent,
    updateAgent,
    isLoading,
    error,
  };
}

export default ConfigContext;
