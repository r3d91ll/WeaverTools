/**
 * ConfigEditor component - main configuration editor container.
 *
 * Orchestrates the form-based editor for Weaver configuration,
 * including agents, backends, and session settings.
 */
import { useState, useCallback, useEffect } from 'react';
import type { Config, AgentConfig } from '@/types';
import { DEFAULT_CONFIG } from '@/types/config';
import { getConfig, updateConfig, validateConfig } from '@/services/configApi';
import { AgentForm } from './AgentForm';
import { BackendForm } from './BackendForm';
import { SessionForm } from './SessionForm';
import { YamlEditor } from './YamlEditor';
import { ValidationError, type ValidationMessage } from '@/components/common/ValidationError';

/**
 * Editor mode type.
 */
export type EditorMode = 'form' | 'yaml';

/**
 * ConfigEditor component props.
 */
export interface ConfigEditorProps {
  /** Initial editor mode */
  initialMode?: EditorMode;
  /** Callback when config is saved */
  onSave?: (config: Config) => void;
  /** Callback when config validation fails */
  onValidationError?: (errors: string[]) => void;
}

/**
 * Loading state type.
 */
type LoadingState = 'idle' | 'loading' | 'saving' | 'validating';

/**
 * Main configuration editor component.
 */
export const ConfigEditor: React.FC<ConfigEditorProps> = ({
  initialMode = 'form',
  onSave,
  onValidationError,
}) => {
  // State
  const [config, setConfig] = useState<Config>(DEFAULT_CONFIG);
  const [originalConfig, setOriginalConfig] = useState<Config>(DEFAULT_CONFIG);
  const [mode, setMode] = useState<EditorMode>(initialMode);
  const [loadingState, setLoadingState] = useState<LoadingState>('loading');
  const [error, setError] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationMessage[]>([]);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<'agents' | 'backends' | 'session'>('agents');
  const [yamlParseError, setYamlParseError] = useState<string | null>(null);

  // Check if config has been modified
  const isDirty = JSON.stringify(config) !== JSON.stringify(originalConfig);

  // Load config on mount
  useEffect(() => {
    loadConfig();
  }, []);

  /** Load configuration from API */
  const loadConfig = useCallback(async () => {
    setLoadingState('loading');
    setError(null);
    try {
      const loadedConfig = await getConfig();
      setConfig(loadedConfig);
      setOriginalConfig(loadedConfig);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load configuration');
    } finally {
      setLoadingState('idle');
    }
  }, []);

  /** Save configuration to API */
  const handleSave = useCallback(async () => {
    setLoadingState('saving');
    setError(null);
    setSuccessMessage(null);
    setValidationErrors([]);

    try {
      const savedConfig = await updateConfig(config);
      setConfig(savedConfig);
      setOriginalConfig(savedConfig);
      setSuccessMessage('Configuration saved successfully');
      onSave?.(savedConfig);
      // Clear success message after 3 seconds
      setTimeout(() => setSuccessMessage(null), 3000);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save configuration');
    } finally {
      setLoadingState('idle');
    }
  }, [config, onSave]);

  /**
   * Parse validation error string to extract field path if present.
   * Errors from the API may have format: "field.path: error message" or just "error message"
   */
  const parseValidationError = useCallback((errorStr: string): ValidationMessage => {
    // Try to extract field path from error message (common formats)
    const colonMatch = errorStr.match(/^([a-zA-Z0-9_.]+):\s*(.+)$/);
    if (colonMatch) {
      return {
        field: colonMatch[1],
        message: colonMatch[2],
        severity: 'error',
      };
    }

    // Check for "in field" pattern
    const inFieldMatch = errorStr.match(/(.+)\s+in\s+([a-zA-Z0-9_.]+)$/i);
    if (inFieldMatch) {
      return {
        field: inFieldMatch[2],
        message: inFieldMatch[1],
        severity: 'error',
      };
    }

    // No field path detected
    return {
      message: errorStr,
      severity: 'error',
    };
  }, []);

  /** Validate configuration */
  const handleValidate = useCallback(async () => {
    setLoadingState('validating');
    setError(null);
    setValidationErrors([]);
    setSuccessMessage(null);

    try {
      const result = await validateConfig(config);
      if (result.valid) {
        setSuccessMessage('Configuration is valid');
        setTimeout(() => setSuccessMessage(null), 3000);
      } else {
        const parsedErrors = (result.errors ?? []).map(parseValidationError);
        setValidationErrors(parsedErrors);
        onValidationError?.(result.errors ?? []);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to validate configuration');
    } finally {
      setLoadingState('idle');
    }
  }, [config, onValidationError, parseValidationError]);

  /** Reset to original config */
  const handleReset = useCallback(() => {
    setConfig(originalConfig);
    setValidationErrors([]);
    setError(null);
    setSuccessMessage(null);
  }, [originalConfig]);

  /** Reset to defaults */
  const handleResetToDefaults = useCallback(() => {
    setConfig(DEFAULT_CONFIG);
    setValidationErrors([]);
    setError(null);
    setSuccessMessage(null);
  }, []);

  /** Update agents in config */
  const handleAgentChange = useCallback((name: string, agentConfig: AgentConfig) => {
    setConfig((prev) => ({
      ...prev,
      agents: { ...prev.agents, [name]: agentConfig },
    }));
  }, []);

  /** Delete an agent */
  const handleAgentDelete = useCallback((name: string) => {
    setConfig((prev) => {
      const { [name]: deleted, ...remaining } = prev.agents;
      return { ...prev, agents: remaining };
    });
  }, []);

  /** Add a new agent */
  const handleAddAgent = useCallback(() => {
    const baseName = 'new_agent';
    let name = baseName;
    let counter = 1;
    while (config.agents[name]) {
      name = `${baseName}_${counter}`;
      counter++;
    }

    const newAgent: AgentConfig = {
      role: 'junior',
      backend: 'loom',
      model: '',
      systemPrompt: '',
      tools: [],
      toolsEnabled: false,
      active: false,
    };

    setConfig((prev) => ({
      ...prev,
      agents: { ...prev.agents, [name]: newAgent },
    }));
  }, [config.agents]);

  /** Update backends config */
  const handleBackendsChange = useCallback((backends: Config['backends']) => {
    setConfig((prev) => ({ ...prev, backends }));
  }, []);

  /** Update session config */
  const handleSessionChange = useCallback((session: Config['session']) => {
    setConfig((prev) => ({ ...prev, session }));
  }, []);

  /** Update config from YAML editor */
  const handleYamlChange = useCallback((newConfig: Config) => {
    setConfig(newConfig);
  }, []);

  /** Handle YAML parse error */
  const handleYamlParseError = useCallback((error: string | null) => {
    setYamlParseError(error);
  }, []);

  const isLoading = loadingState !== 'idle';
  const isSaving = loadingState === 'saving';

  return (
    <div className="space-y-6">
      {/* Mode Tabs */}
      <div className="border-b border-gray-200">
        <nav className="-mb-px flex space-x-8">
          <button
            type="button"
            onClick={() => setMode('form')}
            className={`py-4 px-1 border-b-2 text-sm font-medium transition-colors ${
              mode === 'form'
                ? 'border-weaver-500 text-weaver-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            Form Editor
          </button>
          <button
            type="button"
            onClick={() => setMode('yaml')}
            className={`py-4 px-1 border-b-2 text-sm font-medium transition-colors ${
              mode === 'yaml'
                ? 'border-weaver-500 text-weaver-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            YAML View
          </button>
        </nav>
      </div>

      {/* API Error Display */}
      {error && (
        <ValidationError
          errors={error}
          title="Error"
          severity="error"
          onDismiss={() => setError(null)}
        />
      )}

      {/* Validation Errors Display */}
      {validationErrors.length > 0 && (
        <ValidationError
          errors={validationErrors}
          title="Validation Errors"
          severity="error"
          showFieldPaths={true}
          onDismiss={() => setValidationErrors([])}
        />
      )}

      {/* YAML Parse Error Display */}
      {yamlParseError && mode === 'yaml' && (
        <ValidationError
          errors={yamlParseError}
          title="YAML Syntax Error"
          severity="warning"
        />
      )}

      {/* Success Message */}
      {successMessage && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <div className="flex">
            <svg className="w-5 h-5 text-green-400 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                clipRule="evenodd"
              />
            </svg>
            <p className="ml-3 text-sm font-medium text-green-800">{successMessage}</p>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loadingState === 'loading' && (
        <div className="flex items-center justify-center py-12">
          <div className="flex items-center gap-3">
            <svg className="animate-spin h-5 w-5 text-weaver-600" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            <span className="text-gray-600">Loading configuration...</span>
          </div>
        </div>
      )}

      {/* Form Editor Mode */}
      {mode === 'form' && loadingState !== 'loading' && (
        <div className="space-y-6">
          {/* Section Tabs */}
          <div className="flex space-x-2">
            <button
              type="button"
              onClick={() => setActiveSection('agents')}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                activeSection === 'agents'
                  ? 'bg-weaver-100 text-weaver-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              Agents
            </button>
            <button
              type="button"
              onClick={() => setActiveSection('backends')}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                activeSection === 'backends'
                  ? 'bg-weaver-100 text-weaver-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              Backends
            </button>
            <button
              type="button"
              onClick={() => setActiveSection('session')}
              className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                activeSection === 'session'
                  ? 'bg-weaver-100 text-weaver-700'
                  : 'text-gray-600 hover:bg-gray-100'
              }`}
            >
              Session
            </button>
          </div>

          {/* Agents Section */}
          {activeSection === 'agents' && (
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">Agents</h2>
                  <p className="text-sm text-gray-500">
                    Configure AI agents with their roles, backends, and parameters.
                  </p>
                </div>
                <button
                  type="button"
                  onClick={handleAddAgent}
                  disabled={isLoading}
                  className="btn-secondary flex items-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  Add Agent
                </button>
              </div>
              <div className="space-y-4">
                {Object.entries(config.agents).map(([name, agentConfig]) => (
                  <AgentForm
                    key={name}
                    name={name}
                    config={agentConfig}
                    onChange={handleAgentChange}
                    onDelete={handleAgentDelete}
                    disabled={isLoading}
                    canDelete={Object.keys(config.agents).length > 1}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Backends Section */}
          {activeSection === 'backends' && (
            <div className="card">
              <div className="mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Backends</h2>
                <p className="text-sm text-gray-500">
                  Configure backend connections (Claude Code, TheLoom).
                </p>
              </div>
              <BackendForm
                backends={config.backends}
                onChange={handleBackendsChange}
                disabled={isLoading}
              />
            </div>
          )}

          {/* Session Section */}
          {activeSection === 'session' && (
            <div className="card">
              <div className="mb-4">
                <h2 className="text-lg font-semibold text-gray-900">Session Settings</h2>
                <p className="text-sm text-gray-500">
                  Configure session defaults and measurement modes.
                </p>
              </div>
              <SessionForm
                session={config.session}
                onChange={handleSessionChange}
                disabled={isLoading}
              />
            </div>
          )}
        </div>
      )}

      {/* YAML View Mode */}
      {mode === 'yaml' && loadingState !== 'loading' && (
        <div className="card">
          <div className="mb-4">
            <h2 className="text-lg font-semibold text-gray-900">YAML Configuration</h2>
            <p className="text-sm text-gray-500">
              View and edit the raw YAML configuration. Changes are applied in real-time.
            </p>
          </div>
          <YamlEditor
            config={config}
            onChange={handleYamlChange}
            onParseError={handleYamlParseError}
            disabled={isLoading}
          />
        </div>
      )}

      {/* Action Buttons */}
      {loadingState !== 'loading' && (
        <div className="flex items-center justify-between pt-4 border-t border-gray-200">
          <div className="flex items-center gap-2">
            {isDirty && (
              <span className="text-sm text-yellow-600">
                You have unsaved changes
              </span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={handleResetToDefaults}
              disabled={isLoading}
              className="text-sm text-gray-600 hover:text-gray-900 hover:underline disabled:opacity-50"
            >
              Reset to Defaults
            </button>
            {isDirty && (
              <button
                type="button"
                onClick={handleReset}
                disabled={isLoading}
                className="btn-secondary"
              >
                Discard Changes
              </button>
            )}
            <button
              type="button"
              onClick={handleValidate}
              disabled={isLoading || yamlParseError !== null}
              className="btn-secondary"
              title={yamlParseError ? 'Fix YAML errors before validating' : undefined}
            >
              {loadingState === 'validating' ? 'Validating...' : 'Validate'}
            </button>
            <button
              type="button"
              onClick={handleSave}
              disabled={isLoading || !isDirty || yamlParseError !== null}
              className="btn-primary"
              title={yamlParseError ? 'Fix YAML errors before saving' : undefined}
            >
              {isSaving ? 'Saving...' : 'Save Configuration'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ConfigEditor;
