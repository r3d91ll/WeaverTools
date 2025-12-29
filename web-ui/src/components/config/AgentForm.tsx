/**
 * AgentForm component for editing agent configurations.
 *
 * Provides form fields for all agent settings including role, backend,
 * model selection, system prompt, tools, and inference parameters.
 */
import { useState, useCallback } from 'react';
import type { AgentConfig, AgentRole, BackendType } from '@/types';
import { INFERENCE_DEFAULTS } from '@/types/config';

/** Available agent roles */
const AGENT_ROLES: { value: AgentRole | string; label: string }[] = [
  { value: 'senior', label: 'Senior Engineer' },
  { value: 'junior', label: 'Junior Engineer' },
  { value: 'conversant', label: 'Conversant' },
  { value: 'subject', label: 'Subject' },
  { value: 'analyst', label: 'Analyst' },
  { value: 'architect', label: 'Architect' },
  { value: 'reviewer', label: 'Reviewer' },
];

/** Available backend types */
const BACKEND_TYPES: { value: BackendType; label: string }[] = [
  { value: 'claudecode', label: 'Claude Code' },
  { value: 'loom', label: 'TheLoom' },
];

/** Available tools for agents */
const AVAILABLE_TOOLS = [
  'read_file',
  'write_file',
  'list_directory',
  'execute_command',
  'search_files',
  'context_read',
  'context_write',
] as const;

/** GPU assignment options */
const GPU_OPTIONS = [
  { value: 'auto', label: 'Auto' },
  { value: '0', label: 'GPU 0 (cuda:0)' },
  { value: '1', label: 'GPU 1 (cuda:1)' },
];

/**
 * AgentForm component props.
 */
export interface AgentFormProps {
  /** Agent name (key in agents map) */
  name: string;
  /** Current agent configuration */
  config: AgentConfig;
  /** Callback when agent config changes */
  onChange: (name: string, config: AgentConfig) => void;
  /** Whether form is disabled */
  disabled?: boolean;
  /** Callback to delete agent (optional, hides delete button if not provided) */
  onDelete?: (name: string) => void;
  /** Whether this agent can be deleted */
  canDelete?: boolean;
}

/**
 * Form component for editing individual agent configuration.
 */
export const AgentForm: React.FC<AgentFormProps> = ({
  name,
  config,
  onChange,
  disabled = false,
  onDelete,
  canDelete = true,
}) => {
  const [expanded, setExpanded] = useState(false);

  /** Update a single field in the config */
  const updateField = useCallback(
    <K extends keyof AgentConfig>(field: K, value: AgentConfig[K]) => {
      onChange(name, { ...config, [field]: value });
    },
    [name, config, onChange]
  );

  /** Toggle a tool in the tools array */
  const toggleTool = useCallback(
    (tool: string) => {
      const currentTools = config.tools ?? [];
      const newTools = currentTools.includes(tool)
        ? currentTools.filter((t) => t !== tool)
        : [...currentTools, tool];
      updateField('tools', newTools);
    },
    [config.tools, updateField]
  );

  /** Handle number input changes with validation */
  const handleNumberChange = useCallback(
    (field: keyof AgentConfig, value: string, min?: number, max?: number) => {
      if (value === '') {
        updateField(field, undefined as AgentConfig[typeof field]);
        return;
      }
      let num = parseFloat(value);
      if (isNaN(num)) return;
      if (min !== undefined && num < min) num = min;
      if (max !== undefined && num > max) num = max;
      updateField(field, num as AgentConfig[typeof field]);
    },
    [updateField]
  );

  const showLoomSettings = config.backend === 'loom';

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Agent Header */}
      <div
        className={`flex items-center justify-between px-4 py-3 bg-gray-50 cursor-pointer ${
          disabled ? 'opacity-50' : ''
        }`}
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          {/* Active indicator */}
          <span
            className={`w-2.5 h-2.5 rounded-full ${
              config.active ? 'bg-green-500' : 'bg-gray-300'
            }`}
            title={config.active ? 'Active' : 'Inactive'}
          />
          <div>
            <h3 className="text-sm font-medium text-gray-900">{name}</h3>
            <p className="text-xs text-gray-500">
              {config.role} / {config.backend}
              {config.model && ` / ${config.model}`}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {/* Toggle active button */}
          <button
            type="button"
            onClick={(e) => {
              e.stopPropagation();
              updateField('active', !config.active);
            }}
            disabled={disabled}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              config.active
                ? 'bg-green-100 text-green-700 hover:bg-green-200'
                : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
            }`}
          >
            {config.active ? 'Active' : 'Inactive'}
          </button>
          {/* Expand/collapse icon */}
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform ${
              expanded ? 'rotate-180' : ''
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </div>

      {/* Agent Form (expanded) */}
      {expanded && (
        <div className="p-4 space-y-4 border-t border-gray-200">
          {/* Basic Settings Row */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Role */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Role
              </label>
              <select
                value={config.role}
                onChange={(e) => updateField('role', e.target.value)}
                disabled={disabled}
                className="input w-full"
              >
                {AGENT_ROLES.map((role) => (
                  <option key={role.value} value={role.value}>
                    {role.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Backend */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Backend
              </label>
              <select
                value={config.backend}
                onChange={(e) => updateField('backend', e.target.value as BackendType)}
                disabled={disabled}
                className="input w-full"
              >
                {BACKEND_TYPES.map((backend) => (
                  <option key={backend.value} value={backend.value}>
                    {backend.label}
                  </option>
                ))}
              </select>
            </div>

            {/* Model (for Loom backend) */}
            {showLoomSettings && (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Model
                </label>
                <input
                  type="text"
                  value={config.model ?? ''}
                  onChange={(e) => updateField('model', e.target.value)}
                  disabled={disabled}
                  placeholder="e.g., Qwen/Qwen2.5-Coder-7B-Instruct"
                  className="input w-full"
                />
              </div>
            )}
          </div>

          {/* System Prompt */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              System Prompt
            </label>
            <textarea
              value={config.systemPrompt}
              onChange={(e) => updateField('systemPrompt', e.target.value)}
              disabled={disabled}
              rows={4}
              placeholder="Enter system prompt..."
              className="input w-full resize-y"
            />
          </div>

          {/* Tools Section */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">
                Tools
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  checked={config.toolsEnabled}
                  onChange={(e) => updateField('toolsEnabled', e.target.checked)}
                  disabled={disabled}
                  className="rounded border-gray-300 text-weaver-600 focus:ring-weaver-500"
                />
                <span className="text-gray-600">Tools Enabled</span>
              </label>
            </div>
            <div className="flex flex-wrap gap-2">
              {AVAILABLE_TOOLS.map((tool) => {
                const isSelected = (config.tools ?? []).includes(tool);
                return (
                  <button
                    key={tool}
                    type="button"
                    onClick={() => toggleTool(tool)}
                    disabled={disabled || !config.toolsEnabled}
                    className={`px-3 py-1 text-xs rounded-full transition-colors ${
                      isSelected
                        ? 'bg-weaver-100 text-weaver-700 border border-weaver-300'
                        : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                    } ${!config.toolsEnabled ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    {tool}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Inference Parameters (for Loom backend) */}
          {showLoomSettings && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-3">
                Inference Parameters
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {/* Max Tokens */}
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    Max Tokens
                  </label>
                  <input
                    type="number"
                    value={config.maxTokens ?? ''}
                    onChange={(e) =>
                      handleNumberChange('maxTokens', e.target.value, 1)
                    }
                    disabled={disabled}
                    placeholder={String(INFERENCE_DEFAULTS.maxTokens)}
                    className="input w-full"
                  />
                </div>

                {/* Temperature */}
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    Temperature
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={config.temperature ?? ''}
                    onChange={(e) =>
                      handleNumberChange('temperature', e.target.value, 0, 2)
                    }
                    disabled={disabled}
                    placeholder={String(INFERENCE_DEFAULTS.temperature)}
                    className="input w-full"
                  />
                </div>

                {/* Top P */}
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    Top P
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    value={config.topP ?? ''}
                    onChange={(e) =>
                      handleNumberChange('topP', e.target.value, 0, 1)
                    }
                    disabled={disabled}
                    placeholder={String(INFERENCE_DEFAULTS.topP)}
                    className="input w-full"
                  />
                </div>

                {/* Top K */}
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    Top K
                  </label>
                  <input
                    type="number"
                    value={config.topK ?? ''}
                    onChange={(e) =>
                      handleNumberChange('topK', e.target.value, 1)
                    }
                    disabled={disabled}
                    placeholder="50"
                    className="input w-full"
                  />
                </div>

                {/* Context Length */}
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    Context Length
                  </label>
                  <input
                    type="number"
                    value={config.contextLength ?? ''}
                    onChange={(e) =>
                      handleNumberChange('contextLength', e.target.value, 1)
                    }
                    disabled={disabled}
                    placeholder={String(INFERENCE_DEFAULTS.contextLength)}
                    className="input w-full"
                  />
                </div>

                {/* GPU */}
                <div>
                  <label className="block text-xs text-gray-500 mb-1">
                    GPU
                  </label>
                  <select
                    value={config.gpu ?? 'auto'}
                    onChange={(e) => updateField('gpu', e.target.value)}
                    disabled={disabled}
                    className="input w-full"
                  >
                    {GPU_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          )}

          {/* Delete Button */}
          {onDelete && canDelete && (
            <div className="pt-4 border-t border-gray-200">
              <button
                type="button"
                onClick={() => onDelete(name)}
                disabled={disabled}
                className="text-sm text-red-600 hover:text-red-700 hover:underline"
              >
                Delete Agent
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AgentForm;
