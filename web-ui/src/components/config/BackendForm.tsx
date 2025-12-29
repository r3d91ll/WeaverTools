/**
 * BackendForm component for editing backend configurations.
 *
 * Provides form controls for Claude Code and TheLoom backend settings.
 */
import { useState } from 'react';
import type { BackendsConfig, ClaudeCodeConfig, LoomConfig } from '@/types';

/**
 * Props for the BackendForm component.
 */
export interface BackendFormProps {
  /** Current backend configuration */
  backends: BackendsConfig;
  /** Callback when backend config changes */
  onChange: (backends: BackendsConfig) => void;
  /** Whether the form is disabled */
  disabled?: boolean;
}

/**
 * Toggle switch component for boolean settings.
 */
interface ToggleSwitchProps {
  label: string;
  description?: string;
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
}

const ToggleSwitch: React.FC<ToggleSwitchProps> = ({
  label,
  description,
  checked,
  onChange,
  disabled = false,
}) => (
  <div className="flex items-center justify-between">
    <div>
      <label className="text-sm font-medium text-gray-700">{label}</label>
      {description && (
        <p className="text-xs text-gray-500">{description}</p>
      )}
    </div>
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-weaver-500 focus:ring-offset-2 ${
        checked ? 'bg-weaver-600' : 'bg-gray-200'
      } ${disabled ? 'cursor-not-allowed opacity-50' : ''}`}
    >
      <span
        className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
          checked ? 'translate-x-5' : 'translate-x-0'
        }`}
      />
    </button>
  </div>
);

/**
 * Text input component for form fields.
 */
interface FormInputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  type?: 'text' | 'number' | 'url';
  min?: number;
  max?: number;
  description?: string;
}

const FormInput: React.FC<FormInputProps> = ({
  label,
  value,
  onChange,
  placeholder,
  disabled = false,
  type = 'text',
  min,
  max,
  description,
}) => (
  <div>
    <label className="block text-sm font-medium text-gray-700 mb-1">
      {label}
    </label>
    {description && (
      <p className="text-xs text-gray-500 mb-1">{description}</p>
    )}
    <input
      type={type}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      disabled={disabled}
      min={min}
      max={max}
      className={`block w-full rounded-md border-gray-300 shadow-sm focus:border-weaver-500 focus:ring-weaver-500 sm:text-sm ${
        disabled ? 'bg-gray-100 cursor-not-allowed' : ''
      }`}
    />
  </div>
);

/**
 * BackendForm component for editing Claude Code and TheLoom settings.
 */
export const BackendForm: React.FC<BackendFormProps> = ({
  backends,
  onChange,
  disabled = false,
}) => {
  const [expandedSection, setExpandedSection] = useState<'claudecode' | 'loom' | null>('claudecode');

  const updateClaudeCode = (updates: Partial<ClaudeCodeConfig>) => {
    onChange({
      ...backends,
      claudeCode: { ...backends.claudeCode, ...updates },
    });
  };

  const updateLoom = (updates: Partial<LoomConfig>) => {
    onChange({
      ...backends,
      loom: { ...backends.loom, ...updates },
    });
  };

  return (
    <div className="space-y-4">
      {/* Claude Code Section */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          type="button"
          onClick={() => setExpandedSection(expandedSection === 'claudecode' ? null : 'claudecode')}
          className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className={`w-2.5 h-2.5 rounded-full ${backends.claudeCode.enabled ? 'bg-green-500' : 'bg-gray-300'}`} />
            <span className="font-medium text-gray-900">Claude Code</span>
          </div>
          <svg
            className={`w-5 h-5 text-gray-500 transition-transform ${expandedSection === 'claudecode' ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {expandedSection === 'claudecode' && (
          <div className="p-4 space-y-4 border-t border-gray-200">
            <p className="text-sm text-gray-600">
              Claude Code provides access to Anthropic's Claude models with tool use and streaming support.
            </p>
            <ToggleSwitch
              label="Enabled"
              description="Enable Claude Code backend for agent communication"
              checked={backends.claudeCode.enabled}
              onChange={(enabled) => updateClaudeCode({ enabled })}
              disabled={disabled}
            />
          </div>
        )}
      </div>

      {/* TheLoom Section */}
      <div className="border border-gray-200 rounded-lg overflow-hidden">
        <button
          type="button"
          onClick={() => setExpandedSection(expandedSection === 'loom' ? null : 'loom')}
          className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 hover:bg-gray-100 transition-colors"
        >
          <div className="flex items-center gap-3">
            <span className={`w-2.5 h-2.5 rounded-full ${backends.loom.enabled ? 'bg-green-500' : 'bg-gray-300'}`} />
            <span className="font-medium text-gray-900">TheLoom</span>
          </div>
          <svg
            className={`w-5 h-5 text-gray-500 transition-transform ${expandedSection === 'loom' ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {expandedSection === 'loom' && (
          <div className="p-4 space-y-4 border-t border-gray-200">
            <p className="text-sm text-gray-600">
              TheLoom provides local model inference with hidden state extraction for conveyance measurements.
            </p>

            <ToggleSwitch
              label="Enabled"
              description="Enable TheLoom backend for local model inference"
              checked={backends.loom.enabled}
              onChange={(enabled) => updateLoom({ enabled })}
              disabled={disabled}
            />

            <FormInput
              label="URL"
              value={backends.loom.url}
              onChange={(url) => updateLoom({ url })}
              placeholder="http://localhost:8080"
              disabled={disabled || !backends.loom.enabled}
              type="url"
              description="TheLoom server URL"
            />

            <FormInput
              label="Path"
              value={backends.loom.path}
              onChange={(path) => updateLoom({ path })}
              placeholder="../TheLoom/the-loom"
              disabled={disabled || !backends.loom.enabled}
              description="Path to TheLoom directory (for auto-start)"
            />

            <FormInput
              label="Port"
              value={String(backends.loom.port)}
              onChange={(port) => updateLoom({ port: parseInt(port) || 8080 })}
              placeholder="8080"
              disabled={disabled || !backends.loom.enabled}
              type="number"
              min={1}
              max={65535}
              description="Port for TheLoom server"
            />

            <ToggleSwitch
              label="Auto Start"
              description="Automatically start TheLoom if not running"
              checked={backends.loom.autoStart}
              onChange={(autoStart) => updateLoom({ autoStart })}
              disabled={disabled || !backends.loom.enabled}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default BackendForm;
