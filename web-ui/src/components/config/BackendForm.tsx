/**
 * BackendForm component for editing backend configurations.
 *
 * Provides form controls for Claude Code and TheLoom backend settings.
 */
import { useState, useEffect } from 'react';
import type { BackendsConfig, ClaudeCodeConfig, LoomConfig, GPUInfo } from '@/types';
import { getGPUs } from '@/services/resourceApi';

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
 * GPU selector component for selecting which GPUs to use.
 */
interface GPUSelectorProps {
  gpus: GPUInfo[];
  selectedGpus: number[];
  onChange: (gpus: number[]) => void;
  disabled?: boolean;
  loading?: boolean;
}

const GPUSelector: React.FC<GPUSelectorProps> = ({
  gpus,
  selectedGpus,
  onChange,
  disabled = false,
  loading = false,
}) => {
  const toggleGpu = (index: number) => {
    if (selectedGpus.includes(index)) {
      onChange(selectedGpus.filter(i => i !== index));
    } else {
      onChange([...selectedGpus, index].sort((a, b) => a - b));
    }
  };

  const selectAll = () => {
    onChange(gpus.map(g => g.index));
  };

  const deselectAll = () => {
    onChange([]);
  };

  if (loading) {
    return (
      <div className="text-sm text-gray-500 italic">
        Detecting GPUs...
      </div>
    );
  }

  if (gpus.length === 0) {
    return (
      <div className="text-sm text-gray-500">
        No GPUs detected. TheLoom will run on CPU.
      </div>
    );
  }

  const formatMemory = (mb: number) => {
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${mb} MB`;
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <label className="block text-sm font-medium text-gray-700">
          GPU Selection
        </label>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={selectAll}
            disabled={disabled}
            className="text-xs text-weaver-600 hover:text-weaver-700 disabled:opacity-50"
          >
            Select All
          </button>
          <span className="text-gray-300">|</span>
          <button
            type="button"
            onClick={deselectAll}
            disabled={disabled}
            className="text-xs text-gray-600 hover:text-gray-700 disabled:opacity-50"
          >
            Clear
          </button>
        </div>
      </div>
      <p className="text-xs text-gray-500 mb-2">
        Select which GPUs TheLoom can use. Leave empty to use all available GPUs.
      </p>
      <div className="space-y-2">
        {gpus.map((gpu) => (
          <label
            key={gpu.index}
            className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
              selectedGpus.includes(gpu.index)
                ? 'border-weaver-500 bg-weaver-50'
                : 'border-gray-200 hover:border-gray-300'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            <input
              type="checkbox"
              checked={selectedGpus.includes(gpu.index)}
              onChange={() => toggleGpu(gpu.index)}
              disabled={disabled}
              className="mt-1 h-4 w-4 rounded border-gray-300 text-weaver-600 focus:ring-weaver-500"
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <span className="font-medium text-gray-900 text-sm">
                  GPU {gpu.index}: {gpu.name}
                </span>
                <span className={`text-xs px-2 py-0.5 rounded-full ${
                  gpu.available
                    ? 'bg-green-100 text-green-700'
                    : 'bg-yellow-100 text-yellow-700'
                }`}>
                  {gpu.available ? 'Available' : 'Busy'}
                </span>
              </div>
              <div className="flex gap-4 mt-1 text-xs text-gray-500">
                <span>Memory: {formatMemory(gpu.memoryFree)} / {formatMemory(gpu.memoryTotal)} free</span>
                <span>Utilization: {gpu.utilization}%</span>
              </div>
              {/* Memory usage bar */}
              {gpu.memoryTotal > 0 && (
                <div className="mt-2 h-1.5 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      gpu.memoryUsed / gpu.memoryTotal > 0.8
                        ? 'bg-red-500'
                        : gpu.memoryUsed / gpu.memoryTotal > 0.5
                        ? 'bg-yellow-500'
                        : 'bg-green-500'
                    }`}
                    style={{ width: `${(gpu.memoryUsed / gpu.memoryTotal) * 100}%` }}
                  />
                </div>
              )}
            </div>
          </label>
        ))}
      </div>
    </div>
  );
};

/**
 * BackendForm component for editing Claude Code and TheLoom settings.
 */
export const BackendForm: React.FC<BackendFormProps> = ({
  backends,
  onChange,
  disabled = false,
}) => {
  const [expandedSection, setExpandedSection] = useState<'claudecode' | 'loom' | null>('claudecode');
  const [availableGpus, setAvailableGpus] = useState<GPUInfo[]>([]);
  const [gpusLoading, setGpusLoading] = useState(false);

  // Fetch available GPUs when TheLoom section is expanded
  useEffect(() => {
    if (expandedSection === 'loom') {
      setGpusLoading(true);
      getGPUs()
        .then((response) => {
          setAvailableGpus(response.gpus);
        })
        .catch((error) => {
          console.warn('Failed to fetch GPUs:', error);
          setAvailableGpus([]);
        })
        .finally(() => {
          setGpusLoading(false);
        });
    }
  }, [expandedSection]);

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

            {/* GPU Selection */}
            <div className="pt-4 border-t border-gray-200">
              <GPUSelector
                gpus={availableGpus}
                selectedGpus={backends.loom.gpus ?? []}
                onChange={(gpus) => updateLoom({ gpus })}
                disabled={disabled || !backends.loom.enabled}
                loading={gpusLoading}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default BackendForm;
