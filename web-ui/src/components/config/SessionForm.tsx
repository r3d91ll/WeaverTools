/**
 * SessionForm component for editing session configuration.
 *
 * Provides form controls for measurement mode, auto-export, and export path.
 */
import type { SessionConfig, MeasurementMode } from '@/types';

/**
 * Props for the SessionForm component.
 */
export interface SessionFormProps {
  /** Current session configuration */
  session: SessionConfig;
  /** Callback when session config changes */
  onChange: (session: SessionConfig) => void;
  /** Whether the form is disabled */
  disabled?: boolean;
}

/**
 * Measurement mode options with descriptions.
 */
const MEASUREMENT_MODES: Array<{
  value: MeasurementMode;
  label: string;
  description: string;
}> = [
  {
    value: 'active',
    label: 'Active',
    description: 'Measure conveyance for every exchange',
  },
  {
    value: 'passive',
    label: 'Passive',
    description: 'Record exchanges without measurement',
  },
  {
    value: 'triggered',
    label: 'Triggered',
    description: 'Measure only on explicit trigger',
  },
  {
    value: 'disabled',
    label: 'Disabled',
    description: 'No measurement or recording',
  },
];

/**
 * SessionForm component for editing session settings.
 */
export const SessionForm: React.FC<SessionFormProps> = ({
  session,
  onChange,
  disabled = false,
}) => {
  const updateSession = (updates: Partial<SessionConfig>) => {
    onChange({ ...session, ...updates });
  };

  return (
    <div className="space-y-6">
      {/* Measurement Mode */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Measurement Mode
        </label>
        <p className="text-xs text-gray-500 mb-3">
          Controls how conveyance metrics are collected during sessions
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {MEASUREMENT_MODES.map((mode) => (
            <label
              key={mode.value}
              className={`relative flex cursor-pointer rounded-lg border p-4 focus:outline-none ${
                session.measurementMode === mode.value
                  ? 'border-weaver-500 ring-2 ring-weaver-500 bg-weaver-50'
                  : 'border-gray-300 hover:border-gray-400'
              } ${disabled ? 'cursor-not-allowed opacity-50' : ''}`}
            >
              <input
                type="radio"
                name="measurementMode"
                value={mode.value}
                checked={session.measurementMode === mode.value}
                onChange={() => updateSession({ measurementMode: mode.value })}
                disabled={disabled}
                className="sr-only"
              />
              <div className="flex flex-col">
                <span
                  className={`block text-sm font-medium ${
                    session.measurementMode === mode.value
                      ? 'text-weaver-900'
                      : 'text-gray-900'
                  }`}
                >
                  {mode.label}
                </span>
                <span
                  className={`block text-xs ${
                    session.measurementMode === mode.value
                      ? 'text-weaver-700'
                      : 'text-gray-500'
                  }`}
                >
                  {mode.description}
                </span>
              </div>
              {session.measurementMode === mode.value && (
                <svg
                  className="absolute top-4 right-4 h-5 w-5 text-weaver-600"
                  viewBox="0 0 20 20"
                  fill="currentColor"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                    clipRule="evenodd"
                  />
                </svg>
              )}
            </label>
          ))}
        </div>
      </div>

      {/* Auto Export Toggle */}
      <div className="flex items-center justify-between py-4 border-t border-gray-200">
        <div>
          <label className="text-sm font-medium text-gray-700">
            Auto Export
          </label>
          <p className="text-xs text-gray-500">
            Automatically export session data when completed
          </p>
        </div>
        <button
          type="button"
          role="switch"
          aria-checked={session.autoExport}
          disabled={disabled}
          onClick={() => updateSession({ autoExport: !session.autoExport })}
          className={`relative inline-flex h-6 w-11 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-weaver-500 focus:ring-offset-2 ${
            session.autoExport ? 'bg-weaver-600' : 'bg-gray-200'
          } ${disabled ? 'cursor-not-allowed opacity-50' : ''}`}
        >
          <span
            className={`pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out ${
              session.autoExport ? 'translate-x-5' : 'translate-x-0'
            }`}
          />
        </button>
      </div>

      {/* Export Path */}
      <div className="border-t border-gray-200 pt-4">
        <label className="block text-sm font-medium text-gray-700 mb-1">
          Export Path
        </label>
        <p className="text-xs text-gray-500 mb-2">
          Directory for exported session data (relative to working directory)
        </p>
        <div className="flex gap-2">
          <input
            type="text"
            value={session.exportPath}
            onChange={(e) => updateSession({ exportPath: e.target.value })}
            placeholder="./experiments"
            disabled={disabled}
            className={`flex-1 block rounded-md border-gray-300 shadow-sm focus:border-weaver-500 focus:ring-weaver-500 sm:text-sm ${
              disabled ? 'bg-gray-100 cursor-not-allowed' : ''
            }`}
          />
          <button
            type="button"
            disabled={disabled}
            className={`px-3 py-2 text-sm font-medium text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-weaver-500 ${
              disabled ? 'cursor-not-allowed opacity-50' : ''
            }`}
            title="Browse for directory"
          >
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
              />
            </svg>
          </button>
        </div>
      </div>

      {/* Session Info */}
      <div className="border-t border-gray-200 pt-4">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex">
            <svg
              className="w-5 h-5 text-blue-400 flex-shrink-0"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                clipRule="evenodd"
              />
            </svg>
            <div className="ml-3">
              <h4 className="text-sm font-medium text-blue-800">
                Session Configuration
              </h4>
              <p className="mt-1 text-xs text-blue-700">
                These settings apply to new sessions. Active sessions will
                retain their original configuration.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SessionForm;
