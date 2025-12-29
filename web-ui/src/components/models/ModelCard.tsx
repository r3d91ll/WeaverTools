/**
 * ModelCard component - displays a single model with status and controls.
 *
 * Shows model information including name, size, memory usage, and provides
 * load/unload controls.
 */
import type { ModelInfo } from '@/services/modelApi';
import { formatModelSize, formatParameterCount } from '@/services/modelApi';
import { ModelStatus, getModelStatusType } from './ModelStatus';

/**
 * ModelCard component props.
 */
export interface ModelCardProps {
  /** Model data to display */
  model: ModelInfo;
  /** Callback when load is requested */
  onLoad?: (name: string) => void;
  /** Callback when unload is requested */
  onUnload?: (name: string) => void;
  /** Whether the model is currently loading */
  isLoading?: boolean;
  /** Whether the model is currently unloading */
  isUnloading?: boolean;
  /** Whether actions are disabled */
  disabled?: boolean;
  /** Error message if operation failed */
  error?: string | null;
  /** Callback to clear error state */
  onClearError?: (name: string) => void;
  /** Whether to use compact view */
  compact?: boolean;
  /** Whether to show the estimated load time */
  showLoadEstimate?: boolean;
  /** Optional className for the card container */
  className?: string;
}

/**
 * Get backend display name and color.
 */
function getBackendInfo(backend: string): { name: string; color: string } {
  switch (backend.toLowerCase()) {
    case 'loom':
    case 'theloom':
      return { name: 'TheLoom', color: 'bg-purple-100 text-purple-800' };
    case 'claudecode':
    case 'claude':
      return { name: 'Claude Code', color: 'bg-blue-100 text-blue-800' };
    default:
      return { name: backend, color: 'bg-gray-100 text-gray-800' };
  }
}

/**
 * Estimate load time based on model size.
 * Rough estimate: ~2 seconds per GB on average hardware.
 */
function estimateLoadTime(sizeBytes: number): string {
  if (!sizeBytes) return '';
  const sizeGB = sizeBytes / (1024 * 1024 * 1024);
  const seconds = Math.ceil(sizeGB * 2);
  if (seconds < 60) return `~${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  return `~${minutes}min`;
}

/**
 * ModelCard component for displaying model information.
 */
export const ModelCard: React.FC<ModelCardProps> = ({
  model,
  onLoad,
  onUnload,
  isLoading = false,
  isUnloading = false,
  disabled = false,
  error = null,
  onClearError,
  compact = false,
  showLoadEstimate = false,
  className = '',
}) => {
  const status = getModelStatusType(model.loaded, isLoading, isUnloading, !!error);
  const isActionInProgress = isLoading || isUnloading;
  const isDisabled = disabled || isActionInProgress;
  const backendInfo = getBackendInfo(model.backend);

  const handleLoad = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onLoad?.(model.name);
  };

  const handleUnload = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onUnload?.(model.name);
  };

  const handleClearError = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onClearError?.(model.name);
  };

  const handleRetry = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onClearError?.(model.name);
    // Retry the last failed operation
    if (model.loaded) {
      onUnload?.(model.name);
    } else {
      onLoad?.(model.name);
    }
  };

  return (
    <div className={`card ${isActionInProgress ? 'opacity-75' : ''} transition-all ${className}`}>
      {/* Header with name and status */}
      <div className={`flex items-start justify-between ${compact ? 'mb-2' : 'mb-3'}`}>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-900 truncate">
              {model.name}
            </h3>
            {model.quantization && (
              <span className="text-xs font-mono bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded">
                {model.quantization}
              </span>
            )}
          </div>
          {model.path && !compact && (
            <p className="text-sm text-gray-500 truncate mt-0.5" title={model.path}>
              {model.path}
            </p>
          )}
        </div>
        <div className="flex-shrink-0 ml-4">
          <ModelStatus status={status} showLabel size="md" />
        </div>
      </div>

      {/* Model info row */}
      {!compact && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-3">
          {/* Backend */}
          <div>
            <p className="text-xs text-gray-500">Backend</p>
            <span className={`inline-block text-xs font-medium px-2 py-0.5 rounded-full mt-1 ${backendInfo.color}`}>
              {backendInfo.name}
            </span>
          </div>

          {/* Size */}
          <div>
            <p className="text-xs text-gray-500">Size</p>
            <p className="text-sm font-medium text-gray-900">
              {model.size ? formatModelSize(model.size) : 'N/A'}
            </p>
          </div>

          {/* Parameters */}
          {model.parameters !== undefined && (
            <div>
              <p className="text-xs text-gray-500">Parameters</p>
              <p className="text-sm font-medium text-gray-900">
                {formatParameterCount(model.parameters)}
              </p>
            </div>
          )}

          {/* Memory Usage */}
          <div>
            <p className="text-xs text-gray-500">Memory Used</p>
            <p className={`text-sm font-medium ${model.loaded ? 'text-weaver-600' : 'text-gray-400'}`}>
              {model.loaded && model.memoryUsed
                ? formatModelSize(model.memoryUsed)
                : '---'}
            </p>
          </div>
        </div>
      )}

      {/* Compact info row */}
      {compact && (
        <div className="flex items-center gap-4 text-sm text-gray-500 mb-3">
          <span className={`text-xs font-medium px-2 py-0.5 rounded-full ${backendInfo.color}`}>
            {backendInfo.name}
          </span>
          {model.size && (
            <span>{formatModelSize(model.size)}</span>
          )}
          {model.parameters !== undefined && (
            <span>{formatParameterCount(model.parameters)} params</span>
          )}
        </div>
      )}

      {/* Error message with retry/dismiss actions */}
      {error && (
        <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded-md">
          <div className="flex items-start justify-between gap-2">
            <div className="flex items-start gap-2 flex-1 min-w-0">
              <svg
                className="w-4 h-4 text-red-500 flex-shrink-0 mt-0.5"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                  clipRule="evenodd"
                />
              </svg>
              <p className="text-sm text-red-700">{error}</p>
            </div>
            <div className="flex items-center gap-1 flex-shrink-0">
              {onClearError && (
                <>
                  <button
                    type="button"
                    onClick={handleRetry}
                    className="text-xs text-red-600 hover:text-red-800 font-medium px-1.5 py-0.5 hover:bg-red-100 rounded"
                    title="Retry operation"
                  >
                    Retry
                  </button>
                  <button
                    type="button"
                    onClick={handleClearError}
                    className="text-xs text-gray-500 hover:text-gray-700 px-1 py-0.5 hover:bg-gray-100 rounded"
                    title="Dismiss error"
                  >
                    <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-100">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          {model.loaded && model.memoryUsed ? (
            <span>Using {formatModelSize(model.memoryUsed)} memory</span>
          ) : showLoadEstimate && model.size ? (
            <span>Est. load time: {estimateLoadTime(model.size)}</span>
          ) : null}
        </div>
        <div className="flex items-center gap-2">
          {model.loaded ? (
            <button
              type="button"
              onClick={handleUnload}
              disabled={isDisabled}
              className="btn-secondary text-sm py-1.5 px-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isUnloading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
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
                  Unloading...
                </span>
              ) : (
                'Unload'
              )}
            </button>
          ) : (
            <button
              type="button"
              onClick={handleLoad}
              disabled={isDisabled}
              className="btn-primary text-sm py-1.5 px-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
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
                  Loading...
                </span>
              ) : (
                'Load'
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelCard;
