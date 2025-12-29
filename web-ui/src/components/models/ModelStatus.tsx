/**
 * ModelStatus component - displays model status indicator.
 *
 * Shows a visual indicator of whether a model is loaded, available, or in transition.
 */

/**
 * Model status type.
 */
export type ModelStatusType = 'loaded' | 'available' | 'loading' | 'unloading' | 'error';

/**
 * ModelStatus component props.
 */
export interface ModelStatusProps {
  /** Current status of the model */
  status: ModelStatusType;
  /** Whether to show the status label */
  showLabel?: boolean;
  /** Size of the indicator */
  size?: 'sm' | 'md' | 'lg';
}

/**
 * Get status configuration for styling.
 */
function getStatusConfig(status: ModelStatusType): {
  color: string;
  bgColor: string;
  textColor: string;
  label: string;
  animate: boolean;
} {
  switch (status) {
    case 'loaded':
      return {
        color: 'bg-green-500',
        bgColor: 'bg-green-100',
        textColor: 'text-green-800',
        label: 'Loaded',
        animate: false,
      };
    case 'available':
      return {
        color: 'bg-gray-400',
        bgColor: 'bg-gray-100',
        textColor: 'text-gray-600',
        label: 'Available',
        animate: false,
      };
    case 'loading':
      return {
        color: 'bg-blue-500',
        bgColor: 'bg-blue-100',
        textColor: 'text-blue-800',
        label: 'Loading',
        animate: true,
      };
    case 'unloading':
      return {
        color: 'bg-yellow-500',
        bgColor: 'bg-yellow-100',
        textColor: 'text-yellow-800',
        label: 'Unloading',
        animate: true,
      };
    case 'error':
      return {
        color: 'bg-red-500',
        bgColor: 'bg-red-100',
        textColor: 'text-red-800',
        label: 'Error',
        animate: false,
      };
  }
}

/**
 * Get size classes for the indicator.
 */
function getSizeClasses(size: 'sm' | 'md' | 'lg'): {
  indicator: string;
  badge: string;
  text: string;
} {
  switch (size) {
    case 'sm':
      return {
        indicator: 'w-1.5 h-1.5',
        badge: 'px-2 py-0.5',
        text: 'text-xs',
      };
    case 'md':
      return {
        indicator: 'w-2 h-2',
        badge: 'px-2.5 py-0.5',
        text: 'text-xs',
      };
    case 'lg':
      return {
        indicator: 'w-2.5 h-2.5',
        badge: 'px-3 py-1',
        text: 'text-sm',
      };
  }
}

/**
 * ModelStatus component showing model load state.
 */
export const ModelStatus: React.FC<ModelStatusProps> = ({
  status,
  showLabel = true,
  size = 'md',
}) => {
  const config = getStatusConfig(status);
  const sizeClasses = getSizeClasses(size);

  if (!showLabel) {
    // Just show the indicator dot
    return (
      <span
        className={`inline-block rounded-full ${sizeClasses.indicator} ${config.color} ${
          config.animate ? 'animate-pulse' : ''
        }`}
        title={config.label}
      />
    );
  }

  return (
    <span
      className={`inline-flex items-center rounded-full font-medium ${sizeClasses.badge} ${sizeClasses.text} ${config.bgColor} ${config.textColor}`}
    >
      <span
        className={`rounded-full ${sizeClasses.indicator} ${config.color} ${
          config.animate ? 'animate-pulse' : ''
        } mr-1.5`}
      />
      {config.label}
    </span>
  );
};

/**
 * Get status type from model state.
 */
export function getModelStatusType(
  loaded: boolean,
  isLoading = false,
  isUnloading = false,
  hasError = false
): ModelStatusType {
  if (hasError) return 'error';
  if (isLoading) return 'loading';
  if (isUnloading) return 'unloading';
  if (loaded) return 'loaded';
  return 'available';
}

export default ModelStatus;
