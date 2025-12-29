/**
 * ConnectionStatus component - WebSocket connection status indicator.
 *
 * A flexible component for displaying WebSocket connection status
 * with various display modes, controls, and detailed information.
 */
import React, { useState, useMemo } from 'react';
import { useConnection, useConnectionStatus } from '@/contexts/ConnectionContext';

/**
 * Connection status visual style.
 */
export type ConnectionStatusVariant = 'dot' | 'badge' | 'detailed' | 'minimal';

/**
 * Connection status size.
 */
export type ConnectionStatusSize = 'xs' | 'sm' | 'md' | 'lg';

/**
 * ConnectionStatus component props.
 */
export interface ConnectionStatusProps {
  /** Display variant */
  variant?: ConnectionStatusVariant;
  /** Component size */
  size?: ConnectionStatusSize;
  /** Whether to show connect/disconnect controls */
  showControls?: boolean;
  /** Whether to show additional details on hover/click */
  showDetails?: boolean;
  /** Whether to show the status text label */
  showLabel?: boolean;
  /** Whether to show reconnection attempts */
  showRetryCount?: boolean;
  /** Whether to show last connected timestamp */
  showLastConnected?: boolean;
  /** Whether to show subscribed channels */
  showChannels?: boolean;
  /** Callback when status is clicked */
  onClick?: () => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Get size-specific classes.
 */
function getSizeClasses(size: ConnectionStatusSize): {
  dot: string;
  text: string;
  badge: string;
  icon: string;
} {
  switch (size) {
    case 'xs':
      return {
        dot: 'w-1.5 h-1.5',
        text: 'text-xs',
        badge: 'px-1.5 py-0.5 text-xs',
        icon: 'w-3 h-3',
      };
    case 'sm':
      return {
        dot: 'w-2 h-2',
        text: 'text-sm',
        badge: 'px-2 py-0.5 text-sm',
        icon: 'w-3.5 h-3.5',
      };
    case 'md':
      return {
        dot: 'w-2.5 h-2.5',
        text: 'text-sm',
        badge: 'px-2.5 py-1 text-sm',
        icon: 'w-4 h-4',
      };
    case 'lg':
      return {
        dot: 'w-3 h-3',
        text: 'text-base',
        badge: 'px-3 py-1.5 text-base',
        icon: 'w-5 h-5',
      };
  }
}

/**
 * Get status-specific styling.
 */
function getStatusStyles(status: 'connected' | 'connecting' | 'disconnected'): {
  dotClasses: string;
  textClasses: string;
  badgeClasses: string;
  bgClasses: string;
} {
  switch (status) {
    case 'connected':
      return {
        dotClasses: 'bg-green-500',
        textClasses: 'text-green-700',
        badgeClasses: 'bg-green-100 text-green-800 border-green-200',
        bgClasses: 'bg-green-50',
      };
    case 'connecting':
      return {
        dotClasses: 'bg-yellow-500 animate-pulse',
        textClasses: 'text-yellow-700',
        badgeClasses: 'bg-yellow-100 text-yellow-800 border-yellow-200',
        bgClasses: 'bg-yellow-50',
      };
    case 'disconnected':
      return {
        dotClasses: 'bg-red-500',
        textClasses: 'text-red-700',
        badgeClasses: 'bg-red-100 text-red-800 border-red-200',
        bgClasses: 'bg-red-50',
      };
  }
}

/**
 * ConnectionStatus component for displaying WebSocket connection state.
 *
 * @example
 * ```tsx
 * // Simple dot indicator
 * <ConnectionStatus variant="dot" />
 *
 * // Badge with label
 * <ConnectionStatus variant="badge" showLabel />
 *
 * // Detailed view with controls
 * <ConnectionStatus
 *   variant="detailed"
 *   showControls
 *   showLastConnected
 *   showChannels
 * />
 *
 * // Minimal inline status
 * <ConnectionStatus variant="minimal" size="sm" />
 * ```
 */
export const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  variant = 'dot',
  size = 'md',
  showControls = false,
  showDetails = false,
  showLabel = true,
  showRetryCount = true,
  showLastConnected = false,
  showChannels = false,
  onClick,
  className = '',
}) => {
  const { status, label, isConnected } = useConnectionStatus();
  const connection = useConnection();
  const [isExpanded, setIsExpanded] = useState(false);

  const sizeClasses = getSizeClasses(size);
  const statusStyles = getStatusStyles(status);

  // Toggle expanded state for details
  const handleClick = (): void => {
    if (showDetails) {
      setIsExpanded(!isExpanded);
    }
    onClick?.();
  };

  // Format last connected time
  const lastConnectedLabel = useMemo(() => {
    if (!connection.lastConnectedAt) return 'Never';
    const diff = Date.now() - connection.lastConnectedAt.getTime();
    if (diff < 60000) return 'Just now';
    if (diff < 3600000) return `${Math.floor(diff / 60000)} min ago`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)} hours ago`;
    return connection.lastConnectedAt.toLocaleDateString();
  }, [connection.lastConnectedAt]);

  // Render dot variant
  if (variant === 'dot') {
    return (
      <div
        className={`inline-flex items-center gap-2 ${className}`}
        role="status"
        aria-label={label}
      >
        <span
          className={`rounded-full ${sizeClasses.dot} ${statusStyles.dotClasses}`}
          aria-hidden="true"
        />
        {showLabel && (
          <span className={`${sizeClasses.text} ${statusStyles.textClasses}`}>
            {label}
            {showRetryCount && connection.reconnectAttempts > 0 && status === 'connecting' && (
              <span className="text-gray-500 ml-1">
                ({connection.reconnectAttempts})
              </span>
            )}
          </span>
        )}
      </div>
    );
  }

  // Render minimal variant
  if (variant === 'minimal') {
    return (
      <span
        className={`inline-flex items-center gap-1.5 ${sizeClasses.text} ${statusStyles.textClasses} ${className}`}
        role="status"
        aria-label={label}
      >
        <span
          className={`rounded-full ${sizeClasses.dot} ${statusStyles.dotClasses}`}
          aria-hidden="true"
        />
        {showLabel && label}
      </span>
    );
  }

  // Render badge variant
  if (variant === 'badge') {
    return (
      <span
        className={`inline-flex items-center gap-1.5 rounded-full border ${sizeClasses.badge} ${statusStyles.badgeClasses} ${className}`}
        role="status"
        aria-label={label}
      >
        <span
          className={`rounded-full ${sizeClasses.dot} ${statusStyles.dotClasses}`}
          aria-hidden="true"
        />
        {showLabel && label}
      </span>
    );
  }

  // Render detailed variant
  return (
    <div
      className={`rounded-lg border ${statusStyles.bgClasses} ${
        statusStyles.badgeClasses.split(' ').find((c) => c.startsWith('border-')) ?? ''
      } ${className}`}
      role="status"
      aria-label={label}
    >
      {/* Header with status and controls */}
      <div
        className={`flex items-center justify-between p-3 ${
          showDetails ? 'cursor-pointer hover:bg-black/5' : ''
        }`}
        onClick={handleClick}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            handleClick();
          }
        }}
        tabIndex={showDetails ? 0 : undefined}
        role={showDetails ? 'button' : undefined}
        aria-expanded={showDetails ? isExpanded : undefined}
      >
        <div className="flex items-center gap-2">
          <span
            className={`rounded-full ${sizeClasses.dot} ${statusStyles.dotClasses}`}
            aria-hidden="true"
          />
          <div>
            <span className={`font-medium ${sizeClasses.text} ${statusStyles.textClasses}`}>
              {label}
            </span>
            {showRetryCount && connection.reconnectAttempts > 0 && status === 'connecting' && (
              <span className={`${sizeClasses.text} text-gray-500 ml-1`}>
                (attempt {connection.reconnectAttempts})
              </span>
            )}
          </div>
        </div>

        {/* Controls */}
        {showControls && (
          <div className="flex items-center gap-2">
            {isConnected ? (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  connection.disconnect();
                }}
                className="px-2 py-1 text-xs font-medium text-red-700 bg-red-100 rounded hover:bg-red-200 focus:outline-none focus:ring-2 focus:ring-red-500"
              >
                Disconnect
              </button>
            ) : (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  connection.connect();
                }}
                disabled={status === 'connecting'}
                className="px-2 py-1 text-xs font-medium text-green-700 bg-green-100 rounded hover:bg-green-200 focus:outline-none focus:ring-2 focus:ring-green-500 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Connect
              </button>
            )}
          </div>
        )}

        {/* Expand indicator */}
        {showDetails && (
          <svg
            className={`${sizeClasses.icon} text-gray-400 transition-transform ${
              isExpanded ? 'rotate-180' : ''
            }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        )}
      </div>

      {/* Expanded details */}
      {showDetails && isExpanded && (
        <div className="border-t border-current/10 p-3 space-y-2">
          {/* Error message */}
          {connection.error && (
            <div className="flex items-start gap-2 text-red-600">
              <svg
                className={sizeClasses.icon}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span className="text-xs">{connection.error}</span>
              <button
                type="button"
                onClick={() => connection.clearError()}
                className="text-xs underline hover:no-underline ml-auto"
              >
                Clear
              </button>
            </div>
          )}

          {/* Last connected */}
          {showLastConnected && (
            <div className="flex items-center gap-2 text-gray-600">
              <svg
                className={sizeClasses.icon}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span className="text-xs">Last connected: {lastConnectedLabel}</span>
            </div>
          )}

          {/* Subscribed channels */}
          {showChannels && connection.subscribedChannels.length > 0 && (
            <div className="flex items-start gap-2 text-gray-600">
              <svg
                className={`${sizeClasses.icon} flex-shrink-0 mt-0.5`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                />
              </svg>
              <div className="flex flex-wrap gap-1">
                {connection.subscribedChannels.map((channel) => (
                  <span
                    key={channel}
                    className="inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium bg-gray-200 text-gray-700"
                  >
                    {channel}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

/**
 * ConnectionStatusBanner props for page-level connection alerts.
 */
export interface ConnectionStatusBannerProps {
  /** Whether to show the banner when connected */
  showWhenConnected?: boolean;
  /** Whether to show reconnecting message with attempt count */
  showReconnectProgress?: boolean;
  /** Callback when the connect button is clicked */
  onConnect?: () => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * ConnectionStatusBanner component for page-level connection notifications.
 *
 * @example
 * ```tsx
 * // Shows banner only when disconnected
 * <ConnectionStatusBanner />
 *
 * // With reconnect progress
 * <ConnectionStatusBanner showReconnectProgress />
 * ```
 */
export const ConnectionStatusBanner: React.FC<ConnectionStatusBannerProps> = ({
  showWhenConnected = false,
  showReconnectProgress = true,
  onConnect,
  className = '',
}) => {
  const { status, label, isConnected } = useConnectionStatus();
  const { connect, reconnectAttempts } = useConnection();

  // Don't show when connected (unless explicitly requested)
  if (isConnected && !showWhenConnected) {
    return null;
  }

  const handleConnect = async (): Promise<void> => {
    onConnect?.();
    await connect();
  };

  const statusStyles = getStatusStyles(status);

  return (
    <div
      className={`${statusStyles.bgClasses} border-b ${
        statusStyles.badgeClasses.split(' ').find((c) => c.startsWith('border-')) ?? ''
      } px-4 py-2 ${className}`}
      role="alert"
    >
      <div className="flex items-center justify-between max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <span
            className={`rounded-full w-2 h-2 ${statusStyles.dotClasses}`}
            aria-hidden="true"
          />
          <span className={`text-sm font-medium ${statusStyles.textClasses}`}>
            {label}
            {showReconnectProgress && reconnectAttempts > 0 && status === 'connecting' && (
              <span className="text-gray-600 ml-1">
                (attempt {reconnectAttempts})
              </span>
            )}
          </span>
        </div>

        {!isConnected && status !== 'connecting' && (
          <button
            type="button"
            onClick={handleConnect}
            className="text-sm font-medium text-weaver-700 hover:text-weaver-800 focus:outline-none focus:underline"
          >
            Connect now
          </button>
        )}
      </div>
    </div>
  );
};

/**
 * ConnectionIndicator props for minimal inline status.
 */
export interface ConnectionIndicatorProps {
  /** Size of the indicator */
  size?: ConnectionStatusSize;
  /** Whether to show pulse animation when connecting */
  showPulse?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * ConnectionIndicator component for minimal inline connection status.
 *
 * @example
 * ```tsx
 * <div className="flex items-center gap-2">
 *   <ConnectionIndicator size="sm" />
 *   <span>Live</span>
 * </div>
 * ```
 */
export const ConnectionIndicator: React.FC<ConnectionIndicatorProps> = ({
  size = 'sm',
  showPulse = true,
  className = '',
}) => {
  const { status, label } = useConnectionStatus();
  const statusStyles = getStatusStyles(status);
  const sizeClasses = getSizeClasses(size);

  return (
    <span
      className={`inline-block rounded-full ${sizeClasses.dot} ${statusStyles.dotClasses} ${
        showPulse && status === 'connecting' ? 'animate-pulse' : ''
      } ${className}`}
      role="status"
      aria-label={label}
      title={label}
    />
  );
};

export default ConnectionStatus;
