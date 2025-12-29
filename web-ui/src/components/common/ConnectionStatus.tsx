/**
 * ConnectionStatus components - WebSocket connection status indicators.
 *
 * A collection of components for displaying WebSocket connection status,
 * from simple dot indicators to full panels with controls.
 */
import React, { useState, useEffect, useCallback, type ReactNode } from 'react';
import { useConnection, useConnectionStatus } from '@/contexts/ConnectionContext';
import { Button } from './Button';
import { Alert } from './Alert';
import type { WebSocketChannel } from '@/types';

/**
 * Connection status type for styling.
 */
export type ConnectionStatusType = 'connected' | 'connecting' | 'disconnected' | 'error';

/**
 * Dot size options.
 */
export type DotSize = 'xs' | 'sm' | 'md' | 'lg';

/**
 * ConnectionStatusDot props.
 */
export interface ConnectionStatusDotProps {
  /** Size of the dot */
  size?: DotSize;
  /** Override status (defaults to actual connection status) */
  status?: ConnectionStatusType;
  /** Whether to show pulsing animation when connecting */
  pulse?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Get size classes for the dot.
 */
function getDotSizeClasses(size: DotSize): string {
  switch (size) {
    case 'xs':
      return 'h-1.5 w-1.5';
    case 'sm':
      return 'h-2 w-2';
    case 'md':
      return 'h-2.5 w-2.5';
    case 'lg':
      return 'h-3 w-3';
  }
}

/**
 * Get color classes for the status.
 */
function getStatusColorClasses(status: ConnectionStatusType): string {
  switch (status) {
    case 'connected':
      return 'bg-green-500';
    case 'connecting':
      return 'bg-yellow-500';
    case 'disconnected':
      return 'bg-gray-400';
    case 'error':
      return 'bg-red-500';
  }
}

/**
 * Get text color classes for the status.
 */
function getStatusTextColorClasses(status: ConnectionStatusType): string {
  switch (status) {
    case 'connected':
      return 'text-green-600';
    case 'connecting':
      return 'text-yellow-600';
    case 'disconnected':
      return 'text-gray-500';
    case 'error':
      return 'text-red-600';
  }
}

/**
 * ConnectionStatusDot - A simple dot indicator showing connection state.
 *
 * @example
 * ```tsx
 * // Basic usage (uses actual connection status)
 * <ConnectionStatusDot />
 *
 * // Large dot with pulsing animation
 * <ConnectionStatusDot size="lg" pulse />
 *
 * // Override status for custom use
 * <ConnectionStatusDot status="error" />
 * ```
 */
export const ConnectionStatusDot: React.FC<ConnectionStatusDotProps> = ({
  size = 'sm',
  status: overrideStatus,
  pulse = true,
  className = '',
}) => {
  const { status: contextStatus, isConnected } = useConnectionStatus();
  const { error } = useConnection();

  // Determine actual status
  let status: ConnectionStatusType = overrideStatus ?? contextStatus;
  if (!overrideStatus && error && !isConnected) {
    status = 'error';
  }

  const sizeClasses = getDotSizeClasses(size);
  const colorClasses = getStatusColorClasses(status);
  const pulseClasses = pulse && status === 'connecting' ? 'animate-pulse' : '';

  return (
    <span
      className={`inline-block rounded-full ${sizeClasses} ${colorClasses} ${pulseClasses} ${className}`}
      role="status"
      aria-label={`Connection status: ${status}`}
    />
  );
};

/**
 * ConnectionStatusBadge props.
 */
export interface ConnectionStatusBadgeProps {
  /** Size of the badge */
  size?: 'sm' | 'md';
  /** Whether to show the status text */
  showText?: boolean;
  /** Custom label (overrides default text) */
  label?: string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * ConnectionStatusBadge - A badge with dot and text showing connection state.
 *
 * @example
 * ```tsx
 * // Basic badge with text
 * <ConnectionStatusBadge />
 *
 * // Compact badge without text
 * <ConnectionStatusBadge showText={false} />
 *
 * // Custom label
 * <ConnectionStatusBadge label="Server" />
 * ```
 */
export const ConnectionStatusBadge: React.FC<ConnectionStatusBadgeProps> = ({
  size = 'md',
  showText = true,
  label,
  className = '',
}) => {
  const { status, label: statusLabel, isConnected } = useConnectionStatus();
  const { error } = useConnection();

  // Determine actual status
  let displayStatus: ConnectionStatusType = status;
  if (error && !isConnected) {
    displayStatus = 'error';
  }

  const textColorClasses = getStatusTextColorClasses(displayStatus);
  const paddingClasses = size === 'sm' ? 'px-2 py-0.5' : 'px-2.5 py-1';
  const textSizeClasses = size === 'sm' ? 'text-xs' : 'text-sm';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full bg-gray-100 ${paddingClasses} ${className}`}
      role="status"
    >
      <ConnectionStatusDot size={size === 'sm' ? 'xs' : 'sm'} status={displayStatus} />
      {showText && (
        <span className={`font-medium ${textSizeClasses} ${textColorClasses}`}>
          {label ?? statusLabel}
        </span>
      )}
    </span>
  );
};

/**
 * ConnectionStatusBar props.
 */
export interface ConnectionStatusBarProps {
  /** Whether to show connect/disconnect controls */
  showControls?: boolean;
  /** Whether to show channel subscriptions */
  showChannels?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * ConnectionStatusBar - A horizontal bar with status and optional controls.
 *
 * @example
 * ```tsx
 * // Basic status bar
 * <ConnectionStatusBar />
 *
 * // With controls
 * <ConnectionStatusBar showControls />
 *
 * // With channel info
 * <ConnectionStatusBar showChannels />
 * ```
 */
export const ConnectionStatusBar: React.FC<ConnectionStatusBarProps> = ({
  showControls = false,
  showChannels = false,
  className = '',
}) => {
  const {
    isConnected,
    isConnecting,
    subscribedChannels,
    connect,
    disconnect,
    error,
  } = useConnection();
  const { status, label } = useConnectionStatus();

  const [isLoading, setIsLoading] = useState(false);

  const handleConnect = useCallback(async () => {
    setIsLoading(true);
    try {
      await connect();
    } finally {
      setIsLoading(false);
    }
  }, [connect]);

  const handleDisconnect = useCallback(() => {
    disconnect();
  }, [disconnect]);

  // Determine actual status
  let displayStatus: ConnectionStatusType = status;
  if (error && !isConnected) {
    displayStatus = 'error';
  }

  const textColorClasses = getStatusTextColorClasses(displayStatus);

  return (
    <div
      className={`flex items-center justify-between gap-4 px-4 py-2 bg-gray-50 border rounded-lg ${className}`}
    >
      <div className="flex items-center gap-3">
        <ConnectionStatusDot size="md" status={displayStatus} />
        <div className="flex flex-col">
          <span className={`font-medium text-sm ${textColorClasses}`}>{label}</span>
          {error && displayStatus === 'error' && (
            <span className="text-xs text-red-500">{error}</span>
          )}
        </div>
      </div>

      <div className="flex items-center gap-3">
        {showChannels && subscribedChannels.length > 0 && (
          <div className="hidden sm:flex items-center gap-1">
            <span className="text-xs text-gray-500">Channels:</span>
            {subscribedChannels.map((channel) => (
              <span
                key={channel}
                className="px-1.5 py-0.5 text-xs bg-gray-200 rounded text-gray-600"
              >
                {channel}
              </span>
            ))}
          </div>
        )}

        {showControls && (
          <div className="flex items-center gap-2">
            {isConnected ? (
              <Button
                variant="outline"
                size="xs"
                onClick={handleDisconnect}
              >
                Disconnect
              </Button>
            ) : (
              <Button
                variant="primary"
                size="xs"
                onClick={handleConnect}
                loading={isLoading || isConnecting}
              >
                Connect
              </Button>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * ChannelToggle props.
 */
interface ChannelToggleProps {
  channel: WebSocketChannel;
  isSubscribed: boolean;
  onToggle: (channel: WebSocketChannel) => void;
}

/**
 * ChannelToggle - Toggle button for a single channel.
 */
const ChannelToggle: React.FC<ChannelToggleProps> = ({
  channel,
  isSubscribed,
  onToggle,
}) => {
  return (
    <button
      type="button"
      onClick={() => onToggle(channel)}
      className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
        isSubscribed
          ? 'bg-weaver-100 text-weaver-700 border border-weaver-300'
          : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
      }`}
    >
      {channel}
    </button>
  );
};

/**
 * ConnectionStatusPanel props.
 */
export interface ConnectionStatusPanelProps {
  /** Title for the panel */
  title?: string;
  /** Whether to show channel management */
  showChannelManagement?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * ConnectionStatusPanel - A detailed panel with full info and controls.
 *
 * @example
 * ```tsx
 * // Full panel with all features
 * <ConnectionStatusPanel showChannelManagement />
 *
 * // Custom title
 * <ConnectionStatusPanel title="WebSocket Status" />
 * ```
 */
export const ConnectionStatusPanel: React.FC<ConnectionStatusPanelProps> = ({
  title = 'Connection Status',
  showChannelManagement = true,
  className = '',
}) => {
  const {
    state,
    isConnected,
    isConnecting,
    isReconnecting,
    reconnectAttempts,
    error,
    lastConnectedAt,
    subscribedChannels,
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    subscribeAll,
    unsubscribeAll,
    clearError,
  } = useConnection();

  const [isLoading, setIsLoading] = useState(false);

  const handleConnect = useCallback(async () => {
    setIsLoading(true);
    try {
      await connect();
    } finally {
      setIsLoading(false);
    }
  }, [connect]);

  const handleToggleChannel = useCallback(
    (channel: WebSocketChannel) => {
      if (subscribedChannels.includes(channel)) {
        unsubscribe(channel);
      } else {
        subscribe(channel);
      }
    },
    [subscribedChannels, subscribe, unsubscribe]
  );

  // All available channels
  const allChannels: WebSocketChannel[] = ['measurements', 'messages', 'status', 'resources'];

  // Determine actual status
  let displayStatus: ConnectionStatusType = 'disconnected';
  if (isConnected) displayStatus = 'connected';
  else if (isConnecting || isReconnecting) displayStatus = 'connecting';
  else if (error) displayStatus = 'error';

  return (
    <div className={`bg-white border rounded-lg shadow-sm ${className}`}>
      {/* Header */}
      <div className="px-4 py-3 border-b flex items-center justify-between">
        <h3 className="font-semibold text-gray-900">{title}</h3>
        <ConnectionStatusBadge />
      </div>

      {/* Content */}
      <div className="p-4 space-y-4">
        {/* Status details */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-gray-500">State:</span>
            <span className={`ml-2 font-medium ${getStatusTextColorClasses(displayStatus)}`}>
              {state}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Reconnect attempts:</span>
            <span className="ml-2 font-medium text-gray-900">{reconnectAttempts}</span>
          </div>
          {lastConnectedAt && (
            <div className="col-span-2">
              <span className="text-gray-500">Last connected:</span>
              <span className="ml-2 font-medium text-gray-900">
                {lastConnectedAt.toLocaleTimeString()}
              </span>
            </div>
          )}
        </div>

        {/* Error display */}
        {error && (
          <Alert
            variant="error"
            dismissible
            onDismiss={clearError}
          >
            {error}
          </Alert>
        )}

        {/* Channel management */}
        {showChannelManagement && (
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700">Subscribed Channels</span>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={subscribeAll}
                  className="text-xs text-weaver-600 hover:text-weaver-700"
                >
                  Subscribe All
                </button>
                <span className="text-gray-300">|</span>
                <button
                  type="button"
                  onClick={unsubscribeAll}
                  className="text-xs text-gray-500 hover:text-gray-700"
                >
                  Unsubscribe All
                </button>
              </div>
            </div>
            <div className="flex flex-wrap gap-2">
              {allChannels.map((channel) => (
                <ChannelToggle
                  key={channel}
                  channel={channel}
                  isSubscribed={subscribedChannels.includes(channel)}
                  onToggle={handleToggleChannel}
                />
              ))}
            </div>
          </div>
        )}

        {/* Controls */}
        <div className="flex gap-2 pt-2">
          {isConnected ? (
            <Button variant="outline" onClick={disconnect} fullWidth>
              Disconnect
            </Button>
          ) : (
            <Button
              variant="primary"
              onClick={handleConnect}
              loading={isLoading || isConnecting || isReconnecting}
              fullWidth
            >
              {isReconnecting ? 'Reconnecting...' : 'Connect'}
            </Button>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * ConnectionBanner props.
 */
export interface ConnectionBannerProps {
  /** Whether the banner is sticky */
  sticky?: boolean;
  /** Whether to show reconnect button */
  showReconnect?: boolean;
  /** Whether to auto-hide when connected */
  autoHide?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * ConnectionBanner - A sticky banner for connection issues.
 *
 * @example
 * ```tsx
 * // Sticky banner at top of page
 * <ConnectionBanner sticky />
 *
 * // With reconnect button
 * <ConnectionBanner sticky showReconnect />
 *
 * // Auto-hide when connected
 * <ConnectionBanner sticky autoHide />
 * ```
 */
export const ConnectionBanner: React.FC<ConnectionBannerProps> = ({
  sticky = true,
  showReconnect = true,
  autoHide = true,
  className = '',
}) => {
  const {
    isConnected,
    isConnecting,
    isReconnecting,
    reconnectAttempts,
    error,
    connect,
  } = useConnection();

  const [dismissed, setDismissed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  // Reset dismissed state when connection changes
  useEffect(() => {
    if (!isConnected) {
      setDismissed(false);
    }
  }, [isConnected]);

  const handleReconnect = useCallback(async () => {
    setIsLoading(true);
    try {
      await connect();
    } finally {
      setIsLoading(false);
    }
  }, [connect]);

  // Don't show if connected and autoHide is enabled
  if (autoHide && isConnected) {
    return null;
  }

  // Don't show if dismissed
  if (dismissed) {
    return null;
  }

  // Don't show if connecting (still trying)
  if (isConnecting && !isReconnecting && reconnectAttempts === 0) {
    return null;
  }

  // Determine message
  let message: ReactNode;
  let variant: 'warning' | 'error' = 'warning';

  if (isReconnecting) {
    message = reconnectAttempts > 1
      ? `Connection lost. Reconnecting (attempt ${reconnectAttempts})...`
      : 'Connection lost. Reconnecting...';
  } else if (error) {
    message = `Connection error: ${error}`;
    variant = 'error';
  } else if (!isConnected) {
    message = 'Not connected to server';
  } else {
    return null;
  }

  const stickyClasses = sticky ? 'sticky top-0 z-40' : '';

  return (
    <div className={`${stickyClasses} ${className}`}>
      <Alert
        variant={variant}
        dismissible
        onDismiss={() => setDismissed(true)}
        className="rounded-none border-x-0"
        actions={
          showReconnect && !isReconnecting ? (
            <Button
              variant="outline"
              size="xs"
              onClick={handleReconnect}
              loading={isLoading}
            >
              Reconnect
            </Button>
          ) : undefined
        }
      >
        {message}
      </Alert>
    </div>
  );
};

/**
 * OfflineIndicator props.
 */
export interface OfflineIndicatorProps {
  /** Content to show when online */
  children?: ReactNode;
  /** Content to show when offline */
  offlineContent?: ReactNode;
  /** Whether to show a simple text indicator */
  simple?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * OfflineIndicator - A simple offline/online indicator based on browser status.
 *
 * @example
 * ```tsx
 * // Simple text indicator
 * <OfflineIndicator simple />
 *
 * // Wrap content that should be hidden when offline
 * <OfflineIndicator offlineContent={<p>You are offline</p>}>
 *   <OnlineOnlyContent />
 * </OfflineIndicator>
 * ```
 */
export const OfflineIndicator: React.FC<OfflineIndicatorProps> = ({
  children,
  offlineContent,
  simple = false,
  className = '',
}) => {
  const [isOnline, setIsOnline] = useState(
    typeof navigator !== 'undefined' ? navigator.onLine : true
  );

  useEffect(() => {
    const handleOnline = () => setIsOnline(true);
    const handleOffline = () => setIsOnline(false);

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  if (simple) {
    return (
      <span
        className={`inline-flex items-center gap-1.5 text-sm ${className}`}
        role="status"
      >
        <span
          className={`h-2 w-2 rounded-full ${isOnline ? 'bg-green-500' : 'bg-red-500'}`}
        />
        <span className={isOnline ? 'text-green-600' : 'text-red-600'}>
          {isOnline ? 'Online' : 'Offline'}
        </span>
      </span>
    );
  }

  if (!isOnline) {
    if (offlineContent) {
      return <>{offlineContent}</>;
    }

    return (
      <div
        className={`flex items-center justify-center gap-2 p-4 bg-red-50 border border-red-200 rounded-lg ${className}`}
        role="alert"
      >
        <svg
          className="h-5 w-5 text-red-500"
          viewBox="0 0 20 20"
          fill="currentColor"
          aria-hidden="true"
        >
          <path
            fillRule="evenodd"
            d="M3.707 2.293a1 1 0 00-1.414 1.414l14 14a1 1 0 001.414-1.414l-1.473-1.473A10.014 10.014 0 0019.542 10C18.268 5.943 14.478 3 10 3a9.958 9.958 0 00-4.512 1.074l-1.78-1.781zm4.261 4.26l1.514 1.515a2.003 2.003 0 012.45 2.45l1.514 1.514a4 4 0 00-5.478-5.478z"
            clipRule="evenodd"
          />
          <path d="M12.454 16.697L9.75 13.992a4 4 0 01-3.742-3.742L2.335 6.578A9.98 9.98 0 00.458 10c1.274 4.057 5.065 7 9.542 7 .847 0 1.669-.105 2.454-.303z" />
        </svg>
        <span className="text-sm font-medium text-red-700">
          You are currently offline
        </span>
      </div>
    );
  }

  return <>{children}</>;
};

export default ConnectionStatusDot;
