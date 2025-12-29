/**
 * ConnectionContext provides global WebSocket connection state management.
 *
 * Features:
 * - Track WebSocket connection status across the application
 * - Provide connect/disconnect controls
 * - Manage channel subscriptions
 * - Track reconnection attempts and history
 * - Emit connection status events for UI updates
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  type ReactNode,
} from 'react';

import {
  type ConnectionState,
  getWebSocketService,
  type WebSocketService,
  type WebSocketServiceOptions,
} from '@/services/websocket';

import type { WebSocketChannel } from '@/types';

/**
 * ConnectionContextValue provides access to connection state and actions.
 */
export interface ConnectionContextValue {
  /** Current connection state */
  state: ConnectionState;
  /** Whether WebSocket is connected */
  isConnected: boolean;
  /** Whether WebSocket is currently connecting */
  isConnecting: boolean;
  /** Whether WebSocket is reconnecting after disconnection */
  isReconnecting: boolean;
  /** Number of reconnection attempts */
  reconnectAttempts: number;
  /** Last connection error, if any */
  error: string | null;
  /** Timestamp of last successful connection */
  lastConnectedAt: Date | null;
  /** Currently subscribed channels */
  subscribedChannels: WebSocketChannel[];
  /** Connect to the WebSocket server */
  connect: () => Promise<void>;
  /** Disconnect from the WebSocket server */
  disconnect: () => void;
  /** Subscribe to channels */
  subscribe: (...channels: WebSocketChannel[]) => void;
  /** Unsubscribe from channels */
  unsubscribe: (...channels: WebSocketChannel[]) => void;
  /** Subscribe to all available channels */
  subscribeAll: () => void;
  /** Unsubscribe from all channels */
  unsubscribeAll: () => void;
  /** Clear the current error */
  clearError: () => void;
  /** Get the underlying WebSocket service (for advanced use) */
  getService: () => WebSocketService;
}

/**
 * ConnectionProviderProps defines the props for ConnectionProvider.
 */
export interface ConnectionProviderProps {
  children: ReactNode;
  /** Whether to auto-connect on mount (default: true) */
  autoConnect?: boolean;
  /** Channels to subscribe to on connect (default: all) */
  initialChannels?: WebSocketChannel[];
  /** WebSocket service options */
  options?: WebSocketServiceOptions;
}

/** All available channels */
const ALL_CHANNELS: WebSocketChannel[] = ['measurements', 'messages', 'status', 'resources'];

/** Context for connection state */
const ConnectionContext = createContext<ConnectionContextValue | undefined>(undefined);

/**
 * ConnectionProvider wraps the application with WebSocket connection management.
 *
 * @example
 * ```tsx
 * function App() {
 *   return (
 *     <ConnectionProvider autoConnect initialChannels={['measurements', 'messages']}>
 *       <YourApp />
 *     </ConnectionProvider>
 *   );
 * }
 *
 * function StatusIndicator() {
 *   const { isConnected, state, reconnectAttempts } = useConnection();
 *
 *   return (
 *     <div className={isConnected ? 'text-green-500' : 'text-red-500'}>
 *       {state}
 *       {reconnectAttempts > 0 && ` (attempt ${reconnectAttempts})`}
 *     </div>
 *   );
 * }
 * ```
 */
export function ConnectionProvider({
  children,
  autoConnect = true,
  initialChannels = ALL_CHANNELS,
  options,
}: ConnectionProviderProps): React.ReactElement {
  const serviceRef = useRef<WebSocketService | null>(null);

  // Initialize service once
  if (!serviceRef.current) {
    serviceRef.current = getWebSocketService({
      ...options,
      initialChannels,
    });
  }

  const service = serviceRef.current;

  const [state, setState] = useState<ConnectionState>(service.getState());
  const [error, setError] = useState<string | null>(null);
  const [lastConnectedAt, setLastConnectedAt] = useState<Date | null>(null);
  const [subscribedChannels, setSubscribedChannels] = useState<WebSocketChannel[]>(
    service.getSubscribedChannels()
  );
  const [reconnectAttempts, setReconnectAttempts] = useState(0);

  // Derived states
  const isConnected = state === 'connected';
  const isConnecting = state === 'connecting';
  const isReconnecting = state === 'reconnecting';

  /**
   * Handle state changes from the WebSocket service.
   */
  useEffect(() => {
    const handleStateChange = (newState: ConnectionState) => {
      setState(newState);

      if (newState === 'connected') {
        setLastConnectedAt(new Date());
        setError(null);
        setReconnectAttempts(0);
      } else if (newState === 'reconnecting') {
        setReconnectAttempts((prev) => prev + 1);
      } else if (newState === 'disconnected') {
        // Only set error if we were previously connected and didn't disconnect manually
        if (state === 'connected' || state === 'reconnecting') {
          setError('Connection lost');
        }
      }
    };

    const handleError = (err: Error) => {
      setError(err.message);
    };

    service.on('stateChange', handleStateChange);
    service.on('error', handleError);

    // Sync initial state
    setState(service.getState());
    setSubscribedChannels(service.getSubscribedChannels());

    return () => {
      service.off('stateChange', handleStateChange);
      service.off('error', handleError);
    };
  }, [service, state]);

  /**
   * Connect to the WebSocket server.
   */
  const connect = useCallback(async (): Promise<void> => {
    setError(null);
    try {
      await service.connect();
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to connect';
      setError(message);
      throw err;
    }
  }, [service]);

  /**
   * Disconnect from the WebSocket server.
   */
  const disconnect = useCallback((): void => {
    service.disconnect();
    setReconnectAttempts(0);
    setError(null);
  }, [service]);

  /**
   * Subscribe to channels.
   */
  const subscribe = useCallback(
    (...channels: WebSocketChannel[]): void => {
      service.subscribe(...channels);
      setSubscribedChannels(service.getSubscribedChannels());
    },
    [service]
  );

  /**
   * Unsubscribe from channels.
   */
  const unsubscribe = useCallback(
    (...channels: WebSocketChannel[]): void => {
      service.unsubscribe(...channels);
      setSubscribedChannels(service.getSubscribedChannels());
    },
    [service]
  );

  /**
   * Subscribe to all available channels.
   */
  const subscribeAll = useCallback((): void => {
    service.subscribe(...ALL_CHANNELS);
    setSubscribedChannels(service.getSubscribedChannels());
  }, [service]);

  /**
   * Unsubscribe from all channels.
   */
  const unsubscribeAll = useCallback((): void => {
    service.unsubscribe(...ALL_CHANNELS);
    setSubscribedChannels(service.getSubscribedChannels());
  }, [service]);

  /**
   * Clear the current error.
   */
  const clearError = useCallback((): void => {
    setError(null);
  }, []);

  /**
   * Get the underlying WebSocket service.
   */
  const getService = useCallback((): WebSocketService => {
    return service;
  }, [service]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect && !service.isConnected() && service.getState() === 'disconnected') {
      service.connect().catch(() => {
        // Connection failed, will auto-reconnect if enabled
      });
    }
  }, [autoConnect, service]);

  const value: ConnectionContextValue = useMemo(
    () => ({
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
      getService,
    }),
    [
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
      getService,
    ]
  );

  return <ConnectionContext.Provider value={value}>{children}</ConnectionContext.Provider>;
}

/**
 * useConnection hook provides access to the connection context.
 *
 * @throws Error if used outside of ConnectionProvider
 * @returns Connection context value
 *
 * @example
 * ```tsx
 * function ConnectionStatus() {
 *   const { isConnected, state, connect, disconnect } = useConnection();
 *
 *   return (
 *     <div>
 *       <span>Status: {state}</span>
 *       {isConnected ? (
 *         <button onClick={disconnect}>Disconnect</button>
 *       ) : (
 *         <button onClick={connect}>Connect</button>
 *       )}
 *     </div>
 *   );
 * }
 * ```
 */
export function useConnection(): ConnectionContextValue {
  const context = useContext(ConnectionContext);
  if (context === undefined) {
    throw new Error('useConnection must be used within a ConnectionProvider');
  }
  return context;
}

/**
 * useConnectionStatus hook provides a simplified view of connection status.
 *
 * @returns Connection status for UI indicators
 *
 * @example
 * ```tsx
 * function StatusDot() {
 *   const { status, label } = useConnectionStatus();
 *
 *   const colors = {
 *     connected: 'bg-green-500',
 *     connecting: 'bg-yellow-500',
 *     disconnected: 'bg-red-500',
 *   };
 *
 *   return (
 *     <span className={`rounded-full w-2 h-2 ${colors[status]}`} title={label} />
 *   );
 * }
 * ```
 */
export function useConnectionStatus(): {
  status: 'connected' | 'connecting' | 'disconnected';
  label: string;
  isConnected: boolean;
} {
  const { state, reconnectAttempts } = useConnection();

  const status = useMemo(() => {
    if (state === 'connected') return 'connected' as const;
    if (state === 'connecting' || state === 'reconnecting') return 'connecting' as const;
    return 'disconnected' as const;
  }, [state]);

  const label = useMemo(() => {
    switch (state) {
      case 'connected':
        return 'Connected';
      case 'connecting':
        return 'Connecting...';
      case 'reconnecting':
        return reconnectAttempts > 1
          ? `Reconnecting (attempt ${reconnectAttempts})...`
          : 'Reconnecting...';
      case 'disconnected':
        return 'Disconnected';
      default:
        return 'Unknown';
    }
  }, [state, reconnectAttempts]);

  return {
    status,
    label,
    isConnected: state === 'connected',
  };
}

export default ConnectionContext;
