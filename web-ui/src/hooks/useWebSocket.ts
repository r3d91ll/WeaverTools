/**
 * React hooks for WebSocket integration with the Weaver backend.
 * Provides reactive connection state and typed event subscriptions.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

import {
  WebSocketService,
  getWebSocketService,
  createWebSocketService,
  type ConnectionState,
  type WebSocketServiceOptions,
  type WebSocketEventType,
  type WebSocketEventListener,
  type ChatMessageData,
} from '@/services/websocket';

import type {
  WebSocketChannel,
  WebSocketMessage,
  MeasurementEvent,
  MeasurementBatchEvent,
  BetaAlertEvent,
  ConversationTurnEvent,
  SessionStartEvent,
  SessionEndEvent,
  BackendInfo,
  ModelInfo,
} from '@/types';

/**
 * ResourceUpdateData matches the Go ResourceUpdateData struct in websocket.go.
 * This is specific to WebSocket events, not the general ResourceStatus type.
 */
export interface ResourceUpdateData {
  gpuMemory: number;
  gpuUtil: number;
  queueDepth: number;
}

/** Return type for useWebSocket hook */
export interface UseWebSocketReturn {
  /** Current connection state */
  state: ConnectionState;
  /** Whether the WebSocket is connected */
  isConnected: boolean;
  /** Connect to the WebSocket server */
  connect: () => Promise<void>;
  /** Disconnect from the WebSocket server */
  disconnect: () => void;
  /** Subscribe to channels */
  subscribe: (...channels: WebSocketChannel[]) => void;
  /** Unsubscribe from channels */
  unsubscribe: (...channels: WebSocketChannel[]) => void;
  /** Get currently subscribed channels */
  subscribedChannels: WebSocketChannel[];
  /** The underlying WebSocket service instance */
  service: WebSocketService;
}

/**
 * Hook for managing WebSocket connection to the Weaver backend.
 * Handles connection lifecycle and provides reactive state.
 *
 * @param options - WebSocket service options
 * @param autoConnect - Whether to connect automatically on mount (default: true)
 * @returns WebSocket state and control functions
 *
 * @example
 * ```tsx
 * function MyComponent() {
 *   const { state, isConnected, connect, disconnect, subscribe } = useWebSocket({
 *     initialChannels: ['measurements', 'messages'],
 *   });
 *
 *   useEffect(() => {
 *     if (isConnected) {
 *       subscribe('status');
 *     }
 *   }, [isConnected]);
 *
 *   return <div>Status: {state}</div>;
 * }
 * ```
 */
export function useWebSocket(
  options?: WebSocketServiceOptions & { autoConnect?: boolean }
): UseWebSocketReturn {
  const { autoConnect = true, ...serviceOptions } = options ?? {};

  // Use a ref to hold the service to prevent re-creation on each render
  const serviceRef = useRef<WebSocketService | null>(null);

  if (!serviceRef.current) {
    serviceRef.current = getWebSocketService(serviceOptions);
  }

  const service = serviceRef.current;

  const [state, setState] = useState<ConnectionState>(service.getState());
  const [subscribedChannels, setSubscribedChannels] = useState<WebSocketChannel[]>(
    service.getSubscribedChannels()
  );

  // Update state when service state changes
  useEffect(() => {
    const handleStateChange = (newState: ConnectionState) => {
      setState(newState);
    };

    service.on('stateChange', handleStateChange);

    // Sync initial state
    setState(service.getState());
    setSubscribedChannels(service.getSubscribedChannels());

    return () => {
      service.off('stateChange', handleStateChange);
    };
  }, [service]);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect && !service.isConnected() && service.getState() === 'disconnected') {
      service.connect().catch(() => {
        // Connection failed, will auto-reconnect if enabled
      });
    }
  }, [autoConnect, service]);

  const connect = useCallback(() => service.connect(), [service]);
  const disconnect = useCallback(() => service.disconnect(), [service]);

  const subscribe = useCallback(
    (...channels: WebSocketChannel[]) => {
      service.subscribe(...channels);
      setSubscribedChannels(service.getSubscribedChannels());
    },
    [service]
  );

  const unsubscribe = useCallback(
    (...channels: WebSocketChannel[]) => {
      service.unsubscribe(...channels);
      setSubscribedChannels(service.getSubscribedChannels());
    },
    [service]
  );

  return {
    state,
    isConnected: state === 'connected',
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    subscribedChannels,
    service,
  };
}

/**
 * Hook for subscribing to WebSocket events with automatic cleanup.
 * Use this to listen for specific event types.
 *
 * @param event - Event type to listen for
 * @param handler - Callback function called when event is received
 * @param deps - Dependencies for the handler (optional)
 *
 * @example
 * ```tsx
 * function MeasurementDisplay() {
 *   const [measurements, setMeasurements] = useState<MeasurementEvent[]>([]);
 *
 *   useWebSocketEvent('measurement', (data) => {
 *     setMeasurements((prev) => [...prev, data]);
 *   });
 *
 *   return <div>{measurements.length} measurements</div>;
 * }
 * ```
 */
export function useWebSocketEvent<T>(
  event: WebSocketEventType,
  handler: WebSocketEventListener<T>,
  deps: React.DependencyList = []
): void {
  const service = getWebSocketService();

  useEffect(() => {
    service.on(event, handler);
    return () => {
      service.off(event, handler);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [service, event, ...deps]);
}

/**
 * Hook for receiving measurement events.
 * Returns the latest measurement and accumulated measurements.
 *
 * @param maxMeasurements - Maximum number of measurements to keep (default: 100)
 * @returns Latest measurement and measurement history
 *
 * @example
 * ```tsx
 * function MetricsChart() {
 *   const { latest, measurements, clear } = useMeasurementEvents(50);
 *
 *   return (
 *     <div>
 *       <p>Latest D_eff: {latest?.deff}</p>
 *       <LineChart data={measurements} />
 *       <button onClick={clear}>Clear</button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useMeasurementEvents(maxMeasurements = 100): {
  latest: MeasurementEvent | null;
  measurements: MeasurementEvent[];
  clear: () => void;
} {
  const [latest, setLatest] = useState<MeasurementEvent | null>(null);
  const [measurements, setMeasurements] = useState<MeasurementEvent[]>([]);

  useWebSocketEvent<MeasurementEvent>(
    'measurement',
    (data) => {
      setLatest(data);
      setMeasurements((prev) => {
        const next = [...prev, data];
        if (next.length > maxMeasurements) {
          return next.slice(-maxMeasurements);
        }
        return next;
      });
    },
    [maxMeasurements]
  );

  // Also handle batch events
  useWebSocketEvent<MeasurementBatchEvent>(
    'measurement_batch',
    (data) => {
      if (data.measurements.length > 0) {
        setLatest(data.measurements[data.measurements.length - 1]);
        setMeasurements((prev) => {
          const next = [...prev, ...data.measurements];
          if (next.length > maxMeasurements) {
            return next.slice(-maxMeasurements);
          }
          return next;
        });
      }
    },
    [maxMeasurements]
  );

  const clear = useCallback(() => {
    setLatest(null);
    setMeasurements([]);
  }, []);

  return { latest, measurements, clear };
}

/**
 * Hook for receiving chat message events.
 * Returns the latest message and message history.
 *
 * @param maxMessages - Maximum number of messages to keep (default: 100)
 * @returns Latest message and message history
 */
export function useChatMessages(maxMessages = 100): {
  latest: ChatMessageData | null;
  messages: ChatMessageData[];
  clear: () => void;
} {
  const [latest, setLatest] = useState<ChatMessageData | null>(null);
  const [messages, setMessages] = useState<ChatMessageData[]>([]);

  useWebSocketEvent<ChatMessageData>(
    'message',
    (data) => {
      setLatest(data);
      setMessages((prev) => {
        const next = [...prev, data];
        if (next.length > maxMessages) {
          return next.slice(-maxMessages);
        }
        return next;
      });
    },
    [maxMessages]
  );

  const clear = useCallback(() => {
    setLatest(null);
    setMessages([]);
  }, []);

  return { latest, messages, clear };
}

/**
 * Hook for tracking backend status updates.
 * Returns a map of backend names to their status.
 */
export function useBackendStatus(): {
  backends: Map<string, BackendInfo>;
  getBackend: (name: string) => BackendInfo | undefined;
} {
  const [backends, setBackends] = useState<Map<string, BackendInfo>>(new Map());

  useWebSocketEvent<BackendInfo>('backend_status', (data) => {
    setBackends((prev) => {
      const next = new Map(prev);
      next.set(data.name, data);
      return next;
    });
  });

  const getBackend = useCallback(
    (name: string) => backends.get(name),
    [backends]
  );

  return { backends, getBackend };
}

/**
 * Hook for tracking model status updates.
 * Returns a map of model names to their status.
 */
export function useModelStatus(): {
  models: Map<string, ModelInfo>;
  getModel: (name: string) => ModelInfo | undefined;
  loadedModels: ModelInfo[];
} {
  const [models, setModels] = useState<Map<string, ModelInfo>>(new Map());

  useWebSocketEvent<ModelInfo>('model_status', (data) => {
    setModels((prev) => {
      const next = new Map(prev);
      next.set(data.name, data);
      return next;
    });
  });

  const getModel = useCallback((name: string) => models.get(name), [models]);

  const loadedModels = Array.from(models.values()).filter((m) => m.loaded);

  return { models, getModel, loadedModels };
}

/**
 * Hook for tracking resource usage updates (GPU, memory, queue).
 * Returns the latest resource status from WebSocket events.
 */
export function useResourceStatus(): {
  status: ResourceUpdateData | null;
  gpuMemory: number;
  gpuUtilization: number;
  queueDepth: number;
} {
  const [status, setStatus] = useState<ResourceUpdateData | null>(null);

  useWebSocketEvent<ResourceUpdateData>('resource_update', (data) => {
    setStatus(data);
  });

  return {
    status,
    gpuMemory: status?.gpuMemory ?? 0,
    gpuUtilization: status?.gpuUtil ?? 0,
    queueDepth: status?.queueDepth ?? 0,
  };
}

/**
 * Hook for beta alert events.
 * Tracks when beta values reach concerning or critical levels.
 *
 * @param onAlert - Optional callback when an alert is received
 * @returns List of recent alerts
 */
export function useBetaAlerts(
  maxAlerts = 50,
  onAlert?: (alert: BetaAlertEvent) => void
): {
  alerts: BetaAlertEvent[];
  latest: BetaAlertEvent | null;
  hasUnread: boolean;
  markRead: () => void;
  clear: () => void;
} {
  const [alerts, setAlerts] = useState<BetaAlertEvent[]>([]);
  const [latest, setLatest] = useState<BetaAlertEvent | null>(null);
  const [hasUnread, setHasUnread] = useState(false);

  useWebSocketEvent<BetaAlertEvent>(
    'beta_alert',
    (data) => {
      setLatest(data);
      setHasUnread(true);
      setAlerts((prev) => {
        const next = [...prev, data];
        if (next.length > maxAlerts) {
          return next.slice(-maxAlerts);
        }
        return next;
      });
      onAlert?.(data);
    },
    [maxAlerts, onAlert]
  );

  const markRead = useCallback(() => setHasUnread(false), []);
  const clear = useCallback(() => {
    setAlerts([]);
    setLatest(null);
    setHasUnread(false);
  }, []);

  return { alerts, latest, hasUnread, markRead, clear };
}

/**
 * Hook for session lifecycle events.
 * Tracks when sessions start and end.
 */
export function useSessionEvents(): {
  activeSessions: Set<string>;
  onSessionStart: (callback: (event: SessionStartEvent) => void) => void;
  onSessionEnd: (callback: (event: SessionEndEvent) => void) => void;
} {
  const [activeSessions, setActiveSessions] = useState<Set<string>>(new Set());
  const startCallbacks = useRef<Set<(event: SessionStartEvent) => void>>(new Set());
  const endCallbacks = useRef<Set<(event: SessionEndEvent) => void>>(new Set());

  useWebSocketEvent<SessionStartEvent>('session_start', (data) => {
    setActiveSessions((prev) => new Set([...prev, data.sessionId]));
    startCallbacks.current.forEach((cb) => cb(data));
  });

  useWebSocketEvent<SessionEndEvent>('session_end', (data) => {
    setActiveSessions((prev) => {
      const next = new Set(prev);
      next.delete(data.sessionId);
      return next;
    });
    endCallbacks.current.forEach((cb) => cb(data));
  });

  const onSessionStart = useCallback((callback: (event: SessionStartEvent) => void) => {
    startCallbacks.current.add(callback);
    return () => {
      startCallbacks.current.delete(callback);
    };
  }, []);

  const onSessionEnd = useCallback((callback: (event: SessionEndEvent) => void) => {
    endCallbacks.current.add(callback);
    return () => {
      endCallbacks.current.delete(callback);
    };
  }, []);

  return { activeSessions, onSessionStart, onSessionEnd };
}

/**
 * Hook for conversation turn events.
 * Tracks conversation progress and participants.
 */
export function useConversationTurns(maxTurns = 100): {
  turns: ConversationTurnEvent[];
  latest: ConversationTurnEvent | null;
  currentTurn: number;
  clear: () => void;
} {
  const [turns, setTurns] = useState<ConversationTurnEvent[]>([]);
  const [latest, setLatest] = useState<ConversationTurnEvent | null>(null);

  useWebSocketEvent<ConversationTurnEvent>(
    'conversation_turn',
    (data) => {
      setLatest(data);
      setTurns((prev) => {
        const next = [...prev, data];
        if (next.length > maxTurns) {
          return next.slice(-maxTurns);
        }
        return next;
      });
    },
    [maxTurns]
  );

  const currentTurn = latest?.turn ?? 0;

  const clear = useCallback(() => {
    setTurns([]);
    setLatest(null);
  }, []);

  return { turns, latest, currentTurn, clear };
}

// Re-export types for convenience
export type {
  ConnectionState,
  WebSocketServiceOptions,
  WebSocketEventType,
  WebSocketEventListener,
  ChatMessageData,
};

export {
  WebSocketService,
  getWebSocketService,
  createWebSocketService,
};
