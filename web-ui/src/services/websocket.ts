/**
 * WebSocket service for real-time updates from the Weaver backend.
 * Provides connection management with auto-reconnect and channel-based subscriptions.
 */

import type {
  WebSocketChannel,
  WebSocketMessageType,
  WebSocketMessage,
  MeasurementEvent,
  MeasurementBatchEvent,
  BetaAlertEvent,
  ConversationTurnEvent,
  SessionStartEvent,
  SessionEndEvent,
} from '@/types';

import type { BackendInfo, ModelInfo } from '@/types';

/**
 * ResourceUpdateData matches the Go ResourceUpdateData struct in websocket.go.
 * This is specific to WebSocket events.
 */
export interface ResourceUpdateData {
  gpuMemory: number;
  gpuUtil: number;
  queueDepth: number;
}

/** Default WebSocket URL - can be overridden with environment variable */
const WS_BASE_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8081/ws';

/** Connection state */
export type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'reconnecting';

/** WebSocket event types for listeners */
export type WebSocketEventType =
  | 'open'
  | 'close'
  | 'error'
  | 'message'
  | 'stateChange'
  | 'measurement'
  | 'measurement_batch'
  | 'message'
  | 'backend_status'
  | 'model_status'
  | 'resource_update'
  | 'session_start'
  | 'session_end'
  | 'conversation_turn'
  | 'beta_alert'
  | 'pong';

/** Event listener type */
export type WebSocketEventListener<T = unknown> = (data: T) => void;

/** Options for WebSocket service */
export interface WebSocketServiceOptions {
  /** WebSocket URL (defaults to VITE_WS_URL env var or localhost) */
  url?: string;
  /** Auto-reconnect on disconnect (default: true) */
  autoReconnect?: boolean;
  /** Initial reconnect delay in ms (default: 1000) */
  reconnectDelay?: number;
  /** Maximum reconnect delay in ms (default: 30000) */
  maxReconnectDelay?: number;
  /** Reconnect delay multiplier (default: 2) */
  reconnectMultiplier?: number;
  /** Ping interval in ms (default: 30000) */
  pingInterval?: number;
  /** Ping timeout in ms (default: 5000) */
  pingTimeout?: number;
  /** Channels to subscribe to on connect */
  initialChannels?: WebSocketChannel[];
}

/** Message data types for each event type */
export interface ChatMessageData {
  agent: string;
  content: string;
  turn: number;
}

/**
 * WebSocketService provides a connection to the Weaver backend WebSocket server.
 * Features:
 * - Auto-reconnect with exponential backoff
 * - Channel-based subscriptions
 * - Ping/pong heartbeat
 * - Typed event listeners
 */
export class WebSocketService {
  private ws: WebSocket | null = null;
  private url: string;
  private options: Required<WebSocketServiceOptions>;
  private state: ConnectionState = 'disconnected';
  private reconnectAttempts = 0;
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null;
  private pingInterval: ReturnType<typeof setInterval> | null = null;
  private pingTimeout: ReturnType<typeof setTimeout> | null = null;
  private pendingPong = false;
  private subscribedChannels: Set<WebSocketChannel> = new Set();
  private listeners: Map<WebSocketEventType, Set<WebSocketEventListener<unknown>>> = new Map();
  private manualClose = false;

  constructor(options: WebSocketServiceOptions = {}) {
    this.url = options.url || WS_BASE_URL;
    this.options = {
      url: this.url,
      autoReconnect: options.autoReconnect ?? true,
      reconnectDelay: options.reconnectDelay ?? 1000,
      maxReconnectDelay: options.maxReconnectDelay ?? 30000,
      reconnectMultiplier: options.reconnectMultiplier ?? 2,
      pingInterval: options.pingInterval ?? 30000,
      pingTimeout: options.pingTimeout ?? 5000,
      initialChannels: options.initialChannels ?? [],
    };

    // Subscribe to initial channels
    for (const channel of this.options.initialChannels) {
      this.subscribedChannels.add(channel);
    }
  }

  /** Get current connection state */
  getState(): ConnectionState {
    return this.state;
  }

  /** Check if connected */
  isConnected(): boolean {
    return this.state === 'connected' && this.ws?.readyState === WebSocket.OPEN;
  }

  /** Get subscribed channels */
  getSubscribedChannels(): WebSocketChannel[] {
    return Array.from(this.subscribedChannels);
  }

  /**
   * Connect to the WebSocket server.
   * @returns Promise that resolves when connected or rejects on error
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        resolve();
        return;
      }

      this.manualClose = false;
      this.setState('connecting');

      try {
        this.ws = new WebSocket(this.url);
      } catch (error) {
        this.setState('disconnected');
        reject(error);
        return;
      }

      const onOpen = () => {
        this.ws?.removeEventListener('error', onError);
        this.handleOpen();
        resolve();
      };

      const onError = (event: Event) => {
        this.ws?.removeEventListener('open', onOpen);
        this.setState('disconnected');
        reject(new Error('WebSocket connection failed'));
      };

      this.ws.addEventListener('open', onOpen, { once: true });
      this.ws.addEventListener('error', onError, { once: true });
      this.ws.addEventListener('close', this.handleClose.bind(this));
      this.ws.addEventListener('message', this.handleMessage.bind(this));
      this.ws.addEventListener('error', this.handleError.bind(this));
    });
  }

  /** Disconnect from the WebSocket server */
  disconnect(): void {
    this.manualClose = true;
    this.cleanup();
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.setState('disconnected');
  }

  /**
   * Subscribe to a channel for receiving events.
   * @param channels - Channels to subscribe to
   */
  subscribe(...channels: WebSocketChannel[]): void {
    const newChannels: WebSocketChannel[] = [];

    for (const channel of channels) {
      if (!this.subscribedChannels.has(channel)) {
        this.subscribedChannels.add(channel);
        newChannels.push(channel);
      }
    }

    // Send subscribe message if connected and have new channels
    if (this.isConnected() && newChannels.length > 0) {
      this.sendSubscribe(newChannels);
    }
  }

  /**
   * Unsubscribe from channels.
   * @param channels - Channels to unsubscribe from
   */
  unsubscribe(...channels: WebSocketChannel[]): void {
    for (const channel of channels) {
      this.subscribedChannels.delete(channel);
    }
    // Note: Server doesn't have an unsubscribe message, but we track locally
  }

  /**
   * Add an event listener.
   * @param event - Event type to listen for
   * @param listener - Callback function
   */
  on<T = unknown>(event: WebSocketEventType, listener: WebSocketEventListener<T>): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(listener as WebSocketEventListener<unknown>);
  }

  /**
   * Remove an event listener.
   * @param event - Event type
   * @param listener - Callback function to remove
   */
  off<T = unknown>(event: WebSocketEventType, listener: WebSocketEventListener<T>): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      listeners.delete(listener as WebSocketEventListener<unknown>);
    }
  }

  /**
   * Add a one-time event listener.
   * @param event - Event type to listen for
   * @param listener - Callback function (called once then removed)
   */
  once<T = unknown>(event: WebSocketEventType, listener: WebSocketEventListener<T>): void {
    const wrapper: WebSocketEventListener<T> = (data) => {
      this.off(event, wrapper);
      listener(data);
    };
    this.on(event, wrapper);
  }

  /** Remove all listeners for an event or all events */
  removeAllListeners(event?: WebSocketEventType): void {
    if (event) {
      this.listeners.delete(event);
    } else {
      this.listeners.clear();
    }
  }

  // ---------------------------------------------------------------------------
  // Private methods
  // ---------------------------------------------------------------------------

  private setState(state: ConnectionState): void {
    if (this.state !== state) {
      this.state = state;
      this.emit('stateChange', state);
    }
  }

  private emit<T>(event: WebSocketEventType, data: T): void {
    const listeners = this.listeners.get(event);
    if (listeners) {
      for (const listener of listeners) {
        try {
          listener(data);
        } catch (error) {
          // Don't let listener errors break the event loop
        }
      }
    }
  }

  private handleOpen(): void {
    this.setState('connected');
    this.reconnectAttempts = 0;
    this.startPingInterval();
    this.emit('open', undefined);

    // Subscribe to channels
    if (this.subscribedChannels.size > 0) {
      this.sendSubscribe(Array.from(this.subscribedChannels));
    }
  }

  private handleClose(event: CloseEvent): void {
    this.cleanup();
    this.emit('close', { code: event.code, reason: event.reason });

    if (!this.manualClose && this.options.autoReconnect) {
      this.scheduleReconnect();
    } else {
      this.setState('disconnected');
    }
  }

  private handleError(event: Event): void {
    this.emit('error', new Error('WebSocket error'));
  }

  private handleMessage(event: MessageEvent): void {
    try {
      // Handle multiple messages (newline-separated)
      const messages = String(event.data).split('\n').filter(Boolean);

      for (const msgStr of messages) {
        const msg = JSON.parse(msgStr) as WebSocketMessage;
        this.emit('message', msg);
        this.dispatchMessage(msg);
      }
    } catch (error) {
      // Failed to parse message
    }
  }

  private dispatchMessage(msg: WebSocketMessage): void {
    switch (msg.type) {
      case 'measurement':
        this.emit('measurement', msg.data as MeasurementEvent);
        break;
      case 'measurement_batch':
        this.emit('measurement_batch', msg.data as MeasurementBatchEvent);
        break;
      case 'message':
        this.emit('message', msg.data as ChatMessageData);
        break;
      case 'backend_status':
        this.emit('backend_status', msg.data as BackendInfo);
        break;
      case 'model_status':
        this.emit('model_status', msg.data as ModelInfo);
        break;
      case 'resource_update':
        this.emit('resource_update', msg.data as ResourceUpdateData);
        break;
      case 'session_start':
        this.emit('session_start', msg.data as SessionStartEvent);
        break;
      case 'session_end':
        this.emit('session_end', msg.data as SessionEndEvent);
        break;
      case 'conversation_turn':
        this.emit('conversation_turn', msg.data as ConversationTurnEvent);
        break;
      case 'beta_alert':
        this.emit('beta_alert', msg.data as BetaAlertEvent);
        break;
      case 'pong':
        this.handlePong();
        break;
    }
  }

  private sendSubscribe(channels: WebSocketChannel[]): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const msg = {
      type: 'subscribe',
      channels,
      timestamp: new Date().toISOString(),
    };

    try {
      this.ws.send(JSON.stringify(msg));
    } catch (error) {
      // Failed to send subscribe message
    }
  }

  private sendPing(): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      return;
    }

    const msg = {
      type: 'ping',
      timestamp: new Date().toISOString(),
    };

    try {
      this.ws.send(JSON.stringify(msg));
      this.pendingPong = true;

      // Set timeout for pong response
      this.pingTimeout = setTimeout(() => {
        if (this.pendingPong) {
          // No pong received, connection may be dead
          this.ws?.close(4000, 'Ping timeout');
        }
      }, this.options.pingTimeout);
    } catch (error) {
      // Failed to send ping
    }
  }

  private handlePong(): void {
    this.pendingPong = false;
    if (this.pingTimeout) {
      clearTimeout(this.pingTimeout);
      this.pingTimeout = null;
    }
    this.emit('pong', undefined);
  }

  private startPingInterval(): void {
    this.stopPingInterval();
    this.pingInterval = setInterval(() => {
      this.sendPing();
    }, this.options.pingInterval);
  }

  private stopPingInterval(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    if (this.pingTimeout) {
      clearTimeout(this.pingTimeout);
      this.pingTimeout = null;
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) {
      return;
    }

    this.setState('reconnecting');
    this.reconnectAttempts++;

    // Calculate delay with exponential backoff
    const delay = Math.min(
      this.options.reconnectDelay *
        Math.pow(this.options.reconnectMultiplier, this.reconnectAttempts - 1),
      this.options.maxReconnectDelay
    );

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectTimeout = null;
      this.connect().catch(() => {
        // Reconnect failed, will retry on next close event
      });
    }, delay);
  }

  private cleanup(): void {
    this.stopPingInterval();
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }
}

// ---------------------------------------------------------------------------
// Singleton instance
// ---------------------------------------------------------------------------

/** Default WebSocket service instance */
let defaultInstance: WebSocketService | null = null;

/**
 * Get the default WebSocket service instance.
 * Creates one if it doesn't exist.
 */
export function getWebSocketService(options?: WebSocketServiceOptions): WebSocketService {
  if (!defaultInstance) {
    defaultInstance = new WebSocketService(options);
  }
  return defaultInstance;
}

/**
 * Create a new WebSocket service instance.
 * Use this if you need multiple independent connections.
 */
export function createWebSocketService(options?: WebSocketServiceOptions): WebSocketService {
  return new WebSocketService(options);
}

/**
 * Dispose the default WebSocket service instance.
 * Call this on app shutdown to clean up resources.
 */
export function disposeWebSocketService(): void {
  if (defaultInstance) {
    defaultInstance.disconnect();
    defaultInstance.removeAllListeners();
    defaultInstance = null;
  }
}

export default WebSocketService;
