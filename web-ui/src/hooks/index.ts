/**
 * Hook exports for WeaverTools web-ui.
 * Re-exports all custom hooks for convenient imports.
 */

// Chat hooks
export {
  useChat,
  useChatWithAgent,
  type UseChatOptions,
  type UseChatReturn,
  type UseChatWithAgentOptions,
  type UseChatWithAgentReturn,
} from './useChat';

// Measurement hooks
export {
  useMeasurements,
  useMetric,
  useBetaMonitor,
  METRIC_TYPES,
  type MetricType,
  type UseMeasurementsOptions,
  type UseMeasurementsReturn,
  type MetricStats,
} from './useMeasurements';

// WebSocket hooks
export {
  useWebSocket,
  useWebSocketEvent,
  useMeasurementEvents,
  useChatMessages,
  useBackendStatus,
  useModelStatus,
  useResourceStatus,
  useBetaAlerts,
  useSessionEvents,
  useConversationTurns,
  type UseWebSocketReturn,
  type ResourceUpdateData,
  // Re-export types from websocket service for convenience
  type ConnectionState,
  type WebSocketServiceOptions,
  type WebSocketEventType,
  type WebSocketEventListener,
  type ChatMessageData,
  // Re-export service utilities
  WebSocketService,
  getWebSocketService,
  createWebSocketService,
} from './useWebSocket';

// Model management hooks
export {
  useModels,
  useModelLoad,
  useModelUnload,
  useModelFormatters,
  type ModelFilter,
  type ModelSortBy,
  type ModelSortOrder,
  type ModelActionState,
  type UseModelsOptions,
  type UseModelsReturn,
  type ModelStats,
} from './useModels';
