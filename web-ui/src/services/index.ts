/**
 * Service exports for WeaverTools web-ui.
 * Re-exports all API services for convenient imports.
 */

// Base API client
export {
  get,
  post,
  put,
  patch,
  del,
  buildQueryString,
  streamEvents,
  withRetry,
  downloadFile,
  api,
  ApiError,
  type ApiRequestOptions,
  type ApiResponse,
} from './api';

// Config API
export {
  getConfig,
  updateConfig,
  validateConfig,
  getAgentFromConfig,
  updateAgentInConfig,
  addAgentToConfig,
  removeAgentFromConfig,
  getAgentNames,
  getActiveAgents,
  configApi,
  type ConfigValidationResult,
} from './configApi';

// Agent API
export {
  listAgents,
  getAgent,
  isAgentAvailable,
  chat,
  sendMessage,
  chatStream,
  parseAgentMention,
  formatAgentMention,
  agentApi,
  type AgentInfo,
  type ChatRequestPayload,
  type ChatStreamEvent,
  type ChatResponseNormalized,
} from './agentApi';

// Session API
export {
  listSessions,
  getSession,
  createSession,
  updateSession,
  deleteSession,
  endSession,
  getSessionMessages,
  sessionExists,
  isSessionActive,
  getSessionDuration,
  formatDuration,
  sessionApi,
  type SessionListOptions,
  type SessionSummary,
  type CreateSessionRequest,
  type UpdateSessionRequest,
} from './sessionApi';

// Export API
export {
  exportLatex,
  exportCsv,
  exportPdf,
  exportBibtex,
  exportMeasurements,
  downloadExport,
  exportAndDownload,
  getExtensionForFormat,
  getMimeTypeForFormat,
  generateFilename,
  getFormatDisplayName,
  MEASUREMENT_COLUMNS,
  DEFAULT_COLUMNS,
  exportApi,
  type ExportFormat,
  type LatexTableStyle,
  type CsvDialect,
  type BaseExportOptions,
  type LatexExportOptions,
  type CsvExportOptions,
  type PdfExportOptions,
  type BibtexExportOptions,
  type ExportResponse,
} from './exportApi';

// Backend API
export {
  listBackends,
  getBackend,
  isBackendAvailable,
  getBackendCapabilities,
  supportsStreaming,
  supportsHiddenStates,
  backendApi,
  type BackendStatus,
} from './backendApi';

// Model API
export {
  listModels,
  getModel,
  loadModel,
  unloadModel,
  isModelLoaded,
  getLoadedModels,
  getAvailableModels,
  formatModelSize,
  formatParameterCount,
  loadModelWithRetry,
  modelApi,
  type ModelInfo,
  type ModelLoadOptions,
} from './modelApi';

// WebSocket API
export {
  WebSocketService,
  getWebSocketService,
  createWebSocketService,
  disposeWebSocketService,
  type ConnectionState,
  type WebSocketEventType,
  type WebSocketEventListener,
  type WebSocketServiceOptions,
  type ChatMessageData,
  type ResourceUpdateData,
} from './websocket';

// Concept API
export {
  listConcepts,
  getConcept,
  deleteConcept,
  getConceptStoreStats,
  conceptExists,
  conceptApi,
} from './conceptApi';

// Resource API
export {
  getGPUs,
  resourceApi,
  type GPUListResponse,
} from './resourceApi';
