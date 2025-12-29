/**
 * Utility exports for Weaver web UI.
 */

// Message parsing utilities
export {
  parseMessage,
  findAgentMentions,
  isValidAgentName,
  extractTargetAgent,
  stripAgentPrefix,
  formatAgentMessage,
  hasAgentMention,
  highlightAgentMentions,
  suggestAgents,
  getAutocompleteContext,
  type ParsedMessage,
  type AgentMention,
  type TextSegment,
  type AutocompleteContext,
} from './messageParser';

// Download utilities
export {
  // Core functions
  createExportBlob,
  triggerBlobDownload,
  downloadExportResult,
  downloadContent,
  downloadByFormat,
  // Format-specific
  downloadLatex,
  downloadCsv,
  downloadBibtex,
  downloadPdfLatex,
  // Batch operations
  downloadBatch,
  // Helpers
  sanitizeFilename,
  generateTimestampedFilename,
  copyToClipboard,
  isDownloadSupported,
  formatFileSize,
  estimateContentSize,
  // Constants
  FORMAT_MIME_TYPES,
  FORMAT_EXTENSIONS,
  // Types
  type DownloadOptions,
  type DownloadResult,
  type BatchDownloadOptions,
  type BatchDownloadResult,
  // Namespace
  downloadUtils,
} from './download';
