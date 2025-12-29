/**
 * Download utilities for WeaverTools web-ui.
 *
 * Provides comprehensive download handling for all export formats
 * with progress tracking, error handling, and format-specific options.
 */

import type { ExportFormat, ExportResponse } from '@/services/exportApi';

/**
 * Download options for file exports.
 */
export interface DownloadOptions {
  /** Override the default filename */
  filename?: string;
  /** Callback for download progress (0-100) */
  onProgress?: (progress: number) => void;
  /** Callback when download starts */
  onStart?: () => void;
  /** Callback when download completes */
  onComplete?: (filename: string) => void;
  /** Callback when download fails */
  onError?: (error: Error) => void;
}

/**
 * Download result information.
 */
export interface DownloadResult {
  /** Whether download succeeded */
  success: boolean;
  /** Downloaded filename */
  filename: string;
  /** File size in bytes */
  size: number;
  /** Time taken in milliseconds */
  duration: number;
  /** Error message if failed */
  error?: string;
}

/**
 * MIME types for export formats.
 */
export const FORMAT_MIME_TYPES: Record<ExportFormat, string> = {
  latex: 'application/x-latex',
  csv: 'text/csv',
  pdf: 'application/x-latex', // PDF export returns LaTeX for compilation
  bibtex: 'application/x-bibtex',
};

/**
 * File extensions for export formats.
 */
export const FORMAT_EXTENSIONS: Record<ExportFormat, string> = {
  latex: '.tex',
  csv: '.csv',
  pdf: '.tex',
  bibtex: '.bib',
};

/**
 * Create a Blob from export content with proper MIME type.
 * @param content - Export content string
 * @param format - Export format
 * @returns Blob with correct MIME type
 */
export function createExportBlob(content: string, format: ExportFormat): Blob {
  const mimeType = FORMAT_MIME_TYPES[format] ?? 'text/plain';
  return new Blob([content], { type: mimeType });
}

/**
 * Trigger browser download for a Blob.
 * @param blob - Blob to download
 * @param filename - Download filename
 */
export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  link.style.display = 'none';
  document.body.appendChild(link);
  link.click();
  // Cleanup after a short delay to ensure download starts
  setTimeout(() => {
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }, 100);
}

/**
 * Download export result as a file.
 * @param result - Export response from API
 * @param options - Download options
 * @returns Download result information
 */
export function downloadExportResult(
  result: ExportResponse,
  options: DownloadOptions = {}
): DownloadResult {
  const startTime = Date.now();
  const filename = options.filename ?? result.filename;

  try {
    options.onStart?.();
    options.onProgress?.(0);

    const blob = createExportBlob(result.content, result.format);

    options.onProgress?.(50);

    triggerBlobDownload(blob, filename);

    options.onProgress?.(100);
    options.onComplete?.(filename);

    return {
      success: true,
      filename,
      size: blob.size,
      duration: Date.now() - startTime,
    };
  } catch (error) {
    const errorMessage = error instanceof Error ? error.message : 'Download failed';
    options.onError?.(error instanceof Error ? error : new Error(errorMessage));

    return {
      success: false,
      filename,
      size: 0,
      duration: Date.now() - startTime,
      error: errorMessage,
    };
  }
}

/**
 * Download content directly as a file.
 * @param content - Content to download
 * @param filename - Download filename
 * @param format - Export format for MIME type
 * @param options - Download options
 * @returns Download result
 */
export function downloadContent(
  content: string,
  filename: string,
  format: ExportFormat,
  options: DownloadOptions = {}
): DownloadResult {
  const result: ExportResponse = {
    content,
    filename,
    format,
    mimeType: FORMAT_MIME_TYPES[format] ?? 'text/plain',
  };
  return downloadExportResult(result, options);
}

/**
 * Download LaTeX content as a .tex file.
 * @param content - LaTeX content
 * @param filename - Filename (with or without extension)
 * @param options - Download options
 * @returns Download result
 */
export function downloadLatex(
  content: string,
  filename: string,
  options: DownloadOptions = {}
): DownloadResult {
  const finalFilename = filename.endsWith('.tex') ? filename : `${filename}.tex`;
  return downloadContent(content, finalFilename, 'latex', options);
}

/**
 * Download CSV content as a .csv file.
 * @param content - CSV content
 * @param filename - Filename (with or without extension)
 * @param options - Download options
 * @returns Download result
 */
export function downloadCsv(
  content: string,
  filename: string,
  options: DownloadOptions = {}
): DownloadResult {
  const finalFilename = filename.endsWith('.csv') ? filename : `${filename}.csv`;
  return downloadContent(content, finalFilename, 'csv', options);
}

/**
 * Download BibTeX content as a .bib file.
 * @param content - BibTeX content
 * @param filename - Filename (with or without extension)
 * @param options - Download options
 * @returns Download result
 */
export function downloadBibtex(
  content: string,
  filename: string,
  options: DownloadOptions = {}
): DownloadResult {
  const finalFilename = filename.endsWith('.bib') ? filename : `${filename}.bib`;
  return downloadContent(content, finalFilename, 'bibtex', options);
}

/**
 * Download PDF-compatible LaTeX content as a .tex file.
 * @param content - LaTeX content for PDF compilation
 * @param filename - Filename (with or without extension)
 * @param options - Download options
 * @returns Download result
 */
export function downloadPdfLatex(
  content: string,
  filename: string,
  options: DownloadOptions = {}
): DownloadResult {
  const finalFilename = filename.endsWith('.tex') ? filename : `${filename}.tex`;
  return downloadContent(content, finalFilename, 'pdf', options);
}

/**
 * Download export result based on format.
 * Routes to appropriate format-specific handler.
 * @param result - Export response from API
 * @param options - Download options
 * @returns Download result
 */
export function downloadByFormat(
  result: ExportResponse,
  options: DownloadOptions = {}
): DownloadResult {
  const filename = options.filename ?? result.filename;

  switch (result.format) {
    case 'latex':
      return downloadLatex(result.content, filename, options);
    case 'csv':
      return downloadCsv(result.content, filename, options);
    case 'bibtex':
      return downloadBibtex(result.content, filename, options);
    case 'pdf':
      return downloadPdfLatex(result.content, filename, options);
    default:
      return downloadExportResult(result, options);
  }
}

/**
 * Batch download results for multiple exports.
 */
export interface BatchDownloadOptions extends DownloadOptions {
  /** Delay between downloads in milliseconds */
  delay?: number;
  /** Callback for batch progress (completed, total) */
  onBatchProgress?: (completed: number, total: number) => void;
}

/**
 * Batch download result.
 */
export interface BatchDownloadResult {
  /** Total exports to download */
  total: number;
  /** Successful downloads */
  successful: number;
  /** Failed downloads */
  failed: number;
  /** Individual results */
  results: DownloadResult[];
  /** Total time in milliseconds */
  totalDuration: number;
}

/**
 * Download multiple export results sequentially.
 * @param exports - Array of export responses
 * @param options - Batch download options
 * @returns Batch download result
 */
export async function downloadBatch(
  exports: ExportResponse[],
  options: BatchDownloadOptions = {}
): Promise<BatchDownloadResult> {
  const startTime = Date.now();
  const results: DownloadResult[] = [];
  const delay = options.delay ?? 500;

  options.onStart?.();
  options.onBatchProgress?.(0, exports.length);

  for (let i = 0; i < exports.length; i++) {
    const exportResult = exports[i];
    const result = downloadByFormat(exportResult, {
      onError: options.onError,
    });
    results.push(result);

    options.onBatchProgress?.(i + 1, exports.length);
    options.onProgress?.(Math.round(((i + 1) / exports.length) * 100));

    // Delay between downloads to avoid browser throttling
    if (i < exports.length - 1 && delay > 0) {
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }

  const successful = results.filter((r) => r.success).length;
  const failed = results.filter((r) => !r.success).length;

  options.onComplete?.(`${successful} of ${exports.length} files`);

  return {
    total: exports.length,
    successful,
    failed,
    results,
    totalDuration: Date.now() - startTime,
  };
}

/**
 * Generate a safe filename from session name.
 * @param sessionName - Original session name
 * @returns Sanitized filename (without extension)
 */
export function sanitizeFilename(sessionName: string): string {
  return sessionName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .substring(0, 100); // Limit length
}

/**
 * Generate a timestamped filename.
 * @param baseName - Base name for the file
 * @param format - Export format for extension
 * @returns Filename with timestamp
 */
export function generateTimestampedFilename(
  baseName: string,
  format: ExportFormat
): string {
  const sanitized = sanitizeFilename(baseName);
  const timestamp = new Date().toISOString().slice(0, 10);
  const extension = FORMAT_EXTENSIONS[format] ?? '.txt';
  return `${sanitized}-${timestamp}${extension}`;
}

/**
 * Copy export content to clipboard.
 * @param content - Content to copy
 * @returns Promise resolving when copy completes
 */
export async function copyToClipboard(content: string): Promise<void> {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    await navigator.clipboard.writeText(content);
  } else {
    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = content;
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    document.body.removeChild(textarea);
  }
}

/**
 * Check if download is supported in current environment.
 * @returns True if download is supported
 */
export function isDownloadSupported(): boolean {
  if (typeof document === 'undefined') {
    return false;
  }
  const link = document.createElement('a');
  return 'download' in link;
}

/**
 * Get human-readable file size.
 * @param bytes - Size in bytes
 * @returns Formatted size string
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  const size = bytes / Math.pow(1024, i);
  return `${size.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

/**
 * Estimate file size from content.
 * @param content - String content
 * @returns Estimated size in bytes
 */
export function estimateContentSize(content: string): number {
  return new Blob([content]).size;
}

/**
 * Download utilities namespace.
 */
export const downloadUtils = {
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
};

export default downloadUtils;
