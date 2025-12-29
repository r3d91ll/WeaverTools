/**
 * Export API service for WeaverTools web-ui.
 * Handles export generation for various formats.
 */

import { post, downloadFile, ApiError } from './api';
import type { Measurement } from '@/types';

/** API endpoints for export operations */
const ENDPOINTS = {
  latex: '/api/export/latex',
  csv: '/api/export/csv',
  pdf: '/api/export/pdf',
  bibtex: '/api/export/bibtex',
} as const;

/** Supported export formats */
export type ExportFormat = 'latex' | 'csv' | 'pdf' | 'bibtex';

/** LaTeX table style options */
export type LatexTableStyle = 'booktabs' | 'plain';

/** CSV dialect options */
export type CsvDialect = 'standard' | 'excel' | 'tsv';

/** Base export options common to all formats */
export interface BaseExportOptions {
  /** Session ID to export */
  sessionId?: string;
  /** Specific measurements to export */
  measurements?: Measurement[];
  /** Include summary statistics */
  includeSummary?: boolean;
}

/** LaTeX export options */
export interface LatexExportOptions extends BaseExportOptions {
  /** Table style */
  style?: LatexTableStyle;
  /** Table caption */
  caption?: string;
  /** Table label for cross-references */
  label?: string;
  /** Columns to include */
  columns?: string[];
}

/** CSV export options */
export interface CsvExportOptions extends BaseExportOptions {
  /** CSV dialect */
  dialect?: CsvDialect;
  /** Include header row */
  includeHeaders?: boolean;
  /** Columns to include */
  columns?: string[];
}

/** PDF export options */
export interface PdfExportOptions extends BaseExportOptions {
  /** Document title */
  title?: string;
  /** Document author */
  author?: string;
  /** Include charts */
  includeCharts?: boolean;
  /** Page size */
  pageSize?: 'a4' | 'letter';
}

/** BibTeX export options */
export interface BibtexExportOptions extends BaseExportOptions {
  /** Entry type (article, inproceedings, etc.) */
  entryType?: string;
  /** Citation key prefix */
  keyPrefix?: string;
}

/** Export response from API */
export interface ExportResponse {
  /** Exported content */
  content: string;
  /** Content format */
  format: ExportFormat;
  /** Suggested filename */
  filename: string;
  /** MIME type */
  mimeType: string;
  /** Export statistics */
  stats?: {
    measurementCount: number;
    rowCount: number;
    generatedAt: string;
  };
}

/** API request structure */
interface ExportApiRequest {
  sessionId?: string;
  measurements?: Measurement[];
  options?: Record<string, unknown>;
}

/**
 * Export measurements to LaTeX format.
 * @param options - Export options
 * @returns Promise resolving to export response
 * @throws ApiError on failure
 */
export async function exportLatex(
  options: LatexExportOptions
): Promise<ExportResponse> {
  const request: ExportApiRequest = {
    sessionId: options.sessionId,
    measurements: options.measurements,
    options: {
      style: options.style ?? 'booktabs',
      caption: options.caption,
      label: options.label,
      columns: options.columns,
      includeSummary: options.includeSummary,
    },
  };

  const response = await post<ExportResponse>(ENDPOINTS.latex, request);
  return response.data;
}

/**
 * Export measurements to CSV format.
 * @param options - Export options
 * @returns Promise resolving to export response
 * @throws ApiError on failure
 */
export async function exportCsv(options: CsvExportOptions): Promise<ExportResponse> {
  const request: ExportApiRequest = {
    sessionId: options.sessionId,
    measurements: options.measurements,
    options: {
      dialect: options.dialect ?? 'standard',
      includeHeaders: options.includeHeaders ?? true,
      columns: options.columns,
      includeSummary: options.includeSummary,
    },
  };

  const response = await post<ExportResponse>(ENDPOINTS.csv, request);
  return response.data;
}

/**
 * Export session to PDF format.
 * @param options - Export options
 * @returns Promise resolving to export response
 * @throws ApiError on failure
 */
export async function exportPdf(options: PdfExportOptions): Promise<ExportResponse> {
  const request: ExportApiRequest = {
    sessionId: options.sessionId,
    measurements: options.measurements,
    options: {
      title: options.title,
      author: options.author,
      includeCharts: options.includeCharts ?? false,
      pageSize: options.pageSize ?? 'a4',
      includeSummary: options.includeSummary,
    },
  };

  const response = await post<ExportResponse>(ENDPOINTS.pdf, request);
  return response.data;
}

/**
 * Export session to BibTeX format.
 * @param options - Export options
 * @returns Promise resolving to export response
 * @throws ApiError on failure
 */
export async function exportBibtex(
  options: BibtexExportOptions
): Promise<ExportResponse> {
  const request: ExportApiRequest = {
    sessionId: options.sessionId,
    measurements: options.measurements,
    options: {
      entryType: options.entryType ?? 'misc',
      keyPrefix: options.keyPrefix ?? 'weaver',
      includeSummary: options.includeSummary,
    },
  };

  const response = await post<ExportResponse>(ENDPOINTS.bibtex, request);
  return response.data;
}

/**
 * Export measurements to a specified format.
 * @param format - Export format
 * @param options - Format-specific options
 * @returns Promise resolving to export response
 * @throws ApiError on failure
 */
export async function exportMeasurements(
  format: ExportFormat,
  options: BaseExportOptions &
    (LatexExportOptions | CsvExportOptions | PdfExportOptions | BibtexExportOptions)
): Promise<ExportResponse> {
  switch (format) {
    case 'latex':
      return exportLatex(options as LatexExportOptions);
    case 'csv':
      return exportCsv(options as CsvExportOptions);
    case 'pdf':
      return exportPdf(options as PdfExportOptions);
    case 'bibtex':
      return exportBibtex(options as BibtexExportOptions);
    default:
      throw new Error(`Unsupported export format: ${format}`);
  }
}

/**
 * Download an export result as a file.
 * @param exportResult - Export response from API
 */
export function downloadExport(exportResult: ExportResponse): void {
  const blob = new Blob([exportResult.content], { type: exportResult.mimeType });
  downloadFile(blob, exportResult.filename);
}

/**
 * Export and immediately download.
 * @param format - Export format
 * @param options - Export options
 * @throws ApiError on failure
 */
export async function exportAndDownload(
  format: ExportFormat,
  options: BaseExportOptions
): Promise<void> {
  const result = await exportMeasurements(format, options);
  downloadExport(result);
}

/**
 * Get file extension for export format.
 * @param format - Export format
 * @returns File extension including dot
 */
export function getExtensionForFormat(format: ExportFormat): string {
  switch (format) {
    case 'latex':
      return '.tex';
    case 'csv':
      return '.csv';
    case 'pdf':
      return '.tex'; // PDF export returns LaTeX for compilation
    case 'bibtex':
      return '.bib';
    default:
      return '.txt';
  }
}

/**
 * Get MIME type for export format.
 * @param format - Export format
 * @returns MIME type string
 */
export function getMimeTypeForFormat(format: ExportFormat): string {
  switch (format) {
    case 'latex':
    case 'pdf':
      return 'application/x-latex';
    case 'csv':
      return 'text/csv';
    case 'bibtex':
      return 'application/x-bibtex';
    default:
      return 'text/plain';
  }
}

/**
 * Generate a default filename for export.
 * @param sessionName - Session name
 * @param format - Export format
 * @returns Generated filename
 */
export function generateFilename(sessionName: string, format: ExportFormat): string {
  const sanitized = sessionName
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
  const date = new Date().toISOString().slice(0, 10);
  const ext = getExtensionForFormat(format);
  return `${sanitized}-${date}${ext}`;
}

/**
 * Get human-readable format name.
 * @param format - Export format
 * @returns Display name
 */
export function getFormatDisplayName(format: ExportFormat): string {
  switch (format) {
    case 'latex':
      return 'LaTeX';
    case 'csv':
      return 'CSV';
    case 'pdf':
      return 'PDF (LaTeX)';
    case 'bibtex':
      return 'BibTeX';
    default:
      return format.toUpperCase();
  }
}

/**
 * Available columns for measurement export.
 */
export const MEASUREMENT_COLUMNS = [
  'turn_number',
  'sender_name',
  'receiver_name',
  'd_eff',
  'beta',
  'alignment',
  'c_pair',
  'beta_status',
  'is_unilateral',
  'timestamp',
] as const;

/** Default columns for export */
export const DEFAULT_COLUMNS: (typeof MEASUREMENT_COLUMNS)[number][] = [
  'turn_number',
  'sender_name',
  'receiver_name',
  'd_eff',
  'beta',
  'alignment',
  'c_pair',
];

/** Export API service object */
export const exportApi = {
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
};

export default exportApi;
