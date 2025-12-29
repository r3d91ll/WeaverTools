/**
 * ExportForm component for configuring and generating exports.
 *
 * Provides format-specific options and preview capabilities
 * for exporting session measurements in various formats.
 */
import { useState, useCallback, useEffect } from 'react';
import { FormatSelector } from './FormatSelector';
import {
  exportMeasurements,
  downloadExport,
  getFormatDisplayName,
  generateFilename,
  MEASUREMENT_COLUMNS,
  DEFAULT_COLUMNS,
  type ExportFormat,
  type ExportResponse,
  type LatexExportOptions,
  type CsvExportOptions,
  type PdfExportOptions,
  type BibtexExportOptions,
  type LatexTableStyle,
  type CsvDialect,
} from '@/services/exportApi';
import { listSessions, type SessionSummary } from '@/services/sessionApi';

/**
 * ExportForm component props.
 */
export interface ExportFormProps {
  /** Pre-selected session ID */
  sessionId?: string;
  /** Callback when export completes successfully */
  onExportComplete?: (result: ExportResponse) => void;
  /** Callback when export fails */
  onExportError?: (error: Error) => void;
  /** Whether to show preview by default */
  showPreview?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Form state for export options.
 */
interface FormState {
  format: ExportFormat;
  sessionId: string;
  includeSummary: boolean;
  columns: string[];
  // LaTeX-specific
  latexStyle: LatexTableStyle;
  caption: string;
  label: string;
  // CSV-specific
  csvDialect: CsvDialect;
  includeHeaders: boolean;
  // PDF-specific
  title: string;
  author: string;
  includeCharts: boolean;
  pageSize: 'a4' | 'letter';
  // BibTeX-specific
  entryType: string;
  keyPrefix: string;
}

/**
 * Default form state.
 */
const defaultFormState: FormState = {
  format: 'latex',
  sessionId: '',
  includeSummary: true,
  columns: [...DEFAULT_COLUMNS],
  // LaTeX
  latexStyle: 'booktabs',
  caption: '',
  label: '',
  // CSV
  csvDialect: 'standard',
  includeHeaders: true,
  // PDF
  title: '',
  author: '',
  includeCharts: false,
  pageSize: 'a4',
  // BibTeX
  entryType: 'misc',
  keyPrefix: 'weaver',
};

/**
 * ExportForm component for configuring exports.
 */
export const ExportForm: React.FC<ExportFormProps> = ({
  sessionId: initialSessionId,
  onExportComplete,
  onExportError,
  showPreview: initialShowPreview = false,
  className = '',
}) => {
  // Form state
  const [formState, setFormState] = useState<FormState>({
    ...defaultFormState,
    sessionId: initialSessionId || '',
  });

  // UI state
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [isLoadingSessions, setIsLoadingSessions] = useState(true);
  const [isExporting, setIsExporting] = useState(false);
  const [isPreviewing, setIsPreviewing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<ExportResponse | null>(null);
  const [showPreview, setShowPreview] = useState(initialShowPreview);

  // Load sessions on mount
  useEffect(() => {
    const loadSessions = async () => {
      try {
        const result = await listSessions();
        setSessions(result.sessions);
        // Auto-select first session if none specified
        if (!initialSessionId && result.sessions.length > 0) {
          setFormState((prev) => ({
            ...prev,
            sessionId: result.sessions[0].id,
          }));
        }
      } catch {
        // Sessions list may fail if not connected
      } finally {
        setIsLoadingSessions(false);
      }
    };
    loadSessions();
  }, [initialSessionId]);

  /** Build export options from form state */
  const buildExportOptions = useCallback(():
    LatexExportOptions | CsvExportOptions | PdfExportOptions | BibtexExportOptions => {
    const baseOptions = {
      sessionId: formState.sessionId || undefined,
      includeSummary: formState.includeSummary,
      columns: formState.columns.length > 0 ? formState.columns : undefined,
    };

    switch (formState.format) {
      case 'latex':
        return {
          ...baseOptions,
          style: formState.latexStyle,
          caption: formState.caption || undefined,
          label: formState.label || undefined,
        } as LatexExportOptions;

      case 'csv':
        return {
          ...baseOptions,
          dialect: formState.csvDialect,
          includeHeaders: formState.includeHeaders,
        } as CsvExportOptions;

      case 'pdf':
        return {
          ...baseOptions,
          title: formState.title || undefined,
          author: formState.author || undefined,
          includeCharts: formState.includeCharts,
          pageSize: formState.pageSize,
        } as PdfExportOptions;

      case 'bibtex':
        return {
          ...baseOptions,
          entryType: formState.entryType || undefined,
          keyPrefix: formState.keyPrefix || undefined,
        } as BibtexExportOptions;

      default:
        return baseOptions as LatexExportOptions;
    }
  }, [formState]);

  /** Generate preview */
  const handlePreview = useCallback(async () => {
    setIsPreviewing(true);
    setError(null);

    try {
      const options = buildExportOptions();
      const result = await exportMeasurements(formState.format, options);
      setPreview(result);
      setShowPreview(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate preview');
      onExportError?.(err instanceof Error ? err : new Error('Preview failed'));
    } finally {
      setIsPreviewing(false);
    }
  }, [buildExportOptions, formState.format, onExportError]);

  /** Export and download */
  const handleExport = useCallback(async () => {
    setIsExporting(true);
    setError(null);

    try {
      const options = buildExportOptions();
      const result = await exportMeasurements(formState.format, options);
      downloadExport(result);
      setPreview(result);
      onExportComplete?.(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to export');
      onExportError?.(err instanceof Error ? err : new Error('Export failed'));
    } finally {
      setIsExporting(false);
    }
  }, [buildExportOptions, formState.format, onExportComplete, onExportError]);

  /** Update form field */
  const updateField = useCallback(
    <K extends keyof FormState>(field: K, value: FormState[K]) => {
      setFormState((prev) => ({ ...prev, [field]: value }));
      // Clear preview when options change
      setPreview(null);
    },
    []
  );

  /** Toggle column selection */
  const toggleColumn = useCallback((column: string) => {
    setFormState((prev) => {
      const columns = prev.columns.includes(column)
        ? prev.columns.filter((c) => c !== column)
        : [...prev.columns, column];
      return { ...prev, columns };
    });
    setPreview(null);
  }, []);

  /** Get selected session name */
  const selectedSession = sessions.find((s) => s.id === formState.sessionId);

  const isLoading = isExporting || isPreviewing;

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Format Selector */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-3">
          Export Format
        </label>
        <FormatSelector
          value={formState.format}
          onChange={(format) => updateField('format', format)}
          disabled={isLoading}
        />
      </div>

      {/* Session Selector */}
      <div>
        <label
          htmlFor="sessionId"
          className="block text-sm font-medium text-gray-700 mb-1"
        >
          Session
        </label>
        <select
          id="sessionId"
          value={formState.sessionId}
          onChange={(e) => updateField('sessionId', e.target.value)}
          disabled={isLoading || isLoadingSessions}
          className="input w-full"
        >
          <option value="">All sessions</option>
          {sessions.map((session) => (
            <option key={session.id} value={session.id}>
              {session.name} - {new Date(session.startedAt).toLocaleDateString()}
            </option>
          ))}
        </select>
        {isLoadingSessions && (
          <p className="mt-1 text-sm text-gray-500">Loading sessions...</p>
        )}
      </div>

      {/* Column Selection */}
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Columns to Include
        </label>
        <div className="flex flex-wrap gap-2">
          {MEASUREMENT_COLUMNS.map((column) => {
            const isSelected = formState.columns.includes(column);
            return (
              <button
                key={column}
                type="button"
                onClick={() => toggleColumn(column)}
                disabled={isLoading}
                className={`px-3 py-1.5 text-sm rounded-full transition-colors ${
                  isSelected
                    ? 'bg-weaver-100 text-weaver-700 border border-weaver-300'
                    : 'bg-gray-100 text-gray-600 border border-gray-200 hover:bg-gray-200'
                } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {column.replace(/_/g, ' ')}
              </button>
            );
          })}
        </div>
      </div>

      {/* Format-Specific Options */}
      <div className="border-t border-gray-200 pt-6">
        <h3 className="text-sm font-medium text-gray-700 mb-4">
          {getFormatDisplayName(formState.format)} Options
        </h3>

        {/* LaTeX Options */}
        {formState.format === 'latex' && (
          <div className="space-y-4">
            <div>
              <label htmlFor="latexStyle" className="block text-sm text-gray-600 mb-1">
                Table Style
              </label>
              <select
                id="latexStyle"
                value={formState.latexStyle}
                onChange={(e) =>
                  updateField('latexStyle', e.target.value as LatexTableStyle)
                }
                disabled={isLoading}
                className="input w-full"
              >
                <option value="booktabs">Booktabs (recommended)</option>
                <option value="plain">Plain</option>
              </select>
            </div>
            <div>
              <label htmlFor="caption" className="block text-sm text-gray-600 mb-1">
                Caption
              </label>
              <input
                id="caption"
                type="text"
                value={formState.caption}
                onChange={(e) => updateField('caption', e.target.value)}
                placeholder="Table caption (optional)"
                disabled={isLoading}
                className="input w-full"
              />
            </div>
            <div>
              <label htmlFor="label" className="block text-sm text-gray-600 mb-1">
                Label
              </label>
              <input
                id="label"
                type="text"
                value={formState.label}
                onChange={(e) => updateField('label', e.target.value)}
                placeholder="e.g., tab:results (optional)"
                disabled={isLoading}
                className="input w-full"
              />
            </div>
          </div>
        )}

        {/* CSV Options */}
        {formState.format === 'csv' && (
          <div className="space-y-4">
            <div>
              <label htmlFor="csvDialect" className="block text-sm text-gray-600 mb-1">
                Dialect
              </label>
              <select
                id="csvDialect"
                value={formState.csvDialect}
                onChange={(e) =>
                  updateField('csvDialect', e.target.value as CsvDialect)
                }
                disabled={isLoading}
                className="input w-full"
              >
                <option value="standard">Standard CSV</option>
                <option value="excel">Excel Compatible</option>
                <option value="tsv">Tab-Separated (TSV)</option>
              </select>
            </div>
            <div className="flex items-center">
              <input
                id="includeHeaders"
                type="checkbox"
                checked={formState.includeHeaders}
                onChange={(e) => updateField('includeHeaders', e.target.checked)}
                disabled={isLoading}
                className="h-4 w-4 text-weaver-600 focus:ring-weaver-500 border-gray-300 rounded"
              />
              <label htmlFor="includeHeaders" className="ml-2 text-sm text-gray-600">
                Include header row
              </label>
            </div>
          </div>
        )}

        {/* PDF Options */}
        {formState.format === 'pdf' && (
          <div className="space-y-4">
            <div>
              <label htmlFor="title" className="block text-sm text-gray-600 mb-1">
                Document Title
              </label>
              <input
                id="title"
                type="text"
                value={formState.title}
                onChange={(e) => updateField('title', e.target.value)}
                placeholder="e.g., Conveyance Metrics Report"
                disabled={isLoading}
                className="input w-full"
              />
            </div>
            <div>
              <label htmlFor="author" className="block text-sm text-gray-600 mb-1">
                Author
              </label>
              <input
                id="author"
                type="text"
                value={formState.author}
                onChange={(e) => updateField('author', e.target.value)}
                placeholder="Your name (optional)"
                disabled={isLoading}
                className="input w-full"
              />
            </div>
            <div>
              <label htmlFor="pageSize" className="block text-sm text-gray-600 mb-1">
                Page Size
              </label>
              <select
                id="pageSize"
                value={formState.pageSize}
                onChange={(e) =>
                  updateField('pageSize', e.target.value as 'a4' | 'letter')
                }
                disabled={isLoading}
                className="input w-full"
              >
                <option value="a4">A4</option>
                <option value="letter">Letter</option>
              </select>
            </div>
            <div className="flex items-center">
              <input
                id="includeCharts"
                type="checkbox"
                checked={formState.includeCharts}
                onChange={(e) => updateField('includeCharts', e.target.checked)}
                disabled={isLoading}
                className="h-4 w-4 text-weaver-600 focus:ring-weaver-500 border-gray-300 rounded"
              />
              <label htmlFor="includeCharts" className="ml-2 text-sm text-gray-600">
                Include metric charts
              </label>
            </div>
          </div>
        )}

        {/* BibTeX Options */}
        {formState.format === 'bibtex' && (
          <div className="space-y-4">
            <div>
              <label htmlFor="entryType" className="block text-sm text-gray-600 mb-1">
                Entry Type
              </label>
              <select
                id="entryType"
                value={formState.entryType}
                onChange={(e) => updateField('entryType', e.target.value)}
                disabled={isLoading}
                className="input w-full"
              >
                <option value="misc">misc</option>
                <option value="article">article</option>
                <option value="inproceedings">inproceedings</option>
                <option value="techreport">techreport</option>
                <option value="unpublished">unpublished</option>
              </select>
            </div>
            <div>
              <label htmlFor="keyPrefix" className="block text-sm text-gray-600 mb-1">
                Citation Key Prefix
              </label>
              <input
                id="keyPrefix"
                type="text"
                value={formState.keyPrefix}
                onChange={(e) => updateField('keyPrefix', e.target.value)}
                placeholder="e.g., weaver"
                disabled={isLoading}
                className="input w-full"
              />
            </div>
          </div>
        )}
      </div>

      {/* Include Summary Checkbox */}
      <div className="flex items-center">
        <input
          id="includeSummary"
          type="checkbox"
          checked={formState.includeSummary}
          onChange={(e) => updateField('includeSummary', e.target.checked)}
          disabled={isLoading}
          className="h-4 w-4 text-weaver-600 focus:ring-weaver-500 border-gray-300 rounded"
        />
        <label htmlFor="includeSummary" className="ml-2 text-sm text-gray-600">
          Include summary statistics
        </label>
      </div>

      {/* Error Display */}
      {error && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
          <div className="flex items-center gap-2">
            <svg
              className="w-5 h-5 text-red-500 flex-shrink-0"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <p className="text-sm text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Preview Section */}
      {showPreview && preview && (
        <div className="border border-gray-200 rounded-lg overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
            <span className="text-sm font-medium text-gray-700">
              Preview: {preview.filename}
            </span>
            <button
              type="button"
              onClick={() => setShowPreview(false)}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 18L18 6M6 6l12 12"
                />
              </svg>
            </button>
          </div>
          <pre className="p-4 text-sm text-gray-800 bg-gray-900 overflow-auto max-h-64">
            <code className="text-green-400">{preview.content}</code>
          </pre>
          {preview.stats && (
            <div className="px-4 py-2 bg-gray-50 border-t border-gray-200 text-xs text-gray-500">
              {preview.stats.measurementCount} measurements, {preview.stats.rowCount} rows
            </div>
          )}
        </div>
      )}

      {/* Action Buttons */}
      <div className="flex items-center justify-end gap-3 pt-4 border-t border-gray-200">
        <button
          type="button"
          onClick={handlePreview}
          disabled={isLoading}
          className="btn-secondary"
        >
          {isPreviewing ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Generating...
            </span>
          ) : (
            <>
              <svg
                className="w-4 h-4 mr-1.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                />
              </svg>
              Preview
            </>
          )}
        </button>
        <button
          type="button"
          onClick={handleExport}
          disabled={isLoading}
          className="btn-primary"
        >
          {isExporting ? (
            <span className="flex items-center gap-2">
              <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Exporting...
            </span>
          ) : (
            <>
              <svg
                className="w-4 h-4 mr-1.5"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                />
              </svg>
              Export & Download
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default ExportForm;
