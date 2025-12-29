/**
 * Export page - export tools and format selection.
 *
 * Provides interface for exporting session measurements in
 * various publication-ready formats (LaTeX, CSV, PDF, BibTeX).
 */
import { useCallback, useState } from 'react';
import { ExportForm } from '@/components/export';
import type { ExportResponse } from '@/services/exportApi';

/**
 * Export page component.
 */
export const Export: React.FC = () => {
  // Track successful exports for displaying stats
  const [lastExport, setLastExport] = useState<ExportResponse | null>(null);
  const [exportCount, setExportCount] = useState(0);

  /** Handle successful export */
  const handleExportComplete = useCallback((result: ExportResponse) => {
    setLastExport(result);
    setExportCount((prev) => prev + 1);
  }, []);

  /** Handle export error */
  const handleExportError = useCallback((error: Error) => {
    // Error is already displayed by ExportForm
    // Could add analytics tracking here
  }, []);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Export</h1>
        <p className="mt-2 text-gray-600">
          Export session measurements in publication-ready formats
        </p>
      </div>

      {/* Export Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Exports This Session</h3>
          <p className="text-2xl font-bold text-weaver-600">{exportCount}</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Last Export</h3>
          <p className="text-2xl font-bold text-weaver-600">
            {lastExport?.format.toUpperCase() || 'None'}
          </p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Last File</h3>
          <p className="text-lg font-medium text-gray-700 truncate" title={lastExport?.filename}>
            {lastExport?.filename || 'N/A'}
          </p>
        </div>
      </div>

      {/* Export Form */}
      <div className="card">
        <div className="mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Export Options</h2>
          <p className="text-sm text-gray-500">
            Select a format and configure export settings
          </p>
        </div>
        <ExportForm
          onExportComplete={handleExportComplete}
          onExportError={handleExportError}
        />
      </div>

      {/* Info Panel */}
      <div className="card bg-blue-50 border-blue-200">
        <div className="flex items-start space-x-3">
          <svg
            className="w-5 h-5 text-blue-500 mt-0.5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div>
            <h3 className="text-sm font-medium text-blue-800">
              About Export Formats
            </h3>
            <div className="mt-2 text-sm text-blue-700 space-y-2">
              <p>
                <strong>LaTeX:</strong> Creates publication-ready tables using the
                booktabs package. Includes proper escaping and formatting.
              </p>
              <p>
                <strong>CSV:</strong> Standard comma-separated values format compatible
                with Excel, Google Sheets, and data analysis tools.
              </p>
              <p>
                <strong>PDF:</strong> Generates LaTeX source code for a complete
                document that can be compiled to PDF using pdflatex.
              </p>
              <p>
                <strong>BibTeX:</strong> Creates citation entries for referencing
                your experiments in academic papers.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* CLI Reference */}
      <div className="card">
        <h3 className="text-sm font-medium text-gray-900 mb-2">
          CLI Equivalent Commands
        </h3>
        <div className="space-y-2 text-sm text-gray-600">
          <div className="flex items-center gap-2">
            <code className="bg-gray-100 px-2 py-1 rounded font-mono text-xs">
              /export latex &lt;session&gt;
            </code>
            <span>Export to LaTeX table</span>
          </div>
          <div className="flex items-center gap-2">
            <code className="bg-gray-100 px-2 py-1 rounded font-mono text-xs">
              /export csv &lt;session&gt;
            </code>
            <span>Export to CSV file</span>
          </div>
          <div className="flex items-center gap-2">
            <code className="bg-gray-100 px-2 py-1 rounded font-mono text-xs">
              /export pdf &lt;session&gt;
            </code>
            <span>Export to PDF document</span>
          </div>
          <div className="flex items-center gap-2">
            <code className="bg-gray-100 px-2 py-1 rounded font-mono text-xs">
              /export bibtex &lt;session&gt;
            </code>
            <span>Export to BibTeX citation</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Export;
