/**
 * CSVPreview component - tabular CSV preview.
 *
 * Displays CSV export content as both a formatted table
 * and raw text view with syntax highlighting.
 */
import { useMemo, useCallback, useState } from 'react';
import type { CsvDialect } from '@/services/exportApi';

/**
 * CSVPreview component props.
 */
export interface CSVPreviewProps {
  /** CSV content to display */
  content: string;
  /** Filename for display */
  filename?: string;
  /** CSV dialect used */
  dialect?: CsvDialect;
  /** Maximum height for table view */
  maxHeight?: number | string;
  /** Callback when copy button is clicked */
  onCopy?: () => void;
  /** Whether copy was successful (for feedback) */
  copySuccess?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * View mode for CSV display.
 */
type ViewMode = 'table' | 'raw';

/**
 * Parse CSV content into rows and columns.
 */
function parseCSV(content: string, dialect: CsvDialect = 'standard'): string[][] {
  const delimiter = dialect === 'tsv' ? '\t' : ',';
  const lines = content.trim().split('\n');

  return lines.map((line) => {
    const cells: string[] = [];
    let currentCell = '';
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
      const char = line[i];
      const nextChar = line[i + 1];

      if (inQuotes) {
        if (char === '"') {
          if (nextChar === '"') {
            // Escaped quote
            currentCell += '"';
            i++;
          } else {
            // End of quoted string
            inQuotes = false;
          }
        } else {
          currentCell += char;
        }
      } else {
        if (char === '"') {
          inQuotes = true;
        } else if (char === delimiter) {
          cells.push(currentCell.trim());
          currentCell = '';
        } else {
          currentCell += char;
        }
      }
    }
    cells.push(currentCell.trim());
    return cells;
  });
}

/**
 * Get column alignment based on content type.
 */
function getColumnAlignment(values: string[]): 'left' | 'right' | 'center' {
  // Check if mostly numbers
  const numericCount = values.filter((v) => !isNaN(parseFloat(v)) && v.trim() !== '').length;
  const ratio = numericCount / values.length;

  if (ratio > 0.5) return 'right';
  return 'left';
}

/**
 * Format cell value for display.
 */
function formatCellValue(value: string): { formatted: string; isNumber: boolean } {
  const trimmed = value.trim();

  // Check if it's a number
  const num = parseFloat(trimmed);
  if (!isNaN(num) && trimmed !== '') {
    // Format with appropriate precision
    if (Number.isInteger(num)) {
      return { formatted: num.toString(), isNumber: true };
    }
    // For decimals, show up to 4 significant digits
    return { formatted: num.toPrecision(4), isNumber: true };
  }

  return { formatted: trimmed || '-', isNumber: false };
}

/**
 * Highlight raw CSV content.
 */
function highlightCSV(content: string, dialect: CsvDialect = 'standard'): string {
  const delimiter = dialect === 'tsv' ? '\t' : ',';
  const delimiterRegex = new RegExp(`(${delimiter === '\t' ? '\\t' : delimiter})`, 'g');

  const lines = content.split('\n');
  const highlighted = lines.map((line, index) => {
    // Escape HTML
    let escaped = line
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Header row (first line)
    if (index === 0) {
      // Highlight delimiters and make headers bold
      escaped = escaped
        .replace(delimiterRegex, '<span class="csv-delimiter">$1</span>')
        .replace(/^(.*)$/, '<span class="csv-header">$1</span>');
      return escaped;
    }

    // Data rows
    escaped = escaped
      // Quoted strings
      .replace(/"([^"]+)"/g, '<span class="csv-string">"$1"</span>')
      // Delimiters
      .replace(delimiterRegex, '<span class="csv-delimiter">$1</span>')
      // Numbers
      .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="csv-number">$1</span>');

    return escaped;
  });

  return highlighted.join('\n');
}

/**
 * CSVPreview component for displaying CSV content as table or raw text.
 */
export const CSVPreview: React.FC<CSVPreviewProps> = ({
  content,
  filename,
  dialect = 'standard',
  maxHeight = 400,
  onCopy,
  copySuccess,
  className = '',
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('table');
  const [copied, setCopied] = useState(false);

  // Parse CSV into rows
  const rows = useMemo(() => parseCSV(content, dialect), [content, dialect]);

  // Get headers and data rows
  const headers = rows[0] || [];
  const dataRows = rows.slice(1);

  // Calculate column alignments
  const columnAlignments = useMemo(() => {
    return headers.map((_, colIndex) => {
      const columnValues = dataRows.map((row) => row[colIndex] || '');
      return getColumnAlignment(columnValues);
    });
  }, [headers, dataRows]);

  // Get highlighted raw content
  const highlightedHtml = useMemo(
    () => highlightCSV(content, dialect),
    [content, dialect]
  );

  // Handle copy to clipboard
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      onCopy?.();
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = content;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      onCopy?.();
      setTimeout(() => setCopied(false), 2000);
    }
  }, [content, onCopy]);

  const isCopied = copySuccess ?? copied;

  // Get dialect display name
  const dialectName = dialect === 'tsv' ? 'TSV' : dialect === 'excel' ? 'Excel CSV' : 'CSV';
  const extension = dialect === 'tsv' ? '.tsv' : '.csv';

  return (
    <div className={`border border-gray-200 rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-100 border-b border-gray-200">
        <div className="flex items-center gap-2">
          {/* CSV icon */}
          <svg
            className="w-4 h-4 text-gray-500"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
            />
          </svg>
          <span className="text-sm font-medium text-gray-700">
            {filename || `${dialectName} Preview`}
          </span>
          <span className="text-xs text-gray-500">{extension}</span>
        </div>

        <div className="flex items-center gap-2">
          {/* View Mode Toggle */}
          <div className="flex rounded-md shadow-sm">
            <button
              type="button"
              onClick={() => setViewMode('table')}
              className={`px-2 py-1 text-xs font-medium rounded-l-md border ${
                viewMode === 'table'
                  ? 'bg-weaver-100 text-weaver-700 border-weaver-300'
                  : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'
              }`}
            >
              Table
            </button>
            <button
              type="button"
              onClick={() => setViewMode('raw')}
              className={`px-2 py-1 text-xs font-medium rounded-r-md border-t border-b border-r ${
                viewMode === 'raw'
                  ? 'bg-weaver-100 text-weaver-700 border-weaver-300'
                  : 'bg-white text-gray-600 border-gray-300 hover:bg-gray-50'
              }`}
            >
              Raw
            </button>
          </div>

          {/* Copy Button */}
          <button
            type="button"
            onClick={handleCopy}
            className="flex items-center gap-1 px-2 py-1 text-xs text-gray-600 hover:text-gray-800 bg-white hover:bg-gray-50 border border-gray-300 rounded transition-colors"
          >
            {isCopied ? (
              <>
                <svg className="w-3.5 h-3.5 text-green-500" fill="currentColor" viewBox="0 0 20 20">
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
                <span className="text-green-600">Copied!</span>
              </>
            ) : (
              <>
                <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                  />
                </svg>
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Content */}
      {viewMode === 'table' ? (
        /* Table View */
        <div
          className="overflow-auto bg-white"
          style={{ maxHeight: typeof maxHeight === 'number' ? `${maxHeight}px` : maxHeight }}
        >
          <table className="w-full text-sm border-collapse">
            <thead className="sticky top-0 bg-gray-50 z-10">
              <tr>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-200 bg-gray-100">
                  #
                </th>
                {headers.map((header, index) => (
                  <th
                    key={index}
                    className={`px-3 py-2 text-xs font-medium text-gray-700 uppercase tracking-wider border-b border-gray-200 bg-gray-50 ${
                      columnAlignments[index] === 'right' ? 'text-right' : 'text-left'
                    }`}
                  >
                    {header}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {dataRows.length === 0 ? (
                <tr>
                  <td
                    colSpan={headers.length + 1}
                    className="px-3 py-8 text-center text-gray-500"
                  >
                    No data rows
                  </td>
                </tr>
              ) : (
                dataRows.map((row, rowIndex) => (
                  <tr
                    key={rowIndex}
                    className={rowIndex % 2 === 0 ? 'bg-white' : 'bg-gray-50'}
                  >
                    <td className="px-3 py-2 text-xs text-gray-400 font-mono">
                      {rowIndex + 1}
                    </td>
                    {headers.map((_, colIndex) => {
                      const { formatted, isNumber } = formatCellValue(row[colIndex] || '');
                      return (
                        <td
                          key={colIndex}
                          className={`px-3 py-2 text-gray-800 ${
                            columnAlignments[colIndex] === 'right' ? 'text-right' : 'text-left'
                          } ${isNumber ? 'font-mono' : ''}`}
                        >
                          {formatted}
                        </td>
                      );
                    })}
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      ) : (
        /* Raw View */
        <div
          className="overflow-auto bg-gray-900"
          style={{ maxHeight: typeof maxHeight === 'number' ? `${maxHeight}px` : maxHeight }}
        >
          <pre className="text-xs font-mono py-3 px-4 text-gray-300 whitespace-pre overflow-x-auto m-0 leading-5">
            <code dangerouslySetInnerHTML={{ __html: highlightedHtml }} />
          </pre>
        </div>
      )}

      {/* Footer with stats */}
      <div className="px-4 py-2 bg-gray-100 border-t border-gray-200 text-xs text-gray-500">
        <div className="flex items-center justify-between">
          <span>
            {dataRows.length} row{dataRows.length !== 1 ? 's' : ''}, {headers.length} column{headers.length !== 1 ? 's' : ''}
          </span>
          <span>{dialectName} format</span>
        </div>
      </div>

      {/* Syntax Highlighting Styles for Raw View */}
      <style>{`
        .csv-header {
          color: #7dd3fc;
          font-weight: 500;
        }
        .csv-delimiter {
          color: #f97316;
        }
        .csv-string {
          color: #86efac;
        }
        .csv-number {
          color: #fde68a;
        }
      `}</style>
    </div>
  );
};

export default CSVPreview;
