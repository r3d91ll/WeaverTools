/**
 * ExportPreview component - unified export preview container.
 *
 * Dispatches to format-specific preview components based on the
 * export format (LaTeX, CSV, PDF, BibTeX).
 */
import { useState, useCallback } from 'react';
import { LaTeXPreview } from './LaTeXPreview';
import { CSVPreview } from './CSVPreview';
import type { ExportFormat, ExportResponse } from '@/services/exportApi';

/**
 * ExportPreview component props.
 */
export interface ExportPreviewProps {
  /** Export response containing content and metadata */
  exportResult: ExportResponse;
  /** Whether the preview is in loading state */
  isLoading?: boolean;
  /** Callback to close the preview */
  onClose?: () => void;
  /** Maximum height for the preview content */
  maxHeight?: number | string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Props for the generic text preview component.
 */
interface TextPreviewProps {
  content: string;
  maxHeight?: number | string;
}

/**
 * Generic text preview for BibTeX and other text formats.
 */
const TextPreview: React.FC<TextPreviewProps> = ({
  content,
  maxHeight = 400,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
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
      setTimeout(() => setCopied(false), 2000);
    }
  }, [content]);

  const lineCount = content.split('\n').length;
  const maxHeightStyle = typeof maxHeight === 'number' ? `${maxHeight}px` : maxHeight;

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <span className="text-sm font-medium text-gray-700">Preview</span>
        </div>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
          title="Copy to clipboard"
        >
          {copied ? (
            <>
              <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-green-600">Copied!</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Content */}
      <div
        className="bg-gray-900 overflow-auto"
        style={{ maxHeight: maxHeightStyle }}
      >
        <pre className="text-sm font-mono p-4 whitespace-pre overflow-x-auto text-gray-100 leading-6">
          <code>{content}</code>
        </pre>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-t border-gray-200 text-xs text-gray-500">
        <span>{lineCount} lines</span>
        <span>{content.length} characters</span>
      </div>
    </div>
  );
};

/**
 * BibTeX-specific preview with entry highlighting.
 */
const BibTeXPreview: React.FC<TextPreviewProps> = ({
  content,
  maxHeight = 400,
}) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback
      const textarea = document.createElement('textarea');
      textarea.value = content;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [content]);

  // Highlight BibTeX syntax
  const highlightBibtex = (text: string): string => {
    let result = text
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Entry type and key (e.g., @misc{key,)
    result = result.replace(
      /(@\w+)(\{)([^,]+)(,)/g,
      '<span class="bibtex-type">$1</span>$2<span class="bibtex-key">$3</span>$4'
    );

    // Field names
    result = result.replace(
      /^\s*(\w+)\s*(=)/gm,
      '  <span class="bibtex-field">$1</span> $2'
    );

    // Braced values
    result = result.replace(
      /(\{)([^}]*)(\})/g,
      '<span class="bibtex-brace">$1</span><span class="bibtex-value">$2</span><span class="bibtex-brace">$3</span>'
    );

    // Comments
    result = result.replace(
      /(%[^\n]*)/g,
      '<span class="bibtex-comment">$1</span>'
    );

    return result;
  };

  const highlightedContent = highlightBibtex(content);
  const lineCount = content.split('\n').length;
  const maxHeightStyle = typeof maxHeight === 'number' ? `${maxHeight}px` : maxHeight;

  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
        <div className="flex items-center gap-2">
          <svg className="w-4 h-4 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
          </svg>
          <span className="text-sm font-medium text-gray-700">BibTeX Preview</span>
        </div>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1.5 px-2 py-1 text-sm text-gray-600 hover:text-gray-900 hover:bg-gray-100 rounded transition-colors"
          title="Copy to clipboard"
        >
          {copied ? (
            <>
              <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
              </svg>
              <span className="text-green-600">Copied!</span>
            </>
          ) : (
            <>
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <span>Copy</span>
            </>
          )}
        </button>
      </div>

      {/* Content */}
      <div
        className="bg-gray-900 overflow-auto"
        style={{ maxHeight: maxHeightStyle }}
      >
        <pre className="text-sm font-mono p-4 whitespace-pre overflow-x-auto text-gray-100 leading-6">
          <code dangerouslySetInnerHTML={{ __html: highlightedContent }} />
        </pre>
      </div>

      {/* Footer */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-t border-gray-200 text-xs text-gray-500">
        <span>{lineCount} lines</span>
        <span>{content.length} characters</span>
      </div>

      {/* BibTeX Highlighting Styles */}
      <style>{`
        .bibtex-type {
          color: #f472b6;
          font-weight: bold;
        }
        .bibtex-key {
          color: #fbbf24;
        }
        .bibtex-field {
          color: #7dd3fc;
        }
        .bibtex-value {
          color: #86efac;
        }
        .bibtex-brace {
          color: #9ca3af;
        }
        .bibtex-comment {
          color: #6b7280;
          font-style: italic;
        }
      `}</style>
    </div>
  );
};

/**
 * Loading skeleton for preview.
 */
const PreviewSkeleton: React.FC = () => (
  <div className="border border-gray-200 rounded-lg overflow-hidden animate-pulse">
    <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-b border-gray-200">
      <div className="h-4 w-32 bg-gray-200 rounded" />
      <div className="h-6 w-16 bg-gray-200 rounded" />
    </div>
    <div className="bg-gray-900 p-4 space-y-2" style={{ height: 200 }}>
      <div className="h-4 w-3/4 bg-gray-700 rounded" />
      <div className="h-4 w-1/2 bg-gray-700 rounded" />
      <div className="h-4 w-2/3 bg-gray-700 rounded" />
      <div className="h-4 w-3/5 bg-gray-700 rounded" />
    </div>
    <div className="flex items-center justify-between px-4 py-2 bg-gray-50 border-t border-gray-200">
      <div className="h-3 w-16 bg-gray-200 rounded" />
      <div className="h-3 w-20 bg-gray-200 rounded" />
    </div>
  </div>
);

/**
 * Get format-specific icon.
 */
function getFormatIcon(format: ExportFormat): React.ReactNode {
  switch (format) {
    case 'latex':
    case 'pdf':
      return (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
      );
    case 'csv':
      return (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
        </svg>
      );
    case 'bibtex':
      return (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      );
    default:
      return (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z" />
        </svg>
      );
  }
}

/**
 * ExportPreview component - dispatches to format-specific previews.
 */
export const ExportPreview: React.FC<ExportPreviewProps> = ({
  exportResult,
  isLoading = false,
  onClose,
  maxHeight = 400,
  className = '',
}) => {
  if (isLoading) {
    return (
      <div className={className}>
        <PreviewSkeleton />
      </div>
    );
  }

  const { content, format, filename, stats } = exportResult;

  return (
    <div className={`space-y-4 ${className}`}>
      {/* Preview Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <span className="text-gray-500">{getFormatIcon(format)}</span>
          <div>
            <h3 className="text-sm font-medium text-gray-900">{filename}</h3>
            {stats && (
              <p className="text-xs text-gray-500">
                {stats.measurementCount} measurements, {stats.rowCount} rows
              </p>
            )}
          </div>
        </div>
        {onClose && (
          <button
            type="button"
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 transition-colors"
            title="Close preview"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* Format-specific Preview */}
      {format === 'latex' || format === 'pdf' ? (
        <LaTeXPreview content={content} maxHeight={maxHeight} />
      ) : format === 'csv' ? (
        <CSVPreview content={content} maxHeight={maxHeight} />
      ) : format === 'bibtex' ? (
        <BibTeXPreview content={content} maxHeight={maxHeight} />
      ) : (
        <TextPreview content={content} maxHeight={maxHeight} />
      )}

      {/* Generation Info */}
      {stats?.generatedAt && (
        <div className="text-xs text-gray-400 text-right">
          Generated at {new Date(stats.generatedAt).toLocaleString()}
        </div>
      )}
    </div>
  );
};

export default ExportPreview;
