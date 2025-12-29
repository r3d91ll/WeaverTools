/**
 * LaTeXPreview component - syntax-highlighted LaTeX preview.
 *
 * Displays LaTeX export content with syntax highlighting for
 * commands, environments, comments, and special characters.
 */
import { useMemo, useCallback, useState } from 'react';

/**
 * LaTeXPreview component props.
 */
export interface LaTeXPreviewProps {
  /** LaTeX content to display */
  content: string;
  /** Filename for display */
  filename?: string;
  /** Whether to show line numbers */
  showLineNumbers?: boolean;
  /** Maximum height for preview */
  maxHeight?: number | string;
  /** Callback when copy button is clicked */
  onCopy?: () => void;
  /** Whether copy was successful (for feedback) */
  copySuccess?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Basic LaTeX syntax highlighting.
 * Returns HTML with spans for different token types.
 */
function highlightLatex(content: string): string {
  const lines = content.split('\n');
  const highlighted = lines.map((line) => {
    // Escape HTML special characters (except for already processed parts)
    let escaped = line
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Comment lines (starting with %)
    if (/^\s*%/.test(escaped)) {
      return `<span class="latex-comment">${escaped}</span>`;
    }

    // Process different LaTeX elements
    escaped = escaped
      // Commands (backslash followed by letters)
      .replace(/\\([a-zA-Z]+)(\*?)/g, '<span class="latex-command">\\$1$2</span>')
      // Environment begin/end
      .replace(
        /\\(begin|end)\{([^}]+)\}/g,
        '<span class="latex-env">\\$1{<span class="latex-env-name">$2</span>}</span>'
      )
      // Braces grouping
      .replace(/\{([^{}]*)\}/g, '{<span class="latex-group">$1</span>}')
      // Math delimiters
      .replace(/\$([^$]+)\$/g, '<span class="latex-math">$$1$</span>')
      // Comments (% followed by content)
      .replace(/%(.*)$/g, '<span class="latex-comment">%$1</span>')
      // Ampersand (table column separator)
      .replace(/&amp;/g, '<span class="latex-amp">&amp;</span>')
      // Table rule commands (already highlighted as commands, add emphasis)
      .replace(
        /<span class="latex-command">\\(toprule|midrule|bottomrule|hline|cline)<\/span>/g,
        '<span class="latex-rule">\\$1</span>'
      )
      // Numbers
      .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="latex-number">$1</span>');

    return escaped;
  });

  return highlighted.join('\n');
}

/**
 * LaTeXPreview component for displaying syntax-highlighted LaTeX content.
 */
export const LaTeXPreview: React.FC<LaTeXPreviewProps> = ({
  content,
  filename,
  showLineNumbers = true,
  maxHeight = 400,
  onCopy,
  copySuccess,
  className = '',
}) => {
  const [copied, setCopied] = useState(false);

  // Get highlighted HTML
  const highlightedHtml = useMemo(() => highlightLatex(content), [content]);

  // Generate line numbers
  const lineCount = content.split('\n').length;
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1);

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

  return (
    <div className={`border border-gray-200 rounded-lg overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center gap-2">
          {/* LaTeX icon */}
          <svg
            className="w-4 h-4 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <span className="text-sm font-medium text-gray-300">
            {filename || 'LaTeX Preview'}
          </span>
          <span className="text-xs text-gray-500">.tex</span>
        </div>
        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1 px-2 py-1 text-xs text-gray-300 hover:text-white bg-gray-700 hover:bg-gray-600 rounded transition-colors"
        >
          {isCopied ? (
            <>
              <svg className="w-3.5 h-3.5 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                <path
                  fillRule="evenodd"
                  d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                  clipRule="evenodd"
                />
              </svg>
              <span className="text-green-400">Copied!</span>
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

      {/* Content */}
      <div
        className="bg-gray-900 overflow-auto"
        style={{ maxHeight: typeof maxHeight === 'number' ? `${maxHeight}px` : maxHeight }}
      >
        <div className="flex">
          {/* Line Numbers */}
          {showLineNumbers && (
            <div className="flex-shrink-0 bg-gray-800 text-gray-500 text-xs font-mono py-3 px-2 select-none border-r border-gray-700">
              {lineNumbers.map((num) => (
                <div key={num} className="text-right pr-2 leading-5">
                  {num}
                </div>
              ))}
            </div>
          )}

          {/* Code Content */}
          <pre className="flex-1 text-xs font-mono py-3 px-4 text-gray-300 whitespace-pre overflow-x-auto m-0 leading-5">
            <code dangerouslySetInnerHTML={{ __html: highlightedHtml }} />
          </pre>
        </div>
      </div>

      {/* Footer with stats */}
      <div className="px-4 py-2 bg-gray-800 border-t border-gray-700 text-xs text-gray-500">
        <div className="flex items-center justify-between">
          <span>{lineCount} lines</span>
          <span>{content.length} characters</span>
        </div>
      </div>

      {/* Syntax Highlighting Styles */}
      <style>{`
        .latex-command {
          color: #7dd3fc;
        }
        .latex-env {
          color: #c4b5fd;
        }
        .latex-env-name {
          color: #f9a8d4;
          font-weight: 500;
        }
        .latex-group {
          color: #86efac;
        }
        .latex-math {
          color: #fde68a;
        }
        .latex-comment {
          color: #6b7280;
          font-style: italic;
        }
        .latex-amp {
          color: #f97316;
          font-weight: bold;
        }
        .latex-rule {
          color: #22d3ee;
          font-weight: 500;
        }
        .latex-number {
          color: #fde68a;
        }
      `}</style>
    </div>
  );
};

export default LaTeXPreview;
