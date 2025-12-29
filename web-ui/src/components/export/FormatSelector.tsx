/**
 * FormatSelector component for choosing export formats.
 *
 * Displays available export formats (LaTeX, CSV, PDF, BibTeX) with
 * icons and descriptions for user selection.
 */
import type { ExportFormat } from '@/services/exportApi';

/**
 * Format information for display.
 */
interface FormatInfo {
  id: ExportFormat;
  name: string;
  description: string;
  icon: React.ReactNode;
  extension: string;
}

/**
 * Available export formats with display information.
 */
const FORMATS: FormatInfo[] = [
  {
    id: 'latex',
    name: 'LaTeX',
    description: 'Publication-ready tables with booktabs styling',
    extension: '.tex',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
        />
      </svg>
    ),
  },
  {
    id: 'csv',
    name: 'CSV',
    description: 'Spreadsheet-compatible comma-separated values',
    extension: '.csv',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M3 10h18M3 14h18m-9-4v8m-7 0h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z"
        />
      </svg>
    ),
  },
  {
    id: 'pdf',
    name: 'PDF',
    description: 'LaTeX source for PDF compilation',
    extension: '.tex',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M7 21h10a2 2 0 002-2V9.414a1 1 0 00-.293-.707l-5.414-5.414A1 1 0 0012.586 3H7a2 2 0 00-2 2v14a2 2 0 002 2z"
        />
      </svg>
    ),
  },
  {
    id: 'bibtex',
    name: 'BibTeX',
    description: 'Citation entries for bibliography',
    extension: '.bib',
    icon: (
      <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
        />
      </svg>
    ),
  },
];

/**
 * FormatSelector component props.
 */
export interface FormatSelectorProps {
  /** Currently selected format */
  value: ExportFormat;
  /** Callback when format is changed */
  onChange: (format: ExportFormat) => void;
  /** Whether the selector is disabled */
  disabled?: boolean;
  /** Additional CSS classes */
  className?: string;
}

/**
 * FormatSelector component for choosing export formats.
 */
export const FormatSelector: React.FC<FormatSelectorProps> = ({
  value,
  onChange,
  disabled = false,
  className = '',
}) => {
  return (
    <div className={`grid grid-cols-2 md:grid-cols-4 gap-3 ${className}`}>
      {FORMATS.map((format) => {
        const isSelected = value === format.id;
        return (
          <button
            key={format.id}
            type="button"
            onClick={() => onChange(format.id)}
            disabled={disabled}
            className={`relative flex flex-col items-center p-4 rounded-lg border-2 transition-all ${
              isSelected
                ? 'border-weaver-500 bg-weaver-50 text-weaver-700'
                : 'border-gray-200 bg-white text-gray-600 hover:border-gray-300 hover:bg-gray-50'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            {/* Selected indicator */}
            {isSelected && (
              <span className="absolute top-2 right-2">
                <svg
                  className="w-5 h-5 text-weaver-500"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                    clipRule="evenodd"
                  />
                </svg>
              </span>
            )}

            {/* Icon */}
            <span
              className={`mb-2 ${isSelected ? 'text-weaver-600' : 'text-gray-400'}`}
            >
              {format.icon}
            </span>

            {/* Name */}
            <span className="font-medium text-sm">{format.name}</span>

            {/* Extension */}
            <span className="text-xs text-gray-400 mt-0.5">{format.extension}</span>

            {/* Description - hidden on small screens */}
            <span className="hidden md:block text-xs text-center text-gray-500 mt-2 line-clamp-2">
              {format.description}
            </span>
          </button>
        );
      })}
    </div>
  );
};

/**
 * Get format info by ID.
 */
export function getFormatInfo(format: ExportFormat): FormatInfo | undefined {
  return FORMATS.find((f) => f.id === format);
}

/**
 * Get all available formats.
 */
export function getAvailableFormats(): FormatInfo[] {
  return [...FORMATS];
}

export default FormatSelector;
