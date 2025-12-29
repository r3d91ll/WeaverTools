/**
 * ConceptCard component - displays a single concept with statistics.
 *
 * Shows concept information including name, sample count, dimension,
 * models used, and health status.
 */
import type { ConceptStats } from '@/types/concept';
import { formatDimension, formatRelativeTime, getConceptHealth } from '@/types/concept';

/**
 * Concept status type.
 */
export type ConceptStatusType = 'healthy' | 'warning' | 'error';

/**
 * ConceptCard component props.
 */
export interface ConceptCardProps {
  /** Concept statistics to display */
  concept: ConceptStats;
  /** Callback when view details is requested */
  onViewDetails?: (name: string) => void;
  /** Callback when delete is requested */
  onDelete?: (name: string) => void;
  /** Whether the concept is being deleted */
  isDeleting?: boolean;
  /** Whether to use compact view */
  compact?: boolean;
  /** Optional className for the card container */
  className?: string;
}

/**
 * Convert health status to ConceptStatusType.
 */
function getConceptStatus(concept: ConceptStats): ConceptStatusType {
  return getConceptHealth(concept);
}

/**
 * Get status display configuration.
 */
function getStatusConfig(status: ConceptStatusType): {
  color: string;
  bgColor: string;
  textColor: string;
  label: string;
} {
  switch (status) {
    case 'healthy':
      return {
        color: 'bg-green-500',
        bgColor: 'bg-green-100',
        textColor: 'text-green-800',
        label: 'Healthy',
      };
    case 'warning':
      return {
        color: 'bg-yellow-500',
        bgColor: 'bg-yellow-100',
        textColor: 'text-yellow-800',
        label: 'Warning',
      };
    case 'error':
      return {
        color: 'bg-red-500',
        bgColor: 'bg-red-100',
        textColor: 'text-red-800',
        label: 'Issues',
      };
  }
}


/**
 * ConceptCard component for displaying concept information.
 */
export const ConceptCard: React.FC<ConceptCardProps> = ({
  concept,
  onViewDetails,
  onDelete,
  isDeleting = false,
  compact = false,
  className = '',
}) => {
  const { name, sampleCount, dimension, models = [], mismatchedIds = [] } = concept;
  const status = getConceptStatus(concept);
  const statusConfig = getStatusConfig(status);

  const handleViewDetails = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onViewDetails?.(name);
  };

  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDelete?.(name);
  };

  return (
    <div
      className={`card ${isDeleting ? 'opacity-75' : ''} transition-all ${className}`}
    >
      {/* Header with name and status */}
      <div className={`flex items-start justify-between ${compact ? 'mb-2' : 'mb-3'}`}>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-gray-900 truncate">{name}</h3>
            <span className="text-xs font-mono bg-gray-100 text-gray-600 px-1.5 py-0.5 rounded">
              {sampleCount} {sampleCount === 1 ? 'sample' : 'samples'}
            </span>
          </div>
          {concept.updatedAt && !compact && (
            <p className="text-sm text-gray-500 mt-0.5">
              Updated {formatRelativeTime(concept.updatedAt)}
            </p>
          )}
        </div>
        <div className="flex-shrink-0 ml-4">
          <span
            className={`inline-flex items-center rounded-full font-medium px-2.5 py-0.5 text-xs ${statusConfig.bgColor} ${statusConfig.textColor}`}
          >
            <span
              className={`rounded-full w-2 h-2 ${statusConfig.color} mr-1.5`}
            />
            {statusConfig.label}
          </span>
        </div>
      </div>

      {/* Concept info row */}
      {!compact && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-3">
          {/* Dimension */}
          <div>
            <p className="text-xs text-gray-500">Dimension</p>
            <p className="text-sm font-medium text-gray-900">
              {formatDimension(dimension)}
            </p>
          </div>

          {/* Models */}
          <div>
            <p className="text-xs text-gray-500">Models</p>
            {models.length > 0 ? (
              <div className="flex flex-wrap gap-1 mt-1">
                {models.slice(0, 2).map((model) => (
                  <span
                    key={model}
                    className="text-xs font-medium px-1.5 py-0.5 rounded bg-purple-100 text-purple-800"
                  >
                    {model}
                  </span>
                ))}
                {models.length > 2 && (
                  <span className="text-xs text-gray-500">
                    +{models.length - 2}
                  </span>
                )}
              </div>
            ) : (
              <p className="text-sm font-medium text-gray-400">None</p>
            )}
          </div>

          {/* Sample Count */}
          <div>
            <p className="text-xs text-gray-500">Samples</p>
            <p className="text-sm font-medium text-weaver-600">{sampleCount}</p>
          </div>

          {/* Mismatched */}
          <div>
            <p className="text-xs text-gray-500">Mismatched</p>
            <p
              className={`text-sm font-medium ${
                mismatchedIds.length > 0 ? 'text-red-600' : 'text-gray-400'
              }`}
            >
              {mismatchedIds.length > 0 ? mismatchedIds.length : '---'}
            </p>
          </div>
        </div>
      )}

      {/* Compact info row */}
      {compact && (
        <div className="flex items-center gap-4 text-sm text-gray-500 mb-3">
          <span>Dim: {formatDimension(dimension)}</span>
          {models.length > 0 && (
            <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-purple-100 text-purple-800">
              {models[0]}
            </span>
          )}
          {mismatchedIds.length > 0 && (
            <span className="text-red-600">
              {mismatchedIds.length} mismatched
            </span>
          )}
        </div>
      )}

      {/* Action buttons */}
      <div className="flex items-center justify-between pt-3 border-t border-gray-100">
        <div className="flex items-center gap-2 text-xs text-gray-500">
          {concept.createdAt && (
            <span>Created {formatRelativeTime(concept.createdAt)}</span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {onViewDetails && (
            <button
              type="button"
              onClick={handleViewDetails}
              disabled={isDeleting}
              className="btn-secondary text-sm py-1.5 px-3 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              View Details
            </button>
          )}
          {onDelete && (
            <button
              type="button"
              onClick={handleDelete}
              disabled={isDeleting}
              className="text-sm py-1.5 px-3 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isDeleting ? (
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
                  Deleting...
                </span>
              ) : (
                'Delete'
              )}
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default ConceptCard;
