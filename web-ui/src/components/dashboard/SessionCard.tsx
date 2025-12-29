/**
 * SessionCard component - displays a single session with status and stats.
 *
 * Used in the dashboard session list to show session overview with
 * quick actions like view, delete, and end session.
 */
import { Link } from 'react-router';
import type { SessionSummary } from '@/services/sessionApi';
import { isSessionActive, getSessionDuration, formatDuration } from '@/services/sessionApi';

/**
 * SessionCard component props.
 */
export interface SessionCardProps {
  /** Session data to display */
  session: SessionSummary;
  /** Callback when delete is requested */
  onDelete?: (id: string) => void;
  /** Callback when end session is requested */
  onEnd?: (id: string) => void;
  /** Whether actions are disabled */
  disabled?: boolean;
  /** Whether the card is in a loading state */
  loading?: boolean;
  /** Whether to use compact view (less details) */
  compact?: boolean;
}

/**
 * Format a date for display.
 */
function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format a relative time (e.g., "2 hours ago").
 */
function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffDays > 0) {
    return diffDays === 1 ? '1 day ago' : `${diffDays} days ago`;
  }
  if (diffHours > 0) {
    return diffHours === 1 ? '1 hour ago' : `${diffHours} hours ago`;
  }
  if (diffMinutes > 0) {
    return diffMinutes === 1 ? '1 minute ago' : `${diffMinutes} minutes ago`;
  }
  return 'Just now';
}

/**
 * Session card component displaying session overview.
 */
export const SessionCard: React.FC<SessionCardProps> = ({
  session,
  onDelete,
  onEnd,
  disabled = false,
  loading = false,
  compact = false,
}) => {
  const isActive = isSessionActive(session);
  const duration = getSessionDuration(session);
  const isDisabled = disabled || loading;

  const handleDelete = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onDelete?.(session.id);
  };

  const handleEnd = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    onEnd?.(session.id);
  };

  return (
    <div className="relative">
      <Link
        to={`/sessions/${session.id}`}
        className={`card hover:shadow-md transition-shadow block ${loading ? 'opacity-60' : ''}`}
      >
        {/* Header with status */}
        <div className={`flex items-start justify-between ${compact ? 'mb-2' : 'mb-3'}`}>
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-gray-900 truncate">
            {session.name}
          </h3>
          {session.description && (
            <p className="text-sm text-gray-500 truncate mt-1">
              {session.description}
            </p>
          )}
        </div>
        <div className="flex-shrink-0 ml-4">
          <span
            className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
              isActive
                ? 'bg-green-100 text-green-800'
                : 'bg-gray-100 text-gray-600'
            }`}
          >
            {isActive ? (
              <>
                <span className="w-1.5 h-1.5 bg-green-500 rounded-full mr-1.5 animate-pulse" />
                Active
              </>
            ) : (
              'Completed'
            )}
          </span>
        </div>
      </div>

      {/* Stats row - hidden in compact mode */}
      {!compact && (
      <div className="grid grid-cols-3 gap-4 mb-3">
        <div>
          <p className="text-xs text-gray-500">Messages</p>
          <p className="text-sm font-medium text-gray-900">
            {session.stats?.messageCount ?? 0}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Measurements</p>
          <p className="text-sm font-medium text-gray-900">
            {session.stats?.measurementCount ?? 0}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500">Conversations</p>
          <p className="text-sm font-medium text-gray-900">
            {session.stats?.conversationCount ?? 0}
          </p>
        </div>
      </div>
      )}

      {/* Metrics row (if available) - hidden in compact mode */}
      {!compact && session.stats && (session.stats.measurementCount ?? 0) > 0 && (
        <div className="grid grid-cols-3 gap-4 mb-3 pb-3 border-b border-gray-100">
          <div>
            <p className="text-xs text-gray-500">Avg D_eff</p>
            <p className="text-sm font-medium text-weaver-600">
              {session.stats.avgDEff?.toFixed(3) ?? 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Avg Beta</p>
            <p className="text-sm font-medium text-weaver-600">
              {session.stats.avgBeta?.toFixed(3) ?? 'N/A'}
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-500">Avg Alignment</p>
            <p className="text-sm font-medium text-weaver-600">
              {session.stats.avgAlignment?.toFixed(3) ?? 'N/A'}
            </p>
          </div>
        </div>
      )}

      {/* Footer with date and actions */}
      <div className="flex items-center justify-between text-xs text-gray-500">
        <div className="flex items-center gap-3">
          <span title={formatDate(session.startedAt)}>
            Started {formatRelativeTime(session.startedAt)}
          </span>
          {duration !== null && (
            <span className="text-gray-400">
              Duration: {formatDuration(duration)}
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {isActive && onEnd && (
            <button
              type="button"
              onClick={handleEnd}
              disabled={isDisabled}
              className="text-yellow-600 hover:text-yellow-700 disabled:opacity-50"
              title="End session"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
                />
              </svg>
            </button>
          )}
          {onDelete && (
            <button
              type="button"
              onClick={handleDelete}
              disabled={isDisabled}
              className="text-red-600 hover:text-red-700 disabled:opacity-50"
              title="Delete session"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                />
              </svg>
            </button>
          )}
        </div>
      </div>
      </Link>

      {/* Loading overlay */}
      {loading && (
        <div className="absolute inset-0 bg-white/75 rounded-lg flex items-center justify-center">
          <svg
            className="animate-spin h-6 w-6 text-weaver-600"
            viewBox="0 0 24 24"
          >
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
        </div>
      )}
    </div>
  );
};

export default SessionCard;
