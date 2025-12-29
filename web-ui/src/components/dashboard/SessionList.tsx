/**
 * SessionList component for displaying and managing sessions.
 *
 * Provides a list view of sessions with filtering, sorting, and actions
 * like creating new sessions or deleting existing ones.
 */
import { useState, useEffect, useCallback } from 'react';
import type { SessionSummary, SessionListOptions } from '@/services/sessionApi';
import { listSessions, endSession } from '@/services/sessionApi';
import type { Session } from '@/types';
import { SessionCard } from './SessionCard';
import { CreateSessionModal } from './CreateSessionModal';
import { DeleteSessionModal } from './DeleteSessionModal';

/**
 * Filter type for session list.
 */
export type SessionFilter = 'all' | 'active' | 'ended';

/**
 * Sort options for session list.
 */
export type SessionSortBy = 'started_at' | 'ended_at' | 'name';
export type SessionSortOrder = 'asc' | 'desc';

/**
 * SessionList component props.
 */
export interface SessionListProps {
  /** Initial filter to apply */
  initialFilter?: SessionFilter;
  /** Maximum number of sessions to show (0 for unlimited) */
  limit?: number;
  /** Whether to show the header with filters */
  showHeader?: boolean;
  /** Callback when a session is created (receives the new session) */
  onSessionCreated?: (session: Session) => void;
  /** Legacy callback when create button is clicked (deprecated, use onSessionCreated) */
  onCreateSession?: () => void;
  /** Callback when session list changes */
  onSessionsChange?: (sessions: SessionSummary[]) => void;
  /** Whether to use compact card view */
  compact?: boolean;
  /** Title for the section */
  title?: string;
  /** Whether to show the create session button */
  showCreateButton?: boolean;
}

/**
 * Loading state type.
 */
type LoadingState = 'idle' | 'loading' | 'refreshing' | 'deleting' | 'ending';

/**
 * Session list component with filtering and management.
 */
export const SessionList: React.FC<SessionListProps> = ({
  initialFilter = 'all',
  limit = 0,
  showHeader = true,
  onSessionCreated,
  onCreateSession,
  onSessionsChange,
  compact = false,
  title = 'Recent Sessions',
  showCreateButton = true,
}) => {
  // State
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [filter, setFilter] = useState<SessionFilter>(initialFilter);
  const [sortBy, setSortBy] = useState<SessionSortBy>('started_at');
  const [sortOrder, setSortOrder] = useState<SessionSortOrder>('desc');
  const [loadingState, setLoadingState] = useState<LoadingState>('loading');
  const [error, setError] = useState<string | null>(null);
  const [actionSessionId, setActionSessionId] = useState<string | null>(null);

  // Modal state
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [sessionToDelete, setSessionToDelete] = useState<SessionSummary | null>(null);

  /** Build query options from current state */
  const buildQueryOptions = useCallback((): SessionListOptions => {
    const options: SessionListOptions = {
      sortBy,
      sortOrder,
    };

    if (filter === 'active') {
      options.active = true;
    } else if (filter === 'ended') {
      options.active = false;
    }

    if (limit > 0) {
      options.limit = limit;
    }

    return options;
  }, [filter, sortBy, sortOrder, limit]);

  /** Load sessions from API */
  const loadSessions = useCallback(async (showLoading = true) => {
    if (showLoading) {
      setLoadingState('loading');
    } else {
      setLoadingState('refreshing');
    }
    setError(null);

    try {
      const options = buildQueryOptions();
      const result = await listSessions(options);
      setSessions(result.sessions);
      onSessionsChange?.(result.sessions);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sessions');
    } finally {
      setLoadingState('idle');
    }
  }, [buildQueryOptions, onSessionsChange]);

  /** Load sessions on mount and when filters change */
  useEffect(() => {
    loadSessions();
  }, [filter, sortBy, sortOrder]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Handle delete session - opens confirmation modal */
  const handleDeleteSession = useCallback((id: string) => {
    const session = sessions.find((s) => s.id === id);
    if (session) {
      setSessionToDelete(session);
      setShowDeleteModal(true);
    }
  }, [sessions]);

  /** Handle delete confirmation from modal */
  const handleDeleteConfirmed = useCallback((sessionId: string) => {
    // Remove from local state
    setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    // Close modal
    setShowDeleteModal(false);
    setSessionToDelete(null);
    // Notify parent
    onSessionsChange?.(sessions.filter((s) => s.id !== sessionId));
  }, [sessions, onSessionsChange]);

  /** Handle create button click */
  const handleCreateClick = useCallback(() => {
    // If legacy callback is provided, use it
    if (onCreateSession) {
      onCreateSession();
    } else {
      // Otherwise show the create modal
      setShowCreateModal(true);
    }
  }, [onCreateSession]);

  /** Handle session created from modal */
  const handleSessionCreated = useCallback((session: Session) => {
    // Convert Session to SessionSummary for the list
    const summary: SessionSummary = {
      id: session.id,
      name: session.name,
      description: session.description,
      startedAt: session.startedAt,
      endedAt: session.endedAt ?? undefined,
      isActive: !session.endedAt,
      stats: session.stats ?? {
        conversationCount: 0,
        messageCount: 0,
        measurementCount: 0,
        bilateralCount: 0,
        avgDEff: 0,
        avgBeta: 0,
        avgAlignment: 0,
      },
    };

    // Add to beginning of list
    setSessions((prev) => [summary, ...prev]);
    // Close modal
    setShowCreateModal(false);
    // Notify parent
    onSessionCreated?.(session);
  }, [onSessionCreated]);

  /** Handle end session */
  const handleEndSession = useCallback(async (id: string) => {
    setLoadingState('ending');
    setActionSessionId(id);
    setError(null);

    try {
      await endSession(id);
      // Refresh list to get updated session
      await loadSessions(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to end session');
    } finally {
      setLoadingState('idle');
      setActionSessionId(null);
    }
  }, [loadSessions]);

  /** Handle filter change */
  const handleFilterChange = useCallback((newFilter: SessionFilter) => {
    setFilter(newFilter);
  }, []);

  /** Handle sort change */
  const handleSortChange = useCallback((newSortBy: SessionSortBy) => {
    if (newSortBy === sortBy) {
      // Toggle order if same field
      setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(newSortBy);
      setSortOrder('desc');
    }
  }, [sortBy]);

  const isLoading = loadingState === 'loading';
  const isRefreshing = loadingState === 'refreshing';

  return (
    <div className="space-y-4">
      {/* Header with filters and actions */}
      {showHeader && (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
            {isRefreshing && (
              <svg
                className="animate-spin h-4 w-4 text-weaver-600"
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
            )}
          </div>

          <div className="flex items-center gap-3">
            {/* Filter Buttons */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              {(['all', 'active', 'ended'] as SessionFilter[]).map((filterOption) => (
                <button
                  key={filterOption}
                  type="button"
                  onClick={() => handleFilterChange(filterOption)}
                  className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                    filter === filterOption
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  {filterOption.charAt(0).toUpperCase() + filterOption.slice(1)}
                </button>
              ))}
            </div>

            {/* Sort Dropdown */}
            <select
              value={`${sortBy}_${sortOrder}`}
              onChange={(e) => {
                const [newSortBy, newSortOrder] = e.target.value.split('_') as [SessionSortBy, SessionSortOrder];
                setSortBy(newSortBy);
                setSortOrder(newSortOrder);
              }}
              className="input text-sm py-1.5"
            >
              <option value="started_at_desc">Newest First</option>
              <option value="started_at_asc">Oldest First</option>
              <option value="name_asc">Name A-Z</option>
              <option value="name_desc">Name Z-A</option>
            </select>

            {/* Refresh Button */}
            <button
              type="button"
              onClick={() => loadSessions(false)}
              disabled={isLoading}
              className="btn-secondary p-2"
              title="Refresh"
            >
              <svg
                className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                />
              </svg>
            </button>

            {/* New Session Button */}
            {(showCreateButton || onCreateSession || onSessionCreated) && (
              <button
                type="button"
                onClick={handleCreateClick}
                disabled={isLoading}
                className="btn-primary flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                New Session
              </button>
            )}
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <svg className="w-5 h-5 text-red-400 flex-shrink-0" viewBox="0 0 20 20" fill="currentColor">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <div className="ml-3">
              <p className="text-sm font-medium text-red-800">{error}</p>
              <button
                type="button"
                onClick={() => loadSessions()}
                className="mt-2 text-sm text-red-600 hover:text-red-700 underline"
              >
                Try again
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card animate-pulse">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="h-5 w-48 bg-gray-200 rounded" />
                  <div className="mt-2 h-4 w-72 bg-gray-100 rounded" />
                </div>
                <div className="h-6 w-16 bg-gray-100 rounded-full" />
              </div>
              <div className="mt-4 flex gap-4">
                <div className="h-4 w-24 bg-gray-100 rounded" />
                <div className="h-4 w-24 bg-gray-100 rounded" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Session List */}
      {!isLoading && sessions.length > 0 && (
        <div className="space-y-4">
          {sessions.map((session) => (
            <SessionCard
              key={session.id}
              session={session}
              onDelete={handleDeleteSession}
              onEnd={handleEndSession}
              loading={actionSessionId === session.id}
              compact={compact}
            />
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && sessions.length === 0 && (
        <div className="card text-center py-12">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
            />
          </svg>
          <h3 className="mt-4 text-lg font-medium text-gray-900">No sessions found</h3>
          <p className="mt-2 text-sm text-gray-500">
            {filter === 'all'
              ? 'Start a new session to begin your research.'
              : filter === 'active'
              ? 'No active sessions. Start a new session to begin.'
              : 'No ended sessions yet.'}
          </p>
          {(showCreateButton || onCreateSession || onSessionCreated) && filter !== 'ended' && (
            <button
              type="button"
              onClick={handleCreateClick}
              className="mt-4 btn-primary"
            >
              Start New Session
            </button>
          )}
        </div>
      )}

      {/* Create Session Modal */}
      <CreateSessionModal
        isOpen={showCreateModal}
        onClose={() => setShowCreateModal(false)}
        onCreated={handleSessionCreated}
      />

      {/* Delete Session Modal */}
      <DeleteSessionModal
        isOpen={showDeleteModal}
        session={sessionToDelete}
        onClose={() => {
          setShowDeleteModal(false);
          setSessionToDelete(null);
        }}
        onDeleted={handleDeleteConfirmed}
      />
    </div>
  );
};

export default SessionList;
