/**
 * SessionContext provides global session state management.
 *
 * Features:
 * - Track current active session
 * - Manage session list with filtering and pagination
 * - Create, update, delete, and end sessions
 * - Subscribe to session events via WebSocket
 * - Track session statistics
 */

import React, {
  createContext,
  useContext,
  useState,
  useCallback,
  useEffect,
  useMemo,
  type ReactNode,
} from 'react';

import type { Session, SessionStartEvent, SessionEndEvent } from '@/types';
import {
  listSessions,
  getSession,
  createSession,
  updateSession,
  deleteSession,
  endSession,
  type SessionListOptions,
  type SessionSummary,
  type CreateSessionRequest,
  type UpdateSessionRequest,
} from '@/services/sessionApi';

import { useWebSocketEvent } from '@/hooks/useWebSocket';

/**
 * SessionContextValue provides access to session state and actions.
 */
export interface SessionContextValue {
  /** Current/active session, null if none selected */
  currentSession: Session | null;
  /** ID of the current session */
  currentSessionId: string | null;
  /** List of sessions */
  sessions: SessionSummary[];
  /** Total number of sessions (for pagination) */
  totalSessions: number;
  /** Whether sessions are being loaded */
  isLoading: boolean;
  /** Whether a session operation is in progress */
  isOperating: boolean;
  /** Current error message, if any */
  error: string | null;
  /** Load sessions list from backend */
  loadSessions: (options?: SessionListOptions) => Promise<void>;
  /** Load a specific session and set as current */
  loadSession: (id: string) => Promise<Session>;
  /** Set the current session by ID */
  setCurrentSessionId: (id: string | null) => void;
  /** Create a new session */
  createNewSession: (request: CreateSessionRequest) => Promise<Session>;
  /** Update a session */
  updateCurrentSession: (request: UpdateSessionRequest) => Promise<Session>;
  /** Delete a session */
  deleteSessionById: (id: string) => Promise<void>;
  /** End the current session */
  endCurrentSession: () => Promise<Session>;
  /** Clear current session */
  clearCurrentSession: () => void;
  /** Refresh current session data */
  refreshCurrentSession: () => Promise<void>;
  /** Refresh sessions list */
  refreshSessions: () => Promise<void>;
  /** Clear the current error */
  clearError: () => void;
  /** Get active sessions from the list */
  activeSessions: SessionSummary[];
  /** Get ended sessions from the list */
  endedSessions: SessionSummary[];
}

/**
 * SessionProviderProps defines the props for SessionProvider.
 */
export interface SessionProviderProps {
  children: ReactNode;
  /** Whether to auto-load sessions on mount (default: true) */
  autoLoad?: boolean;
  /** Default list options */
  defaultListOptions?: SessionListOptions;
}

/** Context for session state */
const SessionContext = createContext<SessionContextValue | undefined>(undefined);

/**
 * SessionProvider wraps the application with session state management.
 *
 * @example
 * ```tsx
 * function App() {
 *   return (
 *     <SessionProvider>
 *       <YourApp />
 *     </SessionProvider>
 *   );
 * }
 *
 * function SessionDisplay() {
 *   const { currentSession, sessions, isLoading } = useSession();
 *
 *   if (isLoading) return <Loading />;
 *
 *   return (
 *     <div>
 *       <h2>Current: {currentSession?.name ?? 'None'}</h2>
 *       <ul>
 *         {sessions.map(s => <li key={s.id}>{s.name}</li>)}
 *       </ul>
 *     </div>
 *   );
 * }
 * ```
 */
export function SessionProvider({
  children,
  autoLoad = true,
  defaultListOptions,
}: SessionProviderProps): React.ReactElement {
  const [currentSession, setCurrentSession] = useState<Session | null>(null);
  const [currentSessionId, setCurrentSessionIdState] = useState<string | null>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [totalSessions, setTotalSessions] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isOperating, setIsOperating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [listOptions, setListOptions] = useState<SessionListOptions | undefined>(defaultListOptions);

  /**
   * Load sessions list from backend.
   */
  const loadSessions = useCallback(async (options?: SessionListOptions): Promise<void> => {
    const mergedOptions = options ?? listOptions;
    setListOptions(mergedOptions);
    setIsLoading(true);
    setError(null);

    try {
      const result = await listSessions(mergedOptions);
      setSessions(result.sessions);
      setTotalSessions(result.total);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load sessions';
      setError(message);
    } finally {
      setIsLoading(false);
    }
  }, [listOptions]);

  /**
   * Load a specific session and set as current.
   */
  const loadSession = useCallback(async (id: string): Promise<Session> => {
    setIsOperating(true);
    setError(null);

    try {
      const session = await getSession(id);
      setCurrentSession(session);
      setCurrentSessionIdState(id);
      return session;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load session';
      setError(message);
      throw err;
    } finally {
      setIsOperating(false);
    }
  }, []);

  /**
   * Set the current session by ID.
   */
  const setCurrentSessionId = useCallback((id: string | null): void => {
    setCurrentSessionIdState(id);
    if (id === null) {
      setCurrentSession(null);
    } else if (currentSession?.id !== id) {
      // Load the session if ID changed
      loadSession(id).catch(() => {
        // Error already handled in loadSession
      });
    }
  }, [currentSession?.id, loadSession]);

  /**
   * Create a new session.
   */
  const createNewSession = useCallback(async (request: CreateSessionRequest): Promise<Session> => {
    setIsOperating(true);
    setError(null);

    try {
      const session = await createSession(request);
      // Add to sessions list
      setSessions((prev) => [
        {
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
        },
        ...prev,
      ]);
      setTotalSessions((prev) => prev + 1);
      // Set as current
      setCurrentSession(session);
      setCurrentSessionIdState(session.id);
      return session;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to create session';
      setError(message);
      throw err;
    } finally {
      setIsOperating(false);
    }
  }, []);

  /**
   * Update the current session.
   */
  const updateCurrentSession = useCallback(async (request: UpdateSessionRequest): Promise<Session> => {
    if (!currentSessionId) {
      throw new Error('No current session to update');
    }

    setIsOperating(true);
    setError(null);

    try {
      const session = await updateSession(currentSessionId, request);
      setCurrentSession(session);
      // Update in sessions list
      setSessions((prev) =>
        prev.map((s) =>
          s.id === session.id
            ? {
                ...s,
                name: session.name,
                description: session.description,
              }
            : s
        )
      );
      return session;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to update session';
      setError(message);
      throw err;
    } finally {
      setIsOperating(false);
    }
  }, [currentSessionId]);

  /**
   * Delete a session by ID.
   */
  const deleteSessionById = useCallback(async (id: string): Promise<void> => {
    setIsOperating(true);
    setError(null);

    try {
      await deleteSession(id);
      // Remove from sessions list
      setSessions((prev) => prev.filter((s) => s.id !== id));
      setTotalSessions((prev) => prev - 1);
      // Clear current if deleted
      if (currentSessionId === id) {
        setCurrentSession(null);
        setCurrentSessionIdState(null);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to delete session';
      setError(message);
      throw err;
    } finally {
      setIsOperating(false);
    }
  }, [currentSessionId]);

  /**
   * End the current session.
   */
  const endCurrentSession = useCallback(async (): Promise<Session> => {
    if (!currentSessionId) {
      throw new Error('No current session to end');
    }

    setIsOperating(true);
    setError(null);

    try {
      const session = await endSession(currentSessionId);
      setCurrentSession(session);
      // Update in sessions list
      setSessions((prev) =>
        prev.map((s) =>
          s.id === session.id
            ? {
                ...s,
                endedAt: session.endedAt ?? undefined,
                isActive: false,
              }
            : s
        )
      );
      return session;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to end session';
      setError(message);
      throw err;
    } finally {
      setIsOperating(false);
    }
  }, [currentSessionId]);

  /**
   * Clear current session.
   */
  const clearCurrentSession = useCallback((): void => {
    setCurrentSession(null);
    setCurrentSessionIdState(null);
  }, []);

  /**
   * Refresh current session data.
   */
  const refreshCurrentSession = useCallback(async (): Promise<void> => {
    if (currentSessionId) {
      await loadSession(currentSessionId);
    }
  }, [currentSessionId, loadSession]);

  /**
   * Refresh sessions list.
   */
  const refreshSessions = useCallback(async (): Promise<void> => {
    await loadSessions(listOptions);
  }, [loadSessions, listOptions]);

  /**
   * Clear the current error.
   */
  const clearError = useCallback((): void => {
    setError(null);
  }, []);

  /**
   * Derived state: active sessions.
   */
  const activeSessions = useMemo(
    () => sessions.filter((s) => s.isActive),
    [sessions]
  );

  /**
   * Derived state: ended sessions.
   */
  const endedSessions = useMemo(
    () => sessions.filter((s) => !s.isActive),
    [sessions]
  );

  // Handle session start events from WebSocket
  useWebSocketEvent<SessionStartEvent>(
    'session_start',
    (event) => {
      // Add new session to list if not already present
      setSessions((prev) => {
        if (prev.some((s) => s.id === event.sessionId)) {
          return prev;
        }
        return [
          {
            id: event.sessionId,
            name: event.name ?? event.sessionId,
            description: event.description ?? '',
            startedAt: event.startedAt,
            isActive: true,
            stats: {
              conversationCount: 0,
              messageCount: 0,
              measurementCount: 0,
              bilateralCount: 0,
              avgDEff: 0,
              avgBeta: 0,
              avgAlignment: 0,
            },
          },
          ...prev,
        ];
      });
      setTotalSessions((prev) => prev + 1);
    },
    []
  );

  // Handle session end events from WebSocket
  useWebSocketEvent<SessionEndEvent>(
    'session_end',
    (event) => {
      // Update session in list
      setSessions((prev) =>
        prev.map((s) =>
          s.id === event.sessionId
            ? {
                ...s,
                endedAt: event.endedAt,
                isActive: false,
                stats: event.finalMetrics
                  ? {
                      conversationCount: s.stats.conversationCount,
                      messageCount: s.stats.messageCount,
                      measurementCount: event.finalMetrics.totalMeasurements,
                      bilateralCount: s.stats.bilateralCount,
                      avgDEff: event.finalMetrics.avgDeff,
                      avgBeta: event.finalMetrics.avgBeta,
                      avgAlignment: event.finalMetrics.avgAlignment,
                    }
                  : s.stats,
              }
            : s
        )
      );

      // Update current session if it was ended
      if (currentSessionId === event.sessionId) {
        setCurrentSession((prev) =>
          prev
            ? {
                ...prev,
                endedAt: event.endedAt,
              }
            : null
        );
      }
    },
    [currentSessionId]
  );

  // Auto-load sessions on mount
  useEffect(() => {
    if (autoLoad) {
      loadSessions();
    }
  }, [autoLoad, loadSessions]);

  const value: SessionContextValue = useMemo(
    () => ({
      currentSession,
      currentSessionId,
      sessions,
      totalSessions,
      isLoading,
      isOperating,
      error,
      loadSessions,
      loadSession,
      setCurrentSessionId,
      createNewSession,
      updateCurrentSession,
      deleteSessionById,
      endCurrentSession,
      clearCurrentSession,
      refreshCurrentSession,
      refreshSessions,
      clearError,
      activeSessions,
      endedSessions,
    }),
    [
      currentSession,
      currentSessionId,
      sessions,
      totalSessions,
      isLoading,
      isOperating,
      error,
      loadSessions,
      loadSession,
      setCurrentSessionId,
      createNewSession,
      updateCurrentSession,
      deleteSessionById,
      endCurrentSession,
      clearCurrentSession,
      refreshCurrentSession,
      refreshSessions,
      clearError,
      activeSessions,
      endedSessions,
    ]
  );

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

/**
 * useSession hook provides access to the session context.
 *
 * @throws Error if used outside of SessionProvider
 * @returns Session context value
 *
 * @example
 * ```tsx
 * function SessionList() {
 *   const { sessions, isLoading, loadSession } = useSession();
 *
 *   if (isLoading) return <Loading />;
 *
 *   return (
 *     <ul>
 *       {sessions.map(session => (
 *         <li key={session.id}>
 *           <button onClick={() => loadSession(session.id)}>
 *             {session.name}
 *           </button>
 *         </li>
 *       ))}
 *     </ul>
 *   );
 * }
 * ```
 */
export function useSession(): SessionContextValue {
  const context = useContext(SessionContext);
  if (context === undefined) {
    throw new Error('useSession must be used within a SessionProvider');
  }
  return context;
}

/**
 * useCurrentSession hook provides access to the current session.
 *
 * @returns Current session info and actions
 *
 * @example
 * ```tsx
 * function SessionHeader() {
 *   const { session, isActive, endSession, clear } = useCurrentSession();
 *
 *   if (!session) return <div>No session selected</div>;
 *
 *   return (
 *     <div>
 *       <h1>{session.name}</h1>
 *       {isActive && <button onClick={endSession}>End Session</button>}
 *       <button onClick={clear}>Close</button>
 *     </div>
 *   );
 * }
 * ```
 */
export function useCurrentSession(): {
  session: Session | null;
  sessionId: string | null;
  isActive: boolean;
  isOperating: boolean;
  error: string | null;
  endSession: () => Promise<Session>;
  clear: () => void;
  refresh: () => Promise<void>;
} {
  const {
    currentSession,
    currentSessionId,
    isOperating,
    error,
    endCurrentSession,
    clearCurrentSession,
    refreshCurrentSession,
  } = useSession();

  const isActive = useMemo(
    () => currentSession !== null && !currentSession.endedAt,
    [currentSession]
  );

  return {
    session: currentSession,
    sessionId: currentSessionId,
    isActive,
    isOperating,
    error,
    endSession: endCurrentSession,
    clear: clearCurrentSession,
    refresh: refreshCurrentSession,
  };
}

export default SessionContext;
