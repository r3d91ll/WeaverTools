/**
 * Session API service for WeaverTools web-ui.
 * Handles session and conversation operations.
 */

import { get, post, put, del, buildQueryString, ApiError } from './api';
import type { Session, SessionStats, Message } from '@/types';
import type { SessionAPIConfig } from '@/types/session';

/** API endpoints for session operations */
const ENDPOINTS = {
  sessions: '/api/sessions',
  session: (id: string) => `/api/sessions/${encodeURIComponent(id)}`,
  end: (id: string) => `/api/sessions/${encodeURIComponent(id)}/end`,
  messages: (id: string) => `/api/sessions/${encodeURIComponent(id)}/messages`,
  conversations: (id: string) => `/api/sessions/${encodeURIComponent(id)}/conversations`,
} as const;

/** Session list filter options */
export interface SessionListOptions {
  /** Filter by active status */
  active?: boolean;
  /** Limit number of results */
  limit?: number;
  /** Offset for pagination */
  offset?: number;
  /** Sort by field */
  sortBy?: 'started_at' | 'ended_at' | 'name';
  /** Sort direction */
  sortOrder?: 'asc' | 'desc';
}

/** Session summary for list views */
export interface SessionSummary {
  id: string;
  name: string;
  description: string;
  startedAt: string;
  endedAt?: string;
  isActive: boolean;
  stats: SessionStats;
}

/** Create session request */
export interface CreateSessionRequest {
  name: string;
  description?: string;
  config?: Partial<SessionAPIConfig>;
}

/** Update session request */
export interface UpdateSessionRequest {
  name?: string;
  description?: string;
  metadata?: Record<string, unknown>;
}

/** Sessions list response */
interface SessionsResponse {
  sessions: SessionSummary[];
  total: number;
}

/** Single session response */
interface SessionResponse {
  session: Session;
}

/** Messages response */
interface MessagesResponse {
  messages: Message[];
  total: number;
}

/**
 * List sessions with optional filtering.
 * @param options - Filter and pagination options
 * @returns Promise resolving to session summaries
 * @throws ApiError on failure
 */
export async function listSessions(
  options?: SessionListOptions
): Promise<{ sessions: SessionSummary[]; total: number }> {
  const query = buildQueryString({
    active: options?.active,
    limit: options?.limit,
    offset: options?.offset,
    sort_by: options?.sortBy,
    sort_order: options?.sortOrder,
  });

  const response = await get<SessionsResponse>(`${ENDPOINTS.sessions}${query}`);
  return {
    sessions: response.data.sessions,
    total: response.data.total,
  };
}

/**
 * Get a session by ID.
 * @param id - Session ID
 * @returns Promise resolving to full Session
 * @throws ApiError on failure (404 if not found)
 */
export async function getSession(id: string): Promise<Session> {
  const response = await get<SessionResponse>(ENDPOINTS.session(id));
  return response.data.session;
}

/**
 * Create a new session.
 * @param request - Session creation parameters
 * @returns Promise resolving to created Session
 * @throws ApiError on failure
 */
export async function createSession(request: CreateSessionRequest): Promise<Session> {
  const response = await post<SessionResponse>(ENDPOINTS.sessions, request);
  return response.data.session;
}

/**
 * Update a session.
 * @param id - Session ID
 * @param request - Fields to update
 * @returns Promise resolving to updated Session
 * @throws ApiError on failure
 */
export async function updateSession(
  id: string,
  request: UpdateSessionRequest
): Promise<Session> {
  const response = await put<SessionResponse>(ENDPOINTS.session(id), request);
  return response.data.session;
}

/**
 * Delete a session.
 * @param id - Session ID
 * @throws ApiError on failure
 */
export async function deleteSession(id: string): Promise<void> {
  await del<void>(ENDPOINTS.session(id));
}

/**
 * End an active session.
 * @param id - Session ID
 * @returns Promise resolving to ended Session
 * @throws ApiError on failure
 */
export async function endSession(id: string): Promise<Session> {
  const response = await post<SessionResponse>(ENDPOINTS.end(id));
  return response.data.session;
}

/**
 * Get messages from a session.
 * @param id - Session ID
 * @param options - Pagination options
 * @returns Promise resolving to messages
 * @throws ApiError on failure
 */
export async function getSessionMessages(
  id: string,
  options?: { limit?: number; offset?: number }
): Promise<{ messages: Message[]; total: number }> {
  const query = buildQueryString({
    limit: options?.limit,
    offset: options?.offset,
  });

  const response = await get<MessagesResponse>(`${ENDPOINTS.messages(id)}${query}`);
  return {
    messages: response.data.messages,
    total: response.data.total,
  };
}

/**
 * Check if a session exists.
 * @param id - Session ID
 * @returns Promise resolving to existence status
 */
export async function sessionExists(id: string): Promise<boolean> {
  try {
    await getSession(id);
    return true;
  } catch (error) {
    if (error instanceof ApiError && error.isNotFound()) {
      return false;
    }
    throw error;
  }
}

/**
 * Check if a session is active.
 * @param session - Session or session summary
 * @returns True if session is active (no end time)
 */
export function isSessionActive(session: Session | SessionSummary): boolean {
  if ('isActive' in session) {
    return session.isActive;
  }
  return session.endedAt === undefined || session.endedAt === null;
}

/**
 * Get the duration of a session in milliseconds.
 * @param session - Session object
 * @returns Duration in ms, or null for active sessions
 */
export function getSessionDuration(session: Session | SessionSummary): number | null {
  const startTime = new Date(session.startedAt).getTime();
  const endTime = session.endedAt
    ? new Date(session.endedAt).getTime()
    : null;

  if (endTime === null) {
    return null; // Still active
  }

  return endTime - startTime;
}

/**
 * Format session duration as human-readable string.
 * @param durationMs - Duration in milliseconds
 * @returns Formatted duration string
 */
export function formatDuration(durationMs: number): string {
  const seconds = Math.floor(durationMs / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) {
    return `${days}d ${hours % 24}h`;
  }
  if (hours > 0) {
    return `${hours}h ${minutes % 60}m`;
  }
  if (minutes > 0) {
    return `${minutes}m ${seconds % 60}s`;
  }
  return `${seconds}s`;
}

/** Session API service object */
export const sessionApi = {
  listSessions,
  getSession,
  createSession,
  updateSession,
  deleteSession,
  endSession,
  getSessionMessages,
  sessionExists,
  isSessionActive,
  getSessionDuration,
  formatDuration,
};

export default sessionApi;
