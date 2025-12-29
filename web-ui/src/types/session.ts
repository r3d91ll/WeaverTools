/**
 * Session types matching Weaver/pkg/api/handlers_sessions.go
 * and Yarn/session.go
 *
 * Note: Field names use camelCase to match the API JSON responses.
 */

import type { MeasurementMode } from './config';

/**
 * SessionAPIConfig holds session configuration for the API.
 * Maps to SessionAPIConfig in handlers_sessions.go.
 */
export interface SessionAPIConfig {
  measurementMode: MeasurementMode | string;
  autoExport: boolean;
  exportPath: string;
}

/**
 * SessionStats holds session statistics for the API.
 * Maps to SessionAPIStats in handlers_sessions.go.
 */
export interface SessionStats {
  conversationCount: number;
  messageCount: number;
  measurementCount: number;
  bilateralCount: number;
  avgDEff: number;
  avgBeta: number;
  avgAlignment: number;
}

/**
 * Session represents a research session for the API.
 * Maps to Session in handlers_sessions.go.
 */
export interface Session {
  id: string;
  name: string;
  description: string;
  startedAt: string;
  endedAt?: string | null;
  config: SessionAPIConfig;
  metadata?: Record<string, unknown>;
  stats?: SessionStats | null;
}

/**
 * SessionListResponse is the JSON response for GET /api/sessions.
 */
export interface SessionListResponse {
  sessions: Session[];
}

/**
 * CreateSessionRequest is the expected JSON body for POST /api/sessions.
 * Maps to CreateSessionRequest in handlers_sessions.go.
 */
export interface CreateSessionRequest {
  name: string;
  description?: string;
  config?: SessionAPIConfig;
  metadata?: Record<string, unknown>;
}

/**
 * UpdateSessionRequest is the expected JSON body for PUT /api/sessions/:id.
 * Maps to UpdateSessionRequest in handlers_sessions.go.
 */
export interface UpdateSessionRequest {
  name?: string;
  description?: string;
  config?: SessionAPIConfig;
  metadata?: Record<string, unknown>;
}

/**
 * SessionMetricsSummary provides aggregate statistics for a session.
 * Maps to SessionMetricsSummary in events.go.
 */
export interface SessionMetricsSummary {
  totalMeasurements: number;
  avgDeff: number;
  avgBeta: number;
  avgAlignment: number;
  avgCpair: number;
  minBeta: number;
  maxBeta: number;
  betaAlertCount: number;
}

/**
 * SessionStartEvent is sent when a new measurement session begins.
 * Maps to SessionStartEvent in events.go.
 */
export interface SessionStartEvent {
  sessionId: string;
  name?: string;
  agentIds?: string[];
  startedAt: string;
  description?: string;
}

/**
 * SessionEndEvent is sent when a measurement session ends.
 * Maps to SessionEndEvent in events.go.
 */
export interface SessionEndEvent {
  sessionId: string;
  endedAt: string;
  totalTurns: number;
  finalMetrics?: SessionMetricsSummary;
}

/**
 * SessionStatus represents the current status of a session.
 */
export interface SessionStatus {
  name: string;
  id: string;
  isActive: boolean;
  startedAt: string;
  endedAt?: string;
  stats: SessionStats;
}

/**
 * Check if a measurement mode is valid.
 */
export function isValidMeasurementMode(mode: string): mode is MeasurementMode {
  return ['passive', 'active', 'triggered', 'disabled'].includes(mode);
}

/**
 * Calculate session statistics from session data.
 */
export function calculateSessionStats(
  conversationCount: number,
  messageCount: number,
  measurements: Array<{ dEff: number; beta: number; alignment: number; senderHidden?: unknown; receiverHidden?: unknown }>
): SessionStats {
  const stats: SessionStats = {
    conversationCount,
    messageCount,
    measurementCount: measurements.length,
    bilateralCount: 0,
    avgDEff: 0,
    avgBeta: 0,
    avgAlignment: 0,
  };

  if (measurements.length > 0) {
    let totalDEff = 0;
    let totalBeta = 0;
    let totalAlignment = 0;

    for (const m of measurements) {
      totalDEff += m.dEff;
      totalBeta += m.beta;
      totalAlignment += m.alignment;

      // Count bilateral measurements
      if (m.senderHidden && m.receiverHidden) {
        stats.bilateralCount++;
      }
    }

    const n = measurements.length;
    stats.avgDEff = totalDEff / n;
    stats.avgBeta = totalBeta / n;
    stats.avgAlignment = totalAlignment / n;
  }

  return stats;
}
