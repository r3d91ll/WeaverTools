/**
 * Session types matching Yarn/session.go and Yarn/conversation.go
 * Research session management types.
 */

import type { Measurement } from './measurement';
import type { HiddenState } from './backend';

/** MessageRole represents the sender type. */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

/** Check if a role is valid. */
export function isValidMessageRole(role: string): role is MessageRole {
  return ['system', 'user', 'assistant', 'tool'].includes(role);
}

/** Message is the atomic unit of communication between agents. */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string; // ISO 8601 date string
  agent_id?: string;
  agent_name?: string;
  hidden_state?: HiddenState;
  metadata?: Record<string, unknown>;

  // Tool-related fields
  tool_call_id?: string;
  tool_name?: string;
}

/** Participant tracks an agent's involvement in the conversation. */
export interface Participant {
  agent_id: string;
  agent_name: string;
  role: string;
  joined_at: string; // ISO 8601 date string
  message_count: number;
}

/** Conversation is an ordered sequence of messages with participant tracking. */
export interface Conversation {
  id: string;
  name: string;
  messages: Message[];
  participants: Record<string, Participant>;
  created_at: string; // ISO 8601 date string
  updated_at: string; // ISO 8601 date string
  metadata?: Record<string, unknown>;
}

/** MeasurementMode determines when measurements are captured. */
export type MeasurementMode = 'passive' | 'active' | 'triggered';

/** Check if a measurement mode is valid. */
export function isValidMeasurementMode(mode: string): mode is MeasurementMode {
  return ['passive', 'active', 'triggered'].includes(mode);
}

/** SessionConfig holds session configuration (runtime). */
export interface SessionConfig {
  measurement_mode: MeasurementMode;
  auto_export: boolean;
  export_path: string;
}

/** Session is a named research session grouping conversations and measurements. */
export interface Session {
  id: string;
  name: string;
  description: string;
  started_at: string; // ISO 8601 date string
  ended_at?: string; // ISO 8601 date string
  config: SessionConfig;
  metadata?: Record<string, unknown>;
  conversations: Conversation[];
  measurements: Measurement[];
}

/** SessionStats holds session statistics. */
export interface SessionStats {
  conversation_count: number;
  message_count: number;
  measurement_count: number;
  bilateral_count: number;
  avg_d_eff: number;
  avg_beta: number;
  avg_alignment: number;
}

/** SessionStatus represents the current status of a session in the registry. */
export interface SessionStatus {
  name: string;
  id: string;
  is_active: boolean;
  started_at: string; // ISO 8601 date string
  ended_at?: string; // ISO 8601 date string
  stats: SessionStats;
}

/** Calculate session statistics from session data. */
export function calculateSessionStats(session: Session): SessionStats {
  const stats: SessionStats = {
    conversation_count: session.conversations.length,
    message_count: 0,
    measurement_count: session.measurements.length,
    bilateral_count: 0,
    avg_d_eff: 0,
    avg_beta: 0,
    avg_alignment: 0,
  };

  // Count messages across all conversations
  for (const conv of session.conversations) {
    stats.message_count += conv.messages.length;
  }

  // Calculate averages from measurements
  if (session.measurements.length > 0) {
    let totalDEff = 0;
    let totalBeta = 0;
    let totalAlignment = 0;

    for (const m of session.measurements) {
      totalDEff += m.d_eff;
      totalBeta += m.beta;
      totalAlignment += m.alignment;

      // Count bilateral measurements
      const hasSender = m.sender_hidden && m.sender_hidden.vector.length > 0;
      const hasReceiver = m.receiver_hidden && m.receiver_hidden.vector.length > 0;
      if (hasSender && hasReceiver) {
        stats.bilateral_count++;
      }
    }

    const n = session.measurements.length;
    stats.avg_d_eff = totalDEff / n;
    stats.avg_beta = totalBeta / n;
    stats.avg_alignment = totalAlignment / n;
  }

  return stats;
}
