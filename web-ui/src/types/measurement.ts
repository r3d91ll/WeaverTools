/**
 * Measurement types matching Yarn/measurement.go
 * Conveyance metrics from agent interactions.
 */

import type { HiddenState } from './backend';

/**
 * BetaStatus indicates the quality of the beta value.
 * Beta is the collapse indicator from the Conveyance Framework.
 * Lower values indicate better dimensional preservation.
 */
export type BetaStatus = 'optimal' | 'monitor' | 'concerning' | 'critical' | 'unknown';

/** Beta status thresholds. */
export const BETA_THRESHOLDS = {
  /** Optimal: beta in [1.5, 2.0) - ideal range */
  optimal: { min: 1.5, max: 2.0 },
  /** Monitor: beta in [2.0, 2.5) - acceptable, watch for drift */
  monitor: { min: 2.0, max: 2.5 },
  /** Concerning: beta in [2.5, 3.0) - dimensional compression detected */
  concerning: { min: 2.5, max: 3.0 },
  /** Critical: beta >= 3.0 - severe collapse, intervention needed */
  critical: { min: 3.0, max: Infinity },
} as const;

/** Compute BetaStatus from a beta value. */
export function computeBetaStatus(beta: number): BetaStatus {
  if (beta <= 0) return 'unknown';
  if (beta < 1.5) return 'unknown';
  if (beta < 2.0) return 'optimal';
  if (beta < 2.5) return 'monitor';
  if (beta < 3.0) return 'concerning';
  return 'critical';
}

/** Check if a BetaStatus is valid. */
export function isValidBetaStatus(status: string): status is BetaStatus {
  return ['optimal', 'monitor', 'concerning', 'critical', 'unknown'].includes(status);
}

/** Measurement contains conveyance metrics from a single agent interaction. */
export interface Measurement {
  id: string;
  timestamp: string; // ISO 8601 date string

  // Session context
  session_id: string;
  conversation_id: string;
  turn_number: number;

  // Participants
  sender_id: string;
  sender_name: string;
  sender_role: string;

  receiver_id: string;
  receiver_name: string;
  receiver_role: string;

  // Boundary objects (hidden states)
  sender_hidden?: HiddenState;
  receiver_hidden?: HiddenState;

  // Core conveyance metrics
  /** Effective dimensionality */
  d_eff: number;
  /** Collapse indicator */
  beta: number;
  /** Cosine similarity */
  alignment: number;
  /** Bilateral conveyance */
  c_pair: number;

  // Quality indicators
  beta_status: BetaStatus;
  is_unilateral: boolean;

  // Message context
  message_content?: string;
  token_count?: number;
}

/** Check if a measurement is bilateral (both sender and receiver have hidden states). */
export function isBilateral(measurement: Measurement): boolean {
  const hasSender = measurement.sender_hidden && measurement.sender_hidden.vector.length > 0;
  const hasReceiver = measurement.receiver_hidden && measurement.receiver_hidden.vector.length > 0;
  return Boolean(hasSender && hasReceiver);
}
