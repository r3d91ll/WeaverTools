/**
 * Measurement types matching Yarn/measurement.go
 * and Weaver/pkg/api/events.go
 *
 * Note: Field names use camelCase to match the API JSON responses.
 */

/**
 * BetaStatus indicates the quality of the beta value.
 * Beta is the collapse indicator from the Conveyance Framework.
 * Lower values indicate better dimensional preservation.
 */
export type BetaStatus = 'optimal' | 'monitor' | 'concerning' | 'critical' | 'unknown';

/**
 * Beta status thresholds and ranges.
 * - optimal: beta in [1.5, 2.0) - ideal range
 * - monitor: beta in [2.0, 2.5) - acceptable, watch for drift
 * - concerning: beta in [2.5, 3.0) - dimensional compression detected
 * - critical: beta >= 3.0 - severe collapse, intervention needed
 * - unknown: beta <= 0 or in (0, 1.5) - invalid or uncategorized
 */
export const BETA_STATUS_RANGES = {
  optimal: { min: 1.5, max: 2.0 },
  monitor: { min: 2.0, max: 2.5 },
  concerning: { min: 2.5, max: 3.0 },
  critical: { min: 3.0, max: Infinity },
} as const;

/**
 * Computes the BetaStatus based on beta value.
 * Mirrors the Go ComputeBetaStatus function.
 */
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

/**
 * HiddenState represents the boundary object - semantic state before text projection.
 * Maps to HiddenState in Yarn/message.go.
 *
 * Memory note: Vector can be large (e.g., 4096 floats = 16KB for typical LLMs).
 * For models with larger hidden dimensions (e.g., 8192), expect ~32KB per state.
 */
export interface HiddenState {
  /** Hidden state vector, typically 2048-8192 float32 values */
  vector: number[];
  /** Original tensor shape, e.g., [1, seq_len, hidden_dim] */
  shape: number[];
  /** Layer index this state was extracted from */
  layer: number;
  /** Data type, typically "float32" */
  dtype: string;
}

/** Get the hidden dimension size from a HiddenState. */
export function getHiddenDimension(state: HiddenState | null | undefined): number {
  if (!state) return 0;
  if (state.shape.length < 2) return state.vector.length;
  return state.shape[state.shape.length - 1];
}

/**
 * Measurement contains conveyance metrics from a single agent interaction.
 * Maps to Measurement in Yarn/measurement.go.
 */
export interface Measurement {
  id: string;
  timestamp: string;

  // Session context
  sessionId: string;
  conversationId: string;
  turnNumber: number;

  // Participants
  senderId: string;
  senderName: string;
  senderRole: string;

  receiverId: string;
  receiverName: string;
  receiverRole: string;

  // Boundary objects (hidden states)
  senderHidden?: HiddenState | null;
  receiverHidden?: HiddenState | null;

  // Core conveyance metrics
  /** Effective dimensionality */
  dEff: number;
  /** Collapse indicator */
  beta: number;
  /** Cosine similarity */
  alignment: number;
  /** Bilateral conveyance */
  cPair: number;

  // Quality indicators
  betaStatus: BetaStatus;
  isUnilateral: boolean;

  // Message context
  messageContent?: string;
  tokenCount?: number;
}

/** Check if a measurement is bilateral (both sender and receiver have hidden states). */
export function isBilateral(measurement: Measurement): boolean {
  const hasSender = measurement.senderHidden && measurement.senderHidden.vector.length > 0;
  const hasReceiver = measurement.receiverHidden && measurement.receiverHidden.vector.length > 0;
  return Boolean(hasSender && hasReceiver);
}

/**
 * MeasurementEvent provides a richer measurement event structure.
 * Maps to MeasurementEvent in events.go.
 */
export interface MeasurementEvent {
  // Core identification
  id: string;
  timestamp: string;

  // Session context
  sessionId?: string;
  conversationId?: string;
  turn: number;

  // Participants
  senderId?: string;
  senderName?: string;
  senderRole?: string;

  receiverId?: string;
  receiverName?: string;
  receiverRole?: string;

  // Core conveyance metrics
  /** Effective dimensionality */
  deff: number;
  /** Collapse indicator */
  beta: number;
  /** Cosine similarity */
  alignment: number;
  /** Bilateral conveyance */
  cpair: number;

  // Quality indicators
  betaStatus?: BetaStatus;
  isUnilateral?: boolean;

  // Message context
  tokenCount?: number;
}

/**
 * MeasurementBatchEvent contains multiple measurements for batch updates.
 * Maps to MeasurementBatchEvent in events.go.
 */
export interface MeasurementBatchEvent {
  measurements: MeasurementEvent[];
  sessionId?: string;
  turnRange: [number, number]; // [start, end]
}

/**
 * BetaAlertEvent is sent when beta reaches concerning or critical levels.
 * Maps to BetaAlertEvent in events.go.
 */
export interface BetaAlertEvent {
  measurementId: string;
  sessionId?: string;
  turn: number;
  beta: number;
  status: BetaStatus;
  previousBeta?: number;
  previousStatus?: BetaStatus;
  alertMessage: string;
  timestamp: string;
}

/**
 * ConversationTurnEvent is sent for each conversation turn with context.
 * Maps to ConversationTurnEvent in events.go.
 */
export interface ConversationTurnEvent {
  sessionId: string;
  conversationId: string;
  turn: number;
  senderName: string;
  receiverName: string;
  timestamp: string;
}

/**
 * MeasurementData is a simplified measurement structure for charts.
 * Used for backward compatibility with simple data displays.
 */
export interface MeasurementData {
  turn: number;
  deff: number;
  beta: number;
  alignment: number;
  cpair: number;
  sender?: string;
  receiver?: string;
}
