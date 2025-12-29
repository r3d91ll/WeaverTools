/**
 * React hook for managing measurement data with real-time WebSocket updates.
 *
 * Provides comprehensive measurement data management including:
 * - Real-time updates via WebSocket
 * - Session and conversation filtering
 * - Statistical computations
 * - Metric-specific utilities
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { useWebSocketEvent, useMeasurementEvents } from './useWebSocket';
import { getWebSocketService } from '@/services/websocket';
import { computeBetaStatus } from '@/types/measurement';
import type {
  MeasurementEvent,
  MeasurementData,
  MeasurementBatchEvent,
  BetaStatus,
} from '@/types';

/** Metric type for filtering and statistics. */
export type MetricType = 'deff' | 'beta' | 'alignment' | 'cpair';

/** All available metric types. */
export const METRIC_TYPES: MetricType[] = ['deff', 'beta', 'alignment', 'cpair'];

/** Options for useMeasurements hook. */
export interface UseMeasurementsOptions {
  /** Maximum number of measurements to keep in memory (default: 500) */
  maxMeasurements?: number;
  /** Session ID to filter measurements by (optional) */
  sessionId?: string;
  /** Conversation ID to filter measurements by (optional) */
  conversationId?: string;
  /** Whether to auto-connect to WebSocket (default: true) */
  autoConnect?: boolean;
  /** Initial measurements to populate (optional) */
  initialData?: MeasurementEvent[];
  /** Whether to subscribe to measurements channel (default: true) */
  subscribe?: boolean;
}

/** Statistical summary for a metric. */
export interface MetricStats {
  /** Metric type */
  metric: MetricType;
  /** Current/latest value */
  current: number | null;
  /** Average value */
  average: number;
  /** Minimum value */
  min: number;
  /** Maximum value */
  max: number;
  /** Standard deviation */
  stdDev: number;
  /** Number of samples */
  count: number;
  /** Trend: 'up' | 'down' | 'stable' */
  trend: 'up' | 'down' | 'stable';
  /** Percentage change from previous */
  changePercent: number;
}

/** Return type for useMeasurements hook. */
export interface UseMeasurementsReturn {
  /** All measurements (filtered by session/conversation if specified) */
  measurements: MeasurementEvent[];
  /** Measurements converted to simplified chart data format */
  chartData: MeasurementData[];
  /** Latest measurement received */
  latest: MeasurementEvent | null;
  /** Whether receiving live updates */
  isLive: boolean;
  /** Loading state for initial data fetch */
  loading: boolean;
  /** Error state */
  error: Error | null;
  /** Statistics for all metrics */
  stats: Record<MetricType, MetricStats>;
  /** Get statistics for a specific metric */
  getMetricStats: (metric: MetricType) => MetricStats;
  /** Get filtered measurements by turn range */
  getMeasurementsByTurnRange: (start: number, end: number) => MeasurementEvent[];
  /** Get beta status distribution */
  getBetaDistribution: () => Record<BetaStatus, number>;
  /** Clear all measurements */
  clear: () => void;
  /** Add measurements manually (e.g., from initial fetch) */
  addMeasurements: (measurements: MeasurementEvent[]) => void;
  /** Pause/resume live updates */
  pauseLiveUpdates: () => void;
  resumeLiveUpdates: () => void;
  isPaused: boolean;
  /** Last update timestamp */
  lastUpdateTime: Date | null;
  /** Total number of updates received */
  updateCount: number;
}

/**
 * Compute statistics for a metric from measurements.
 */
function computeMetricStats(
  measurements: MeasurementEvent[],
  metric: MetricType
): MetricStats {
  if (measurements.length === 0) {
    return {
      metric,
      current: null,
      average: 0,
      min: 0,
      max: 0,
      stdDev: 0,
      count: 0,
      trend: 'stable',
      changePercent: 0,
    };
  }

  const values = measurements.map((m) => m[metric]);
  const count = values.length;
  const sum = values.reduce((a, b) => a + b, 0);
  const average = sum / count;
  const min = Math.min(...values);
  const max = Math.max(...values);

  // Standard deviation
  const squareDiffs = values.map((v) => Math.pow(v - average, 2));
  const avgSquareDiff = squareDiffs.reduce((a, b) => a + b, 0) / count;
  const stdDev = Math.sqrt(avgSquareDiff);

  // Current value
  const current = values[values.length - 1];

  // Trend calculation (compare last 5 to previous 5, or available data)
  let trend: 'up' | 'down' | 'stable' = 'stable';
  let changePercent = 0;

  if (values.length >= 2) {
    const recentWindow = Math.min(5, Math.floor(values.length / 2));
    const recentValues = values.slice(-recentWindow);
    const previousValues = values.slice(-recentWindow * 2, -recentWindow);

    if (previousValues.length > 0) {
      const recentAvg = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
      const previousAvg = previousValues.reduce((a, b) => a + b, 0) / previousValues.length;

      if (previousAvg !== 0) {
        changePercent = ((recentAvg - previousAvg) / Math.abs(previousAvg)) * 100;
      }

      const threshold = 0.05; // 5% change threshold
      if (changePercent > threshold * 100) {
        trend = 'up';
      } else if (changePercent < -threshold * 100) {
        trend = 'down';
      }
    }
  }

  return {
    metric,
    current,
    average,
    min,
    max,
    stdDev,
    count,
    trend,
    changePercent,
  };
}

/**
 * Convert MeasurementEvent to simplified MeasurementData for charts.
 */
function toChartData(event: MeasurementEvent): MeasurementData {
  return {
    turn: event.turn,
    deff: event.deff,
    beta: event.beta,
    alignment: event.alignment,
    cpair: event.cpair,
    sender: event.senderName,
    receiver: event.receiverName,
  };
}

/**
 * Hook for managing measurement data with real-time WebSocket updates.
 *
 * @param options - Hook configuration options
 * @returns Measurement data and control functions
 *
 * @example
 * ```tsx
 * function MetricsDashboard() {
 *   const {
 *     measurements,
 *     chartData,
 *     latest,
 *     isLive,
 *     stats,
 *   } = useMeasurements({
 *     sessionId: 'session-123',
 *     maxMeasurements: 200,
 *   });
 *
 *   return (
 *     <div>
 *       <MetricChart data={chartData} metric="deff" />
 *       <p>D_eff Average: {stats.deff.average.toFixed(4)}</p>
 *       {isLive && <span>Live</span>}
 *     </div>
 *   );
 * }
 * ```
 */
export function useMeasurements(options: UseMeasurementsOptions = {}): UseMeasurementsReturn {
  const {
    maxMeasurements = 500,
    sessionId,
    conversationId,
    autoConnect = true,
    initialData = [],
    subscribe = true,
  } = options;

  // State
  const [measurements, setMeasurements] = useState<MeasurementEvent[]>(initialData);
  const [latest, setLatest] = useState<MeasurementEvent | null>(
    initialData.length > 0 ? initialData[initialData.length - 1] : null
  );
  const [isLive, setIsLive] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [lastUpdateTime, setLastUpdateTime] = useState<Date | null>(null);
  const [updateCount, setUpdateCount] = useState(0);

  // Refs for callback stability
  const isPausedRef = useRef(isPaused);
  isPausedRef.current = isPaused;

  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  const conversationIdRef = useRef(conversationId);
  conversationIdRef.current = conversationId;

  /**
   * Check if a measurement matches the current filters.
   */
  const matchesFilters = useCallback((event: MeasurementEvent): boolean => {
    if (sessionIdRef.current && event.sessionId !== sessionIdRef.current) {
      return false;
    }
    if (conversationIdRef.current && event.conversationId !== conversationIdRef.current) {
      return false;
    }
    return true;
  }, []);

  /**
   * Add a single measurement.
   */
  const addMeasurement = useCallback(
    (event: MeasurementEvent) => {
      if (!matchesFilters(event)) return;

      setMeasurements((prev) => {
        const next = [...prev, event];
        if (next.length > maxMeasurements) {
          return next.slice(-maxMeasurements);
        }
        return next;
      });
      setLatest(event);
      setLastUpdateTime(new Date());
      setUpdateCount((c) => c + 1);
      setIsLive(true);
    },
    [matchesFilters, maxMeasurements]
  );

  /**
   * Add multiple measurements (e.g., from batch update or initial fetch).
   */
  const addMeasurements = useCallback(
    (newMeasurements: MeasurementEvent[]) => {
      const filtered = newMeasurements.filter(matchesFilters);
      if (filtered.length === 0) return;

      setMeasurements((prev) => {
        // Merge and deduplicate by ID
        const existingIds = new Set(prev.map((m) => m.id));
        const newFiltered = filtered.filter((m) => !existingIds.has(m.id));

        // Sort by turn number
        const merged = [...prev, ...newFiltered].sort((a, b) => a.turn - b.turn);

        if (merged.length > maxMeasurements) {
          return merged.slice(-maxMeasurements);
        }
        return merged;
      });

      setLatest(filtered[filtered.length - 1]);
      setLastUpdateTime(new Date());
      setUpdateCount((c) => c + filtered.length);
    },
    [matchesFilters, maxMeasurements]
  );

  /**
   * Handle individual measurement event from WebSocket.
   */
  const handleMeasurement = useCallback(
    (data: MeasurementEvent) => {
      if (isPausedRef.current) return;
      addMeasurement(data);
    },
    [addMeasurement]
  );

  /**
   * Handle batch measurement event from WebSocket.
   */
  const handleMeasurementBatch = useCallback(
    (data: MeasurementBatchEvent) => {
      if (isPausedRef.current) return;
      addMeasurements(data.measurements);
    },
    [addMeasurements]
  );

  // Subscribe to WebSocket events
  useWebSocketEvent<MeasurementEvent>('measurement', handleMeasurement, [handleMeasurement]);
  useWebSocketEvent<MeasurementBatchEvent>('measurement_batch', handleMeasurementBatch, [
    handleMeasurementBatch,
  ]);

  // Subscribe to measurements channel
  useEffect(() => {
    if (!subscribe) return;

    const service = getWebSocketService();
    service.subscribe('measurements');

    return () => {
      service.unsubscribe('measurements');
    };
  }, [subscribe]);

  // Auto-connect logic
  useEffect(() => {
    if (!autoConnect) return;

    const service = getWebSocketService();
    if (!service.isConnected()) {
      service.connect().catch(() => {
        // Will auto-reconnect
      });
    }
  }, [autoConnect]);

  // Reset isLive after inactivity
  useEffect(() => {
    if (!isLive) return;

    const timeout = setTimeout(() => {
      setIsLive(false);
    }, 5000); // Mark as not live after 5 seconds of inactivity

    return () => clearTimeout(timeout);
  }, [isLive, lastUpdateTime]);

  /**
   * Clear all measurements.
   */
  const clear = useCallback(() => {
    setMeasurements([]);
    setLatest(null);
    setUpdateCount(0);
    setIsLive(false);
    setError(null);
  }, []);

  /**
   * Pause live updates.
   */
  const pauseLiveUpdates = useCallback(() => {
    setIsPaused(true);
    setIsLive(false);
  }, []);

  /**
   * Resume live updates.
   */
  const resumeLiveUpdates = useCallback(() => {
    setIsPaused(false);
  }, []);

  /**
   * Get measurements by turn range.
   */
  const getMeasurementsByTurnRange = useCallback(
    (start: number, end: number) => {
      return measurements.filter((m) => m.turn >= start && m.turn <= end);
    },
    [measurements]
  );

  /**
   * Get beta status distribution.
   */
  const getBetaDistribution = useCallback((): Record<BetaStatus, number> => {
    const distribution: Record<BetaStatus, number> = {
      optimal: 0,
      monitor: 0,
      concerning: 0,
      critical: 0,
      unknown: 0,
    };

    for (const m of measurements) {
      const status = m.betaStatus || computeBetaStatus(m.beta);
      distribution[status]++;
    }

    return distribution;
  }, [measurements]);

  // Compute chart data
  const chartData = useMemo(() => {
    return measurements.map(toChartData);
  }, [measurements]);

  // Compute stats for all metrics
  const stats = useMemo(() => {
    return {
      deff: computeMetricStats(measurements, 'deff'),
      beta: computeMetricStats(measurements, 'beta'),
      alignment: computeMetricStats(measurements, 'alignment'),
      cpair: computeMetricStats(measurements, 'cpair'),
    };
  }, [measurements]);

  /**
   * Get stats for a specific metric.
   */
  const getMetricStats = useCallback(
    (metric: MetricType): MetricStats => {
      return stats[metric];
    },
    [stats]
  );

  return {
    measurements,
    chartData,
    latest,
    isLive,
    loading,
    error,
    stats,
    getMetricStats,
    getMeasurementsByTurnRange,
    getBetaDistribution,
    clear,
    addMeasurements,
    pauseLiveUpdates,
    resumeLiveUpdates,
    isPaused,
    lastUpdateTime,
    updateCount,
  };
}

/**
 * Hook for a single metric with real-time updates.
 * Lighter weight alternative when you only need one metric.
 *
 * @param metric - The metric type to track
 * @param options - Hook configuration options
 * @returns Metric-specific data and statistics
 *
 * @example
 * ```tsx
 * function BetaChart() {
 *   const { values, stats, isLive } = useMetric('beta', { maxValues: 100 });
 *
 *   return (
 *     <div>
 *       <LineChart data={values.map((v, i) => ({ turn: i + 1, value: v }))} />
 *       <p>Average: {stats.average.toFixed(4)}</p>
 *     </div>
 *   );
 * }
 * ```
 */
export function useMetric(
  metric: MetricType,
  options: Omit<UseMeasurementsOptions, 'initialData'> = {}
): {
  values: number[];
  stats: MetricStats;
  latest: number | null;
  isLive: boolean;
  trend: 'up' | 'down' | 'stable';
} {
  const { measurements, stats, isLive } = useMeasurements(options);

  const values = useMemo(() => {
    return measurements.map((m) => m[metric]);
  }, [measurements, metric]);

  const metricStats = stats[metric];

  return {
    values,
    stats: metricStats,
    latest: metricStats.current,
    isLive,
    trend: metricStats.trend,
  };
}

/**
 * Hook for monitoring beta alerts in real-time.
 * Useful for dashboard components that need to highlight concerning values.
 *
 * @param options - Hook configuration options
 * @returns Beta-specific monitoring data
 */
export function useBetaMonitor(options: UseMeasurementsOptions = {}): {
  currentStatus: BetaStatus;
  recentAlerts: MeasurementEvent[];
  distribution: Record<BetaStatus, number>;
  hasAlerts: boolean;
  isHealthy: boolean;
} {
  const { measurements, latest } = useMeasurements(options);

  const currentStatus = useMemo((): BetaStatus => {
    if (!latest) return 'unknown';
    return latest.betaStatus || computeBetaStatus(latest.beta);
  }, [latest]);

  const recentAlerts = useMemo(() => {
    return measurements.filter((m) => {
      const status = m.betaStatus || computeBetaStatus(m.beta);
      return status === 'concerning' || status === 'critical';
    });
  }, [measurements]);

  const distribution = useMemo(() => {
    const dist: Record<BetaStatus, number> = {
      optimal: 0,
      monitor: 0,
      concerning: 0,
      critical: 0,
      unknown: 0,
    };

    for (const m of measurements) {
      const status = m.betaStatus || computeBetaStatus(m.beta);
      dist[status]++;
    }

    return dist;
  }, [measurements]);

  const hasAlerts = recentAlerts.length > 0;
  const isHealthy = currentStatus === 'optimal' || currentStatus === 'monitor';

  return {
    currentStatus,
    recentAlerts,
    distribution,
    hasAlerts,
    isHealthy,
  };
}

// Default export
export default useMeasurements;
