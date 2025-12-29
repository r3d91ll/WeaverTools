/**
 * MetricChart component - base chart for conveyance metric visualization.
 *
 * Provides a reusable line chart component for displaying time-series
 * measurement data with responsive sizing, interactive tooltips, and
 * real-time WebSocket updates.
 */
import { useState, useEffect, useCallback, useMemo } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  type TooltipProps,
} from 'recharts';
import type { MeasurementData, MeasurementEvent } from '@/types';
import { useMeasurements, type UseMeasurementsOptions } from '@/hooks/useMeasurements';

/** Supported metric types for visualization. */
export type MetricType = 'deff' | 'beta' | 'alignment' | 'cpair';

/** Metric configuration for display. */
export interface MetricConfig {
  /** Display name for the metric */
  name: string;
  /** Data key in measurement objects */
  dataKey: MetricType;
  /** Chart line color */
  color: string;
  /** Y-axis domain (min, max) - auto if not specified */
  domain?: [number, number];
  /** Unit suffix for display */
  unit?: string;
  /** Reference lines for thresholds */
  referenceLines?: Array<{
    value: number;
    label: string;
    color: string;
    dashed?: boolean;
  }>;
  /** Description for tooltip */
  description?: string;
}

/** Predefined metric configurations. */
export const METRIC_CONFIGS: Record<MetricType, MetricConfig> = {
  deff: {
    name: 'Effective Dimension',
    dataKey: 'deff',
    color: '#8B5CF6', // purple-500
    description: 'D_eff measures the effective dimensionality of semantic content',
  },
  beta: {
    name: 'Beta (Collapse)',
    dataKey: 'beta',
    color: '#EF4444', // red-500
    domain: [0, 5],
    referenceLines: [
      { value: 1.5, label: 'Optimal Min', color: '#22C55E', dashed: true },
      { value: 2.0, label: 'Optimal Max', color: '#22C55E', dashed: true },
      { value: 2.5, label: 'Monitor', color: '#EAB308', dashed: true },
      { value: 3.0, label: 'Critical', color: '#EF4444', dashed: true },
    ],
    description: 'Beta indicates dimensional collapse - lower is better (1.5-2.0 optimal)',
  },
  alignment: {
    name: 'Alignment',
    dataKey: 'alignment',
    color: '#3B82F6', // blue-500
    domain: [0, 1],
    unit: '%',
    description: 'Cosine similarity between sender and receiver hidden states',
  },
  cpair: {
    name: 'C_pair (Cross-pair)',
    dataKey: 'cpair',
    color: '#10B981', // emerald-500
    domain: [0, 1],
    description: 'Bilateral conveyance score between agent pair',
  },
};

/**
 * MetricChart component props.
 */
export interface MetricChartProps {
  /** Measurement data to display (not needed if using realTime mode) */
  data?: MeasurementData[] | MeasurementEvent[];
  /** Metric type to visualize */
  metric: MetricType;
  /** Chart height in pixels */
  height?: number;
  /** Whether to show the legend */
  showLegend?: boolean;
  /** Whether to show the grid */
  showGrid?: boolean;
  /** Whether to show reference lines (for metrics that have them) */
  showReferenceLines?: boolean;
  /** Custom chart title (overrides metric name) */
  title?: string;
  /** Additional CSS class names */
  className?: string;
  /** Whether data is currently loading */
  loading?: boolean;
  /** Custom empty state message */
  emptyMessage?: string;
  /** Callback when a data point is clicked */
  onDataPointClick?: (data: MeasurementData | MeasurementEvent) => void;
  /** X-axis data key (default: 'turn') */
  xAxisKey?: string;
  /** X-axis label */
  xAxisLabel?: string;
  /** Y-axis label */
  yAxisLabel?: string;
  /** Whether to animate transitions */
  animate?: boolean;
  /** Line stroke width */
  strokeWidth?: number;
  /** Whether to show dots on data points */
  showDots?: boolean;
  /** Enable real-time updates via WebSocket (default: false) */
  realTime?: boolean;
  /** Options for real-time measurement hook (only used if realTime is true) */
  realTimeOptions?: UseMeasurementsOptions;
  /** Show live indicator when receiving real-time updates */
  showLiveIndicator?: boolean;
  /** Callback when live status changes */
  onLiveStatusChange?: (isLive: boolean) => void;
  /** Callback when new measurement is received in real-time mode */
  onNewMeasurement?: (measurement: MeasurementEvent) => void;
  /** Whether to auto-pause when not visible (uses IntersectionObserver) */
  autoPauseWhenHidden?: boolean;
}

/**
 * Custom tooltip component for metric charts.
 */
const MetricTooltip: React.FC<
  TooltipProps<number, string> & {
    metric: MetricType;
    config: MetricConfig;
  }
> = ({ active, payload, label, config }) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const value = payload[0]?.value as number;
  const formattedValue = typeof value === 'number' ? value.toFixed(4) : 'N/A';

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3">
      <p className="text-sm font-medium text-gray-900">Turn {label}</p>
      <div className="mt-1 flex items-center gap-2">
        <span
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: config.color }}
        />
        <span className="text-sm text-gray-600">{config.name}:</span>
        <span className="text-sm font-medium text-gray-900">
          {formattedValue}
          {config.unit && ` ${config.unit}`}
        </span>
      </div>
      {config.description && (
        <p className="mt-2 text-xs text-gray-500 max-w-xs">{config.description}</p>
      )}
    </div>
  );
};

/**
 * Loading skeleton for chart.
 */
const ChartSkeleton: React.FC<{ height: number }> = ({ height }) => (
  <div
    className="animate-pulse bg-gray-100 rounded-lg flex items-center justify-center"
    style={{ height }}
  >
    <svg
      className="w-12 h-12 text-gray-300"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
      />
    </svg>
  </div>
);

/**
 * Empty state when no data is available.
 */
const EmptyState: React.FC<{ message: string; height: number }> = ({ message, height }) => (
  <div
    className="border border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center text-gray-400"
    style={{ height }}
  >
    <svg className="w-10 h-10 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
      />
    </svg>
    <p className="text-sm">{message}</p>
  </div>
);

/**
 * Live indicator showing real-time update status.
 */
interface LiveIndicatorProps {
  isLive: boolean;
  isPaused: boolean;
  updateCount?: number;
  onPauseToggle?: () => void;
}

const LiveIndicator: React.FC<LiveIndicatorProps> = ({
  isLive,
  isPaused,
  updateCount,
  onPauseToggle,
}) => {
  return (
    <div className="flex items-center gap-2">
      {/* Live status dot */}
      <div className="flex items-center gap-1.5">
        <span
          className={`
            w-2 h-2 rounded-full transition-colors duration-300
            ${isLive && !isPaused ? 'bg-green-500 animate-pulse' : ''}
            ${isPaused ? 'bg-yellow-500' : ''}
            ${!isLive && !isPaused ? 'bg-gray-300' : ''}
          `}
        />
        <span className="text-xs text-gray-500">
          {isPaused ? 'Paused' : isLive ? 'Live' : 'Idle'}
        </span>
      </div>

      {/* Update count */}
      {updateCount !== undefined && updateCount > 0 && (
        <span className="text-xs text-gray-400">
          {updateCount} update{updateCount !== 1 ? 's' : ''}
        </span>
      )}

      {/* Pause/resume button */}
      {onPauseToggle && (
        <button
          onClick={onPauseToggle}
          className="p-1 rounded hover:bg-gray-100 transition-colors"
          title={isPaused ? 'Resume updates' : 'Pause updates'}
          type="button"
        >
          {isPaused ? (
            <svg className="w-3.5 h-3.5 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
                clipRule="evenodd"
              />
            </svg>
          ) : (
            <svg className="w-3.5 h-3.5 text-gray-500" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z"
                clipRule="evenodd"
              />
            </svg>
          )}
        </button>
      )}
    </div>
  );
};

/**
 * MetricChart component for visualizing conveyance metrics.
 *
 * Supports both static data and real-time WebSocket updates.
 *
 * @example
 * ```tsx
 * // Static data mode
 * <MetricChart
 *   data={measurements}
 *   metric="deff"
 *   height={300}
 *   showReferenceLines
 * />
 *
 * // Real-time mode
 * <MetricChart
 *   metric="beta"
 *   realTime
 *   realTimeOptions={{ sessionId: 'session-123' }}
 *   showLiveIndicator
 * />
 * ```
 */
export const MetricChart: React.FC<MetricChartProps> = ({
  data: propData,
  metric,
  height = 300,
  showLegend = true,
  showGrid = true,
  showReferenceLines = true,
  title,
  className = '',
  loading: propLoading = false,
  emptyMessage,
  onDataPointClick,
  xAxisKey = 'turn',
  xAxisLabel,
  yAxisLabel,
  animate = true,
  strokeWidth = 2,
  showDots = true,
  realTime = false,
  realTimeOptions,
  showLiveIndicator = false,
  onLiveStatusChange,
  onNewMeasurement,
  autoPauseWhenHidden = false,
}) => {
  const config = METRIC_CONFIGS[metric];

  // Use real-time hook when realTime is enabled
  const {
    chartData: realTimeData,
    latest,
    isLive,
    isPaused,
    loading: realTimeLoading,
    updateCount,
    pauseLiveUpdates,
    resumeLiveUpdates,
  } = useMeasurements(realTime ? realTimeOptions : { autoConnect: false, subscribe: false });

  // Determine which data source to use
  const data = realTime ? realTimeData : propData;
  const loading = realTime ? realTimeLoading : propLoading;

  // Notify parent of live status changes
  useEffect(() => {
    if (realTime && onLiveStatusChange) {
      onLiveStatusChange(isLive);
    }
  }, [realTime, isLive, onLiveStatusChange]);

  // Notify parent of new measurements
  useEffect(() => {
    if (realTime && onNewMeasurement && latest) {
      onNewMeasurement(latest);
    }
  }, [realTime, latest, onNewMeasurement]);

  // Auto-pause when not visible
  const containerRef = useCallback(
    (node: HTMLDivElement | null) => {
      if (!autoPauseWhenHidden || !realTime || !node) return;

      const observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              resumeLiveUpdates();
            } else {
              pauseLiveUpdates();
            }
          });
        },
        { threshold: 0.1 }
      );

      observer.observe(node);
      return () => observer.disconnect();
    },
    [autoPauseWhenHidden, realTime, pauseLiveUpdates, resumeLiveUpdates]
  );

  // Handle pause toggle
  const handlePauseToggle = useCallback(() => {
    if (isPaused) {
      resumeLiveUpdates();
    } else {
      pauseLiveUpdates();
    }
  }, [isPaused, pauseLiveUpdates, resumeLiveUpdates]);

  // Handle loading state
  if (loading) {
    return (
      <div className={className} ref={containerRef}>
        {title && (
          <h3 className="text-sm font-medium text-gray-700 mb-2">{title}</h3>
        )}
        <ChartSkeleton height={height} />
      </div>
    );
  }

  // Handle empty data
  if (!data || data.length === 0) {
    const emptyStateMessage = realTime
      ? `Waiting for ${config.name} data...`
      : (emptyMessage ?? `No ${config.name} data available`);

    return (
      <div className={className} ref={containerRef}>
        <div className="flex items-center justify-between mb-2">
          {title && (
            <h3 className="text-sm font-medium text-gray-700">{title}</h3>
          )}
          {realTime && showLiveIndicator && (
            <LiveIndicator
              isLive={isLive}
              isPaused={isPaused}
              onPauseToggle={handlePauseToggle}
            />
          )}
        </div>
        <EmptyState message={emptyStateMessage} height={height} />
      </div>
    );
  }

  // Handle click on data point
  const handleClick = onDataPointClick
    ? (event: { payload?: MeasurementData | MeasurementEvent }) => {
        if (event.payload) {
          onDataPointClick(event.payload);
        }
      }
    : undefined;

  return (
    <div className={className} ref={containerRef}>
      <div className="flex items-center justify-between mb-2">
        {title && (
          <h3 className="text-sm font-medium text-gray-700">{title}</h3>
        )}
        {realTime && showLiveIndicator && (
          <LiveIndicator
            isLive={isLive}
            isPaused={isPaused}
            updateCount={updateCount}
            onPauseToggle={handlePauseToggle}
          />
        )}
      </div>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          onClick={handleClick}
        >
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />}
          <XAxis
            dataKey={xAxisKey}
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            label={
              xAxisLabel
                ? { value: xAxisLabel, position: 'bottom', offset: -5 }
                : undefined
            }
          />
          <YAxis
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            domain={config.domain ?? ['auto', 'auto']}
            tickFormatter={(value: number) =>
              typeof value === 'number' ? value.toFixed(2) : value
            }
            label={
              yAxisLabel
                ? { value: yAxisLabel, angle: -90, position: 'insideLeft' }
                : undefined
            }
          />
          <Tooltip
            content={<MetricTooltip metric={metric} config={config} />}
          />
          {showLegend && (
            <Legend
              verticalAlign="top"
              height={36}
              formatter={() => (
                <span className="text-sm text-gray-600">{config.name}</span>
              )}
            />
          )}
          {showReferenceLines &&
            config.referenceLines?.map((line) => (
              <ReferenceLine
                key={`ref-${line.value}`}
                y={line.value}
                stroke={line.color}
                strokeDasharray={line.dashed ? '5 5' : undefined}
                label={{
                  value: line.label,
                  position: 'right',
                  fill: line.color,
                  fontSize: 10,
                }}
              />
            ))}
          <Line
            type="monotone"
            dataKey={config.dataKey}
            stroke={config.color}
            strokeWidth={strokeWidth}
            dot={showDots ? { r: 3, fill: config.color } : false}
            activeDot={{ r: 5, fill: config.color }}
            isAnimationActive={animate}
            animationDuration={realTime ? 200 : 500}
            name={config.name}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

/**
 * RealTimeMetricChart is a convenience wrapper for MetricChart with realTime enabled.
 * Use this when you always want real-time updates.
 */
export const RealTimeMetricChart: React.FC<
  Omit<MetricChartProps, 'realTime' | 'data'> & {
    options?: UseMeasurementsOptions;
  }
> = ({ options, ...props }) => {
  return (
    <MetricChart
      {...props}
      realTime
      realTimeOptions={options}
      showLiveIndicator={props.showLiveIndicator ?? true}
    />
  );
};

export default MetricChart;
