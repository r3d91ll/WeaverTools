/**
 * MetricSummary component - displays summary statistics for a single metric.
 *
 * Shows current value, average, trend indicator, and status visualization.
 * Used in the metrics dashboard for at-a-glance metric overview.
 */
import { useMemo } from 'react';
import { computeBetaStatus, type BetaStatus, type MeasurementData, type MeasurementEvent } from '@/types';

/** Metric types supported for summary display */
export type SummaryMetricType = 'deff' | 'beta' | 'alignment' | 'cpair';

/** Metric configuration for summary display */
interface MetricSummaryConfig {
  name: string;
  shortName: string;
  description: string;
  color: string;
  bgColor: string;
  format: (v: number) => string;
  getStatusColor?: (value: number) => string;
  domain?: [number, number];
}

/** Configuration for each metric type */
const METRIC_SUMMARY_CONFIGS: Record<SummaryMetricType, MetricSummaryConfig> = {
  deff: {
    name: 'Effective Dimension',
    shortName: 'D_eff',
    description: 'Effective dimensionality of semantic content',
    color: '#8B5CF6', // purple-500
    bgColor: 'bg-purple-50',
    format: (v) => v.toFixed(3),
  },
  beta: {
    name: 'Beta (Collapse)',
    shortName: 'Beta',
    description: 'Collapse indicator (1.5-2.0 optimal)',
    color: '#EF4444', // red-500
    bgColor: 'bg-red-50',
    format: (v) => v.toFixed(3),
    domain: [0, 5],
    getStatusColor: (value: number) => {
      const status = computeBetaStatus(value);
      switch (status) {
        case 'optimal':
          return 'text-green-600';
        case 'monitor':
          return 'text-yellow-600';
        case 'concerning':
          return 'text-orange-600';
        case 'critical':
          return 'text-red-600';
        default:
          return 'text-gray-500';
      }
    },
  },
  alignment: {
    name: 'Alignment',
    shortName: 'Alignment',
    description: 'Cosine similarity between hidden states',
    color: '#3B82F6', // blue-500
    bgColor: 'bg-blue-50',
    format: (v) => `${(v * 100).toFixed(1)}%`,
    domain: [0, 1],
  },
  cpair: {
    name: 'Cross-pair',
    shortName: 'C_pair',
    description: 'Bilateral conveyance score',
    color: '#10B981', // emerald-500
    bgColor: 'bg-emerald-50',
    format: (v) => v.toFixed(3),
    domain: [0, 1],
  },
};

/** Calculate summary statistics for a metric */
interface MetricStats {
  current: number | null;
  average: number | null;
  min: number | null;
  max: number | null;
  trend: 'up' | 'down' | 'stable' | null;
  count: number;
}

function calculateMetricStats(
  data: Array<MeasurementData | MeasurementEvent>,
  metric: SummaryMetricType
): MetricStats {
  if (!data || data.length === 0) {
    return {
      current: null,
      average: null,
      min: null,
      max: null,
      trend: null,
      count: 0,
    };
  }

  const values = data.map((d) => d[metric]);
  const current = values[values.length - 1];
  const sum = values.reduce((a, b) => a + b, 0);
  const avg = sum / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);

  // Calculate trend based on last 3 points
  let trend: 'up' | 'down' | 'stable' | null = null;
  if (values.length >= 3) {
    const recent = values.slice(-3);
    const diff = recent[2] - recent[0];
    const threshold = avg * 0.05; // 5% of average as threshold
    if (diff > threshold) {
      trend = 'up';
    } else if (diff < -threshold) {
      trend = 'down';
    } else {
      trend = 'stable';
    }
  }

  return {
    current,
    average: avg,
    min,
    max,
    trend,
    count: data.length,
  };
}

/**
 * MetricSummary component props.
 */
export interface MetricSummaryProps {
  /** Measurement data to calculate summary from */
  data: Array<MeasurementData | MeasurementEvent>;
  /** Metric type to display */
  metric: SummaryMetricType;
  /** Whether data is loading */
  loading?: boolean;
  /** Show mini sparkline (if true) */
  showSparkline?: boolean;
  /** Additional CSS class names */
  className?: string;
  /** Compact display mode */
  compact?: boolean;
  /** Callback when card is clicked */
  onClick?: () => void;
  /** Whether this card is currently selected/active */
  selected?: boolean;
}

/**
 * Loading skeleton for metric summary.
 */
const LoadingSkeleton: React.FC<{ compact?: boolean }> = ({ compact }) => (
  <div className={`card animate-pulse ${compact ? 'p-3' : 'p-4'}`}>
    <div className="h-4 bg-gray-200 rounded w-1/2 mb-2" />
    <div className={`bg-gray-200 rounded w-1/3 ${compact ? 'h-6 mb-1' : 'h-8 mb-2'}`} />
    <div className="h-3 bg-gray-200 rounded w-2/3" />
  </div>
);

/**
 * Mini sparkline for showing trend.
 */
const Sparkline: React.FC<{
  data: number[];
  color: string;
  height?: number;
}> = ({ data, color, height = 24 }) => {
  if (data.length < 2) return null;

  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const width = 60;
  const step = width / (data.length - 1);

  const points = data
    .map((v, i) => {
      const x = i * step;
      const y = height - ((v - min) / range) * height;
      return `${x},${y}`;
    })
    .join(' ');

  return (
    <svg width={width} height={height} className="overflow-visible">
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
};

/**
 * Trend indicator arrow.
 */
const TrendIndicator: React.FC<{ trend: 'up' | 'down' | 'stable' | null; metric: SummaryMetricType }> = ({
  trend,
  metric,
}) => {
  if (!trend) return null;

  // For beta, down is good; for others, up is generally better
  const isPositive = metric === 'beta' ? trend === 'down' : trend === 'up';

  if (trend === 'stable') {
    return (
      <span className="text-gray-400 text-xs">
        <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14" />
        </svg>
      </span>
    );
  }

  return (
    <span className={isPositive ? 'text-green-500' : 'text-red-500'}>
      {trend === 'up' ? (
        <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 10l7-7m0 0l7 7m-7-7v18" />
        </svg>
      ) : (
        <svg className="w-4 h-4 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      )}
    </span>
  );
};

/**
 * Beta status badge.
 */
const BetaStatusBadge: React.FC<{ value: number }> = ({ value }) => {
  const status = computeBetaStatus(value);

  const statusConfig: Record<BetaStatus, { label: string; className: string }> = {
    optimal: { label: 'Optimal', className: 'bg-green-100 text-green-700' },
    monitor: { label: 'Monitor', className: 'bg-yellow-100 text-yellow-700' },
    concerning: { label: 'Concerning', className: 'bg-orange-100 text-orange-700' },
    critical: { label: 'Critical', className: 'bg-red-100 text-red-700' },
    unknown: { label: 'Unknown', className: 'bg-gray-100 text-gray-700' },
  };

  const config = statusConfig[status];

  return (
    <span className={`text-xs px-2 py-0.5 rounded-full ${config.className}`}>
      {config.label}
    </span>
  );
};

/**
 * MetricSummary component displays summary statistics for a single metric.
 *
 * @example
 * ```tsx
 * <MetricSummary
 *   data={measurements}
 *   metric="beta"
 *   showSparkline
 * />
 * ```
 */
export const MetricSummary: React.FC<MetricSummaryProps> = ({
  data,
  metric,
  loading = false,
  showSparkline = false,
  className = '',
  compact = false,
  onClick,
  selected = false,
}) => {
  const config = METRIC_SUMMARY_CONFIGS[metric];
  const stats = useMemo(() => calculateMetricStats(data, metric), [data, metric]);

  if (loading) {
    return <LoadingSkeleton compact={compact} />;
  }

  const valueColor = config.getStatusColor
    ? (stats.current !== null ? config.getStatusColor(stats.current) : 'text-gray-400')
    : 'text-gray-900';

  const sparklineData = showSparkline && data.length > 1
    ? data.slice(-10).map((d) => d[metric])
    : [];

  return (
    <div
      className={`
        card transition-all duration-200
        ${compact ? 'p-3' : 'p-4'}
        ${onClick ? 'cursor-pointer hover:shadow-md' : ''}
        ${selected ? 'ring-2 ring-weaver-500 ring-offset-2' : ''}
        ${className}
      `}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={onClick ? (e) => { if (e.key === 'Enter' || e.key === ' ') onClick(); } : undefined}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span
            className="w-2 h-2 rounded-full"
            style={{ backgroundColor: config.color }}
          />
          <h3 className={`font-medium text-gray-500 ${compact ? 'text-xs' : 'text-sm'}`}>
            {config.shortName}
          </h3>
        </div>
        {showSparkline && sparklineData.length > 1 && (
          <Sparkline data={sparklineData} color={config.color} height={compact ? 16 : 24} />
        )}
      </div>

      {/* Current value */}
      <div className="flex items-baseline gap-2">
        <p className={`font-bold ${valueColor} ${compact ? 'text-xl' : 'text-2xl'}`}>
          {stats.current !== null ? config.format(stats.current) : '--'}
        </p>
        <TrendIndicator trend={stats.trend} metric={metric} />
        {metric === 'beta' && stats.current !== null && (
          <BetaStatusBadge value={stats.current} />
        )}
      </div>

      {/* Stats row */}
      <div className={`flex items-center gap-3 text-gray-400 ${compact ? 'text-xs mt-1' : 'text-xs mt-2'}`}>
        {stats.count > 0 ? (
          <>
            <span>
              Avg: <span className="text-gray-600">{stats.average !== null ? config.format(stats.average) : '--'}</span>
            </span>
            {!compact && (
              <>
                <span>
                  Min: <span className="text-gray-600">{stats.min !== null ? config.format(stats.min) : '--'}</span>
                </span>
                <span>
                  Max: <span className="text-gray-600">{stats.max !== null ? config.format(stats.max) : '--'}</span>
                </span>
              </>
            )}
            <span className="text-gray-400">({stats.count} pts)</span>
          </>
        ) : (
          <span>No data</span>
        )}
      </div>

      {/* Description (non-compact only) */}
      {!compact && (
        <p className="text-xs text-gray-400 mt-2">{config.description}</p>
      )}
    </div>
  );
};

/**
 * MetricSummaryGrid displays all four metrics in a responsive grid.
 */
export interface MetricSummaryGridProps {
  /** Measurement data to calculate summaries from */
  data: Array<MeasurementData | MeasurementEvent>;
  /** Whether data is loading */
  loading?: boolean;
  /** Show sparklines in summaries */
  showSparklines?: boolean;
  /** Compact display mode */
  compact?: boolean;
  /** Currently selected metric */
  selectedMetric?: SummaryMetricType | null;
  /** Callback when a metric is selected */
  onSelectMetric?: (metric: SummaryMetricType) => void;
  /** Additional CSS class names */
  className?: string;
}

/**
 * MetricSummaryGrid component displays all four metric summaries in a grid.
 *
 * @example
 * ```tsx
 * <MetricSummaryGrid
 *   data={measurements}
 *   showSparklines
 *   onSelectMetric={setSelectedMetric}
 * />
 * ```
 */
export const MetricSummaryGrid: React.FC<MetricSummaryGridProps> = ({
  data,
  loading = false,
  showSparklines = false,
  compact = false,
  selectedMetric,
  onSelectMetric,
  className = '',
}) => {
  const metrics: SummaryMetricType[] = ['deff', 'beta', 'alignment', 'cpair'];

  return (
    <div className={`grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 ${className}`}>
      {metrics.map((metric) => (
        <MetricSummary
          key={metric}
          data={data}
          metric={metric}
          loading={loading}
          showSparkline={showSparklines}
          compact={compact}
          selected={selectedMetric === metric}
          onClick={onSelectMetric ? () => onSelectMetric(metric) : undefined}
        />
      ))}
    </div>
  );
};

export default MetricSummary;
