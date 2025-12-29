/**
 * MetricsDashboard component - multi-metric comparison dashboard.
 *
 * Provides a comprehensive view of all conveyance metrics with:
 * - Summary cards for each metric
 * - Multi-line comparison chart
 * - Individual metric detail views
 * - Correlation analysis
 * - Real-time WebSocket updates
 */
import { useState, useMemo, useCallback } from 'react';
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
import { MetricSummaryGrid, type SummaryMetricType } from './MetricSummary';
import { METRIC_CONFIGS, type MetricType } from './MetricChart';

/** View modes for the dashboard */
export type DashboardViewMode = 'comparison' | 'correlation' | 'individual';

/** Metrics that can be displayed */
const ALL_METRICS: MetricType[] = ['deff', 'beta', 'alignment', 'cpair'];

/**
 * MetricsDashboard component props.
 */
export interface MetricsDashboardProps {
  /** Measurement data to display */
  data: Array<MeasurementData | MeasurementEvent>;
  /** Whether data is loading */
  loading?: boolean;
  /** Chart height in pixels */
  chartHeight?: number;
  /** Initial view mode */
  initialView?: DashboardViewMode;
  /** Initial selected metric for individual view */
  initialMetric?: MetricType;
  /** Additional CSS class names */
  className?: string;
  /** Whether to show view mode tabs */
  showViewTabs?: boolean;
  /** Whether to show metric summary cards */
  showSummaryCards?: boolean;
  /** Whether to allow metric selection */
  allowMetricSelection?: boolean;
  /** Callback when a data point is clicked */
  onDataPointClick?: (data: MeasurementData | MeasurementEvent) => void;
  /** Callback when selected metric changes */
  onMetricChange?: (metric: MetricType) => void;
}

/**
 * Loading skeleton for dashboard.
 */
const DashboardSkeleton: React.FC<{ height: number }> = ({ height }) => (
  <div className="space-y-4">
    <div className="grid grid-cols-4 gap-4">
      {[1, 2, 3, 4].map((i) => (
        <div key={i} className="card animate-pulse p-4">
          <div className="h-4 bg-gray-200 rounded w-1/2 mb-2" />
          <div className="h-8 bg-gray-200 rounded w-1/3 mb-2" />
          <div className="h-3 bg-gray-200 rounded w-2/3" />
        </div>
      ))}
    </div>
    <div
      className="animate-pulse bg-gray-100 rounded-lg"
      style={{ height }}
    />
  </div>
);

/**
 * Empty state for dashboard.
 */
const EmptyState: React.FC<{ height: number }> = ({ height }) => (
  <div
    className="border border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center text-gray-400"
    style={{ height }}
  >
    <svg className="w-16 h-16 mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
      />
    </svg>
    <p className="text-lg font-medium">No Metric Data</p>
    <p className="text-sm mt-1">Start a session to see measurements</p>
  </div>
);

/**
 * Custom tooltip for comparison chart.
 */
const ComparisonTooltip: React.FC<TooltipProps<number, string>> = ({
  active,
  payload,
  label,
}) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3">
      <p className="text-sm font-medium text-gray-900 mb-2">Turn {label}</p>
      <div className="space-y-1">
        {payload.map((entry) => {
          const config = METRIC_CONFIGS[entry.dataKey as MetricType];
          const value = entry.value as number;
          return (
            <div key={entry.dataKey} className="flex items-center gap-2">
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: entry.color }}
              />
              <span className="text-xs text-gray-600">{config?.name || entry.dataKey}:</span>
              <span className="text-xs font-medium text-gray-900">
                {typeof value === 'number' ? value.toFixed(4) : 'N/A'}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
};

/**
 * View mode tab buttons.
 */
const ViewModeTabs: React.FC<{
  currentView: DashboardViewMode;
  onViewChange: (view: DashboardViewMode) => void;
}> = ({ currentView, onViewChange }) => {
  const tabs: Array<{ id: DashboardViewMode; label: string; icon: React.ReactNode }> = [
    {
      id: 'comparison',
      label: 'Comparison',
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
        </svg>
      ),
    },
    {
      id: 'correlation',
      label: 'Correlation',
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
        </svg>
      ),
    },
    {
      id: 'individual',
      label: 'Individual',
      icon: (
        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
        </svg>
      ),
    },
  ];

  return (
    <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => onViewChange(tab.id)}
          className={`
            flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors
            ${currentView === tab.id
              ? 'bg-white text-gray-900 shadow-sm'
              : 'text-gray-600 hover:text-gray-900'
            }
          `}
        >
          {tab.icon}
          {tab.label}
        </button>
      ))}
    </div>
  );
};

/**
 * Metric toggle buttons for comparison view.
 */
const MetricToggles: React.FC<{
  selectedMetrics: MetricType[];
  onToggle: (metric: MetricType) => void;
}> = ({ selectedMetrics, onToggle }) => {
  return (
    <div className="flex items-center gap-2">
      {ALL_METRICS.map((metric) => {
        const config = METRIC_CONFIGS[metric];
        const isSelected = selectedMetrics.includes(metric);
        return (
          <button
            key={metric}
            onClick={() => onToggle(metric)}
            className={`
              flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium transition-all
              ${isSelected
                ? 'text-white shadow-sm'
                : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
              }
            `}
            style={isSelected ? { backgroundColor: config.color } : undefined}
          >
            {config.name}
          </button>
        );
      })}
    </div>
  );
};

/**
 * Comparison chart showing multiple metrics on same axes.
 */
const ComparisonChart: React.FC<{
  data: Array<MeasurementData | MeasurementEvent>;
  selectedMetrics: MetricType[];
  height: number;
  onDataPointClick?: (data: MeasurementData | MeasurementEvent) => void;
}> = ({ data, selectedMetrics, height, onDataPointClick }) => {
  const handleClick = onDataPointClick
    ? (event: { activePayload?: Array<{ payload: MeasurementData | MeasurementEvent }> }) => {
        if (event.activePayload?.[0]?.payload) {
          onDataPointClick(event.activePayload[0].payload);
        }
      }
    : undefined;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart
        data={data}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
        onClick={handleClick}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
        <XAxis
          dataKey="turn"
          stroke="#6B7280"
          fontSize={12}
          tickLine={false}
          axisLine={{ stroke: '#E5E7EB' }}
        />
        <YAxis
          stroke="#6B7280"
          fontSize={12}
          tickLine={false}
          axisLine={{ stroke: '#E5E7EB' }}
          domain={['auto', 'auto']}
          tickFormatter={(value: number) => value.toFixed(2)}
        />
        <Tooltip content={<ComparisonTooltip />} />
        <Legend
          verticalAlign="top"
          height={36}
          formatter={(value) => (
            <span className="text-sm text-gray-600">
              {METRIC_CONFIGS[value as MetricType]?.name || value}
            </span>
          )}
        />
        {/* Beta reference lines (only if beta is selected) */}
        {selectedMetrics.includes('beta') && (
          <>
            <ReferenceLine
              y={1.5}
              stroke="#22C55E"
              strokeDasharray="5 5"
              label={{ value: 'Optimal', position: 'right', fill: '#22C55E', fontSize: 10 }}
            />
            <ReferenceLine
              y={2.0}
              stroke="#22C55E"
              strokeDasharray="5 5"
            />
            <ReferenceLine
              y={3.0}
              stroke="#EF4444"
              strokeDasharray="5 5"
              label={{ value: 'Critical', position: 'right', fill: '#EF4444', fontSize: 10 }}
            />
          </>
        )}
        {selectedMetrics.map((metric) => {
          const config = METRIC_CONFIGS[metric];
          return (
            <Line
              key={metric}
              type="monotone"
              dataKey={metric}
              stroke={config.color}
              strokeWidth={2}
              dot={{ r: 3, fill: config.color }}
              activeDot={{ r: 5, fill: config.color }}
              name={metric}
            />
          );
        })}
      </LineChart>
    </ResponsiveContainer>
  );
};

/**
 * Correlation matrix showing relationships between metrics.
 */
const CorrelationMatrix: React.FC<{
  data: Array<MeasurementData | MeasurementEvent>;
}> = ({ data }) => {
  // Calculate Pearson correlation coefficients
  const correlations = useMemo(() => {
    if (data.length < 3) return null;

    const metrics = ALL_METRICS;
    const matrix: Record<string, Record<string, number>> = {};

    for (const m1 of metrics) {
      matrix[m1] = {};
      for (const m2 of metrics) {
        if (m1 === m2) {
          matrix[m1][m2] = 1;
        } else if (matrix[m2]?.[m1] !== undefined) {
          matrix[m1][m2] = matrix[m2][m1];
        } else {
          // Calculate Pearson correlation
          const values1 = data.map((d) => d[m1]);
          const values2 = data.map((d) => d[m2]);
          const n = values1.length;
          const mean1 = values1.reduce((a, b) => a + b, 0) / n;
          const mean2 = values2.reduce((a, b) => a + b, 0) / n;

          let num = 0;
          let den1 = 0;
          let den2 = 0;

          for (let i = 0; i < n; i++) {
            const diff1 = values1[i] - mean1;
            const diff2 = values2[i] - mean2;
            num += diff1 * diff2;
            den1 += diff1 * diff1;
            den2 += diff2 * diff2;
          }

          const corr = den1 > 0 && den2 > 0 ? num / Math.sqrt(den1 * den2) : 0;
          matrix[m1][m2] = corr;
        }
      }
    }

    return matrix;
  }, [data]);

  if (!correlations) {
    return (
      <div className="text-center text-gray-500 py-8">
        <p>Need at least 3 data points for correlation analysis</p>
      </div>
    );
  }

  const getCorrelationColor = (value: number): string => {
    const absValue = Math.abs(value);
    if (absValue > 0.7) return value > 0 ? 'bg-green-500' : 'bg-red-500';
    if (absValue > 0.4) return value > 0 ? 'bg-green-300' : 'bg-red-300';
    if (absValue > 0.2) return value > 0 ? 'bg-green-100' : 'bg-red-100';
    return 'bg-gray-100';
  };

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full">
        <thead>
          <tr>
            <th className="p-2"></th>
            {ALL_METRICS.map((m) => (
              <th key={m} className="p-2 text-xs font-medium text-gray-500">
                {METRIC_CONFIGS[m].name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {ALL_METRICS.map((m1) => (
            <tr key={m1}>
              <td className="p-2 text-xs font-medium text-gray-500">
                {METRIC_CONFIGS[m1].name}
              </td>
              {ALL_METRICS.map((m2) => {
                const value = correlations[m1][m2];
                return (
                  <td key={m2} className="p-2">
                    <div
                      className={`
                        w-full h-12 flex items-center justify-center rounded
                        ${getCorrelationColor(value)}
                        ${m1 === m2 ? 'opacity-50' : ''}
                      `}
                    >
                      <span className={`text-sm font-medium ${Math.abs(value) > 0.4 ? 'text-white' : 'text-gray-700'}`}>
                        {value.toFixed(2)}
                      </span>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4 flex items-center justify-center gap-4 text-xs text-gray-500">
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-green-500 rounded" />
          <span>Strong positive</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-red-500 rounded" />
          <span>Strong negative</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-4 h-4 bg-gray-100 rounded" />
          <span>Weak/none</span>
        </div>
      </div>
    </div>
  );
};

/**
 * Individual metric detail view.
 */
const IndividualMetricView: React.FC<{
  data: Array<MeasurementData | MeasurementEvent>;
  metric: MetricType;
  height: number;
  onDataPointClick?: (data: MeasurementData | MeasurementEvent) => void;
}> = ({ data, metric, height, onDataPointClick }) => {
  const config = METRIC_CONFIGS[metric];

  const handleClick = onDataPointClick
    ? (event: { activePayload?: Array<{ payload: MeasurementData | MeasurementEvent }> }) => {
        if (event.activePayload?.[0]?.payload) {
          onDataPointClick(event.activePayload[0].payload);
        }
      }
    : undefined;

  // Calculate stats
  const stats = useMemo(() => {
    if (data.length === 0) return null;
    const values = data.map((d) => d[metric]);
    const sum = values.reduce((a, b) => a + b, 0);
    return {
      avg: sum / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      current: values[values.length - 1],
      stdDev: Math.sqrt(
        values.reduce((acc, v) => acc + Math.pow(v - sum / values.length, 2), 0) / values.length
      ),
    };
  }, [data, metric]);

  return (
    <div>
      {/* Stats bar */}
      {stats && (
        <div className="flex items-center gap-6 mb-4 p-3 bg-gray-50 rounded-lg">
          <div>
            <span className="text-xs text-gray-500">Current</span>
            <p className="text-lg font-bold" style={{ color: config.color }}>
              {stats.current.toFixed(4)}
            </p>
          </div>
          <div>
            <span className="text-xs text-gray-500">Average</span>
            <p className="text-sm font-medium text-gray-700">{stats.avg.toFixed(4)}</p>
          </div>
          <div>
            <span className="text-xs text-gray-500">Min</span>
            <p className="text-sm font-medium text-gray-700">{stats.min.toFixed(4)}</p>
          </div>
          <div>
            <span className="text-xs text-gray-500">Max</span>
            <p className="text-sm font-medium text-gray-700">{stats.max.toFixed(4)}</p>
          </div>
          <div>
            <span className="text-xs text-gray-500">Std Dev</span>
            <p className="text-sm font-medium text-gray-700">{stats.stdDev.toFixed(4)}</p>
          </div>
        </div>
      )}

      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={data}
          margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
          onClick={handleClick}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />
          <XAxis
            dataKey="turn"
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
          />
          <YAxis
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            domain={config.domain ?? ['auto', 'auto']}
            tickFormatter={(value: number) => value.toFixed(2)}
          />
          <Tooltip
            formatter={(value: number) => [value.toFixed(4), config.name]}
            labelFormatter={(label) => `Turn ${label}`}
          />
          {/* Reference lines for the metric */}
          {config.referenceLines?.map((line) => (
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
          {/* Average reference line */}
          {stats && (
            <ReferenceLine
              y={stats.avg}
              stroke={config.color}
              strokeDasharray="5 5"
              strokeOpacity={0.5}
              label={{
                value: `Avg: ${stats.avg.toFixed(2)}`,
                position: 'right',
                fill: config.color,
                fontSize: 10,
              }}
            />
          )}
          <Line
            type="monotone"
            dataKey={metric}
            stroke={config.color}
            strokeWidth={2}
            dot={{ r: 3, fill: config.color }}
            activeDot={{ r: 5, fill: config.color }}
            name={config.name}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

/**
 * MetricsDashboard component provides a comprehensive multi-metric view.
 *
 * @example
 * ```tsx
 * <MetricsDashboard
 *   data={measurements}
 *   showViewTabs
 *   showSummaryCards
 *   onMetricChange={(metric) => console.log('Selected:', metric)}
 * />
 * ```
 */
export const MetricsDashboard: React.FC<MetricsDashboardProps> = ({
  data,
  loading = false,
  chartHeight = 300,
  initialView = 'comparison',
  initialMetric = 'deff',
  className = '',
  showViewTabs = true,
  showSummaryCards = true,
  allowMetricSelection = true,
  onDataPointClick,
  onMetricChange,
}) => {
  const [viewMode, setViewMode] = useState<DashboardViewMode>(initialView);
  const [selectedMetrics, setSelectedMetrics] = useState<MetricType[]>(ALL_METRICS);
  const [focusedMetric, setFocusedMetric] = useState<MetricType>(initialMetric);

  // Handle metric toggle
  const handleMetricToggle = useCallback((metric: MetricType) => {
    setSelectedMetrics((prev) => {
      if (prev.includes(metric)) {
        // Don't allow deselecting all metrics
        if (prev.length === 1) return prev;
        return prev.filter((m) => m !== metric);
      }
      return [...prev, metric];
    });
  }, []);

  // Handle metric selection from summary card
  const handleMetricSelect = useCallback((metric: SummaryMetricType) => {
    setFocusedMetric(metric);
    setViewMode('individual');
    onMetricChange?.(metric);
  }, [onMetricChange]);

  // Handle loading state
  if (loading) {
    return (
      <div className={className}>
        <DashboardSkeleton height={chartHeight} />
      </div>
    );
  }

  // Handle empty data
  if (!data || data.length === 0) {
    return (
      <div className={className}>
        {showSummaryCards && (
          <MetricSummaryGrid
            data={[]}
            loading={false}
            showSparklines
            className="mb-6"
          />
        )}
        <EmptyState height={chartHeight} />
      </div>
    );
  }

  return (
    <div className={className}>
      {/* Summary Cards */}
      {showSummaryCards && (
        <MetricSummaryGrid
          data={data}
          loading={loading}
          showSparklines
          selectedMetric={viewMode === 'individual' ? focusedMetric : undefined}
          onSelectMetric={allowMetricSelection ? handleMetricSelect : undefined}
          className="mb-6"
        />
      )}

      {/* Chart Section */}
      <div className="card">
        {/* Header with tabs and controls */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">
            {viewMode === 'comparison' && 'Metric Comparison'}
            {viewMode === 'correlation' && 'Metric Correlation'}
            {viewMode === 'individual' && METRIC_CONFIGS[focusedMetric].name}
          </h2>
          <div className="flex items-center gap-4">
            {viewMode === 'comparison' && (
              <MetricToggles
                selectedMetrics={selectedMetrics}
                onToggle={handleMetricToggle}
              />
            )}
            {viewMode === 'individual' && (
              <div className="flex items-center gap-2">
                {ALL_METRICS.map((m) => (
                  <button
                    key={m}
                    onClick={() => {
                      setFocusedMetric(m);
                      onMetricChange?.(m);
                    }}
                    className={`
                      px-2 py-1 rounded text-xs font-medium transition-colors
                      ${focusedMetric === m
                        ? 'text-white'
                        : 'bg-gray-100 text-gray-500 hover:bg-gray-200'
                      }
                    `}
                    style={focusedMetric === m ? { backgroundColor: METRIC_CONFIGS[m].color } : undefined}
                  >
                    {METRIC_CONFIGS[m].name}
                  </button>
                ))}
              </div>
            )}
            {showViewTabs && (
              <ViewModeTabs currentView={viewMode} onViewChange={setViewMode} />
            )}
          </div>
        </div>

        {/* Chart Content */}
        {viewMode === 'comparison' && (
          <ComparisonChart
            data={data}
            selectedMetrics={selectedMetrics}
            height={chartHeight}
            onDataPointClick={onDataPointClick}
          />
        )}
        {viewMode === 'correlation' && (
          <CorrelationMatrix data={data} />
        )}
        {viewMode === 'individual' && (
          <IndividualMetricView
            data={data}
            metric={focusedMetric}
            height={chartHeight}
            onDataPointClick={onDataPointClick}
          />
        )}

        {/* Data point count */}
        <p className="text-xs text-gray-400 mt-4 text-right">
          {data.length} data point{data.length !== 1 ? 's' : ''}
        </p>
      </div>
    </div>
  );
};

export default MetricsDashboard;
