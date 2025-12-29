/**
 * MemoryChart component - specialized chart for system memory visualization.
 *
 * Displays system memory and CPU usage percentage over time with
 * real-time updates, thresholds, and interactive tooltips.
 */
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ComposedChart,
  Line,
  type TooltipProps,
} from 'recharts';
import type { ResourceDataPoint } from './ResourceDashboard';

/**
 * Memory chart mode - which metric to display.
 */
export type MemoryChartMode = 'memory' | 'cpu' | 'combined';

/**
 * MemoryChart component props.
 */
export interface MemoryChartProps {
  /** Resource data points to display */
  data: ResourceDataPoint[];
  /** Chart height in pixels */
  height?: number;
  /** Chart mode - memory, cpu, or combined */
  mode?: MemoryChartMode;
  /** Whether to show the area fill under the line */
  showArea?: boolean;
  /** Whether to show the grid */
  showGrid?: boolean;
  /** Whether data is loading */
  loading?: boolean;
  /** Additional CSS class names */
  className?: string;
  /** Whether to show threshold reference lines */
  showThresholds?: boolean;
  /** Callback when a data point is clicked */
  onDataPointClick?: (data: ResourceDataPoint) => void;
}

/**
 * Chart color configurations.
 */
const CHART_COLORS = {
  memory: {
    stroke: '#8B5CF6', // purple-500
    fill: '#8B5CF6',
    name: 'System Memory',
  },
  cpu: {
    stroke: '#F59E0B', // amber-500
    fill: '#F59E0B',
    name: 'CPU Usage',
  },
} as const;

/**
 * Threshold configurations for reference lines.
 */
const THRESHOLDS = [
  { value: 75, label: 'Warning', color: '#EAB308', dashed: true },
  { value: 90, label: 'Critical', color: '#EF4444', dashed: true },
];

/**
 * Calculate statistics from the data.
 */
function calculateStats(
  data: ResourceDataPoint[],
  key: 'memoryPercent' | 'cpuPercent'
): { avg: number; max: number; min: number; current: number } {
  if (!data || data.length === 0) {
    return { avg: 0, max: 0, min: 0, current: 0 };
  }

  const values = data.map((d) => d[key] ?? 0);
  const sum = values.reduce((acc, v) => acc + v, 0);

  return {
    avg: sum / values.length,
    max: Math.max(...values),
    min: Math.min(...values),
    current: values[values.length - 1] ?? 0,
  };
}

/**
 * Format percentage value.
 */
function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

/**
 * Get status color based on utilization level.
 */
function getStatusColor(percent: number): string {
  if (percent >= 90) return 'text-red-600';
  if (percent >= 75) return 'text-yellow-600';
  return 'text-green-600';
}

/**
 * Custom tooltip for Memory chart.
 */
interface MemoryTooltipProps extends TooltipProps<number, string> {
  mode: MemoryChartMode;
}

const MemoryTooltip: React.FC<MemoryTooltipProps> = ({
  active,
  payload,
  mode,
}) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const data = payload[0]?.payload as ResourceDataPoint & { timeLabel?: string };
  const timeLabel = data.timeLabel ?? new Date(data.timestamp).toLocaleTimeString();

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 max-w-xs">
      <p className="text-sm font-medium text-gray-900">{timeLabel}</p>
      <div className="mt-2 space-y-1">
        {(mode === 'memory' || mode === 'combined') && data.memoryPercent !== undefined && (
          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CHART_COLORS.memory.stroke }}
            />
            <span className="text-sm text-gray-600">Memory:</span>
            <span className={`text-sm font-medium ${getStatusColor(data.memoryPercent)}`}>
              {formatPercent(data.memoryPercent)}
            </span>
          </div>
        )}
        {(mode === 'cpu' || mode === 'combined') && data.cpuPercent !== undefined && (
          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CHART_COLORS.cpu.stroke }}
            />
            <span className="text-sm text-gray-600">CPU:</span>
            <span className={`text-sm font-medium ${getStatusColor(data.cpuPercent)}`}>
              {formatPercent(data.cpuPercent)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Loading skeleton for Memory chart.
 */
const LoadingSkeleton: React.FC<{ height: number }> = ({ height }) => (
  <div
    className="animate-pulse bg-gray-100 rounded-lg flex items-center justify-center"
    style={{ height }}
  >
    <div className="text-gray-300">
      <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"
        />
      </svg>
    </div>
  </div>
);

/**
 * Empty state when no data is available.
 */
const EmptyState: React.FC<{ height: number; message: string }> = ({ height, message }) => (
  <div
    className="border border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center text-gray-400"
    style={{ height }}
  >
    <svg className="w-10 h-10 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"
      />
    </svg>
    <p className="text-sm">{message}</p>
    <p className="text-xs mt-1">Connect to backend to see system metrics</p>
  </div>
);

/**
 * MemoryChart component for visualizing system memory and CPU usage.
 *
 * @example
 * ```tsx
 * <MemoryChart
 *   data={resourceHistory}
 *   mode="combined"
 *   height={200}
 *   showThresholds
 * />
 * ```
 */
export const MemoryChart: React.FC<MemoryChartProps> = ({
  data,
  height = 200,
  mode = 'memory',
  showArea = true,
  showGrid = true,
  loading = false,
  className = '',
  showThresholds = true,
  onDataPointClick,
}) => {
  // Handle loading state
  if (loading) {
    return (
      <div className={className}>
        <LoadingSkeleton height={height} />
      </div>
    );
  }

  // Handle empty data
  if (!data || data.length === 0) {
    const emptyMessage =
      mode === 'memory'
        ? 'No system memory data available'
        : mode === 'cpu'
        ? 'No CPU usage data available'
        : 'No system data available';

    return (
      <div className={className}>
        <EmptyState height={height} message={emptyMessage} />
      </div>
    );
  }

  // Calculate stats based on mode
  const memoryStats = calculateStats(data, 'memoryPercent');
  const cpuStats = calculateStats(data, 'cpuPercent');

  // Prepare chart data with time labels
  const chartData = data.map((point, index) => ({
    ...point,
    time: index,
    timeLabel: new Date(point.timestamp).toLocaleTimeString(),
  }));

  // Handle click on data point
  const handleClick = onDataPointClick
    ? (event: { activePayload?: Array<{ payload: ResourceDataPoint }> }) => {
        if (event.activePayload?.[0]?.payload) {
          onDataPointClick(event.activePayload[0].payload);
        }
      }
    : undefined;

  const ChartComponent = mode === 'combined' ? ComposedChart : AreaChart;

  // Check if we have system memory/cpu data
  const hasMemoryData = data.some((d) => d.memoryPercent !== undefined);
  const hasCpuData = data.some((d) => d.cpuPercent !== undefined);

  return (
    <div className={className}>
      {/* Stats bar */}
      <div className="flex items-center gap-4 mb-2 text-xs flex-wrap">
        {(mode === 'memory' || mode === 'combined') && hasMemoryData && (
          <>
            <div className="flex items-center gap-1">
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: CHART_COLORS.memory.stroke }}
              />
              <span className="text-gray-500">Memory:</span>
              <span className={`font-medium ${getStatusColor(memoryStats.current)}`}>
                {formatPercent(memoryStats.current)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-gray-500">Avg:</span>
              <span className="font-medium text-gray-700">{formatPercent(memoryStats.avg)}</span>
            </div>
          </>
        )}
        {(mode === 'cpu' || mode === 'combined') && hasCpuData && (
          <>
            <div className="flex items-center gap-1">
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: CHART_COLORS.cpu.stroke }}
              />
              <span className="text-gray-500">CPU:</span>
              <span className={`font-medium ${getStatusColor(cpuStats.current)}`}>
                {formatPercent(cpuStats.current)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-gray-500">Avg:</span>
              <span className="font-medium text-gray-700">{formatPercent(cpuStats.avg)}</span>
            </div>
          </>
        )}
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Points:</span>
          <span className="font-medium text-gray-700">{data.length}</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <ChartComponent
          data={chartData}
          margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
          onClick={handleClick}
        >
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />}
          <XAxis
            dataKey="time"
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
          />
          <YAxis
            domain={[0, 100]}
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            tickFormatter={(value: number) => `${value}%`}
          />
          <Tooltip content={<MemoryTooltip mode={mode} />} />

          {/* Threshold reference lines */}
          {showThresholds &&
            THRESHOLDS.map((threshold) => (
              <ReferenceLine
                key={`threshold-${threshold.value}`}
                y={threshold.value}
                stroke={threshold.color}
                strokeDasharray={threshold.dashed ? '5 5' : undefined}
                label={{
                  value: threshold.label,
                  position: 'right',
                  fill: threshold.color,
                  fontSize: 10,
                }}
              />
            ))}

          {/* Memory area/line */}
          {(mode === 'memory' || mode === 'combined') && hasMemoryData && (
            <>
              {showArea && mode !== 'combined' && (
                <Area
                  type="monotone"
                  dataKey="memoryPercent"
                  stroke={CHART_COLORS.memory.stroke}
                  fill={CHART_COLORS.memory.fill}
                  fillOpacity={0.2}
                  strokeWidth={2}
                  name={CHART_COLORS.memory.name}
                />
              )}
              {mode === 'combined' && (
                <Line
                  type="monotone"
                  dataKey="memoryPercent"
                  stroke={CHART_COLORS.memory.stroke}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: CHART_COLORS.memory.stroke }}
                  name={CHART_COLORS.memory.name}
                />
              )}
            </>
          )}

          {/* CPU area/line */}
          {(mode === 'cpu' || mode === 'combined') && hasCpuData && (
            <>
              {showArea && mode !== 'combined' && (
                <Area
                  type="monotone"
                  dataKey="cpuPercent"
                  stroke={CHART_COLORS.cpu.stroke}
                  fill={CHART_COLORS.cpu.fill}
                  fillOpacity={0.2}
                  strokeWidth={2}
                  name={CHART_COLORS.cpu.name}
                />
              )}
              {mode === 'combined' && (
                <Line
                  type="monotone"
                  dataKey="cpuPercent"
                  stroke={CHART_COLORS.cpu.stroke}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: CHART_COLORS.cpu.stroke }}
                  name={CHART_COLORS.cpu.name}
                />
              )}
            </>
          )}
        </ChartComponent>
      </ResponsiveContainer>
    </div>
  );
};

export default MemoryChart;
