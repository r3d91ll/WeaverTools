/**
 * GPUChart component - specialized chart for GPU memory and utilization visualization.
 *
 * Displays GPU memory usage and/or utilization percentage over time with
 * color-coded thresholds and real-time updates via WebSocket.
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
 * GPU chart mode - which metric to display.
 */
export type GPUChartMode = 'memory' | 'utilization' | 'combined';

/**
 * GPUChart component props.
 */
export interface GPUChartProps {
  /** Historical data points to display */
  data: ResourceDataPoint[];
  /** Chart height in pixels */
  height?: number;
  /** Chart mode - memory, utilization, or combined */
  mode?: GPUChartMode;
  /** Whether to show the grid */
  showGrid?: boolean;
  /** Whether data is loading */
  loading?: boolean;
  /** Additional CSS class names */
  className?: string;
  /** Whether to show threshold reference lines */
  showThresholds?: boolean;
  /** GPU index for multi-GPU systems */
  gpuIndex?: number;
  /** Title for the chart */
  title?: string;
  /** Callback when a data point is clicked */
  onDataPointClick?: (data: ResourceDataPoint) => void;
}

/**
 * Chart color configurations.
 */
const CHART_COLORS = {
  memory: {
    stroke: '#4f46e5', // indigo-600
    fill: '#4f46e5',
    name: 'GPU Memory',
  },
  utilization: {
    stroke: '#22c55e', // green-500
    fill: '#22c55e',
    name: 'GPU Utilization',
  },
} as const;

/**
 * Format percentage value.
 */
function formatPercent(value: number): string {
  return `${value.toFixed(1)}%`;
}

/**
 * Get status color class based on utilization level.
 */
function getStatusColorClass(percent: number): string {
  if (percent >= 90) return 'text-red-600';
  if (percent >= 75) return 'text-yellow-600';
  return 'text-green-600';
}

/**
 * Get color based on utilization level.
 */
function getUtilizationColor(percent: number): string {
  if (percent >= 90) return '#EF4444'; // red-500
  if (percent >= 75) return '#EAB308'; // yellow-500
  return '#22C55E'; // green-500
}

/**
 * Calculate statistics from the data.
 */
function calculateStats(
  data: ResourceDataPoint[],
  key: 'gpuMemory' | 'gpuUtil'
): { avg: number; max: number; current: number } {
  if (!data || data.length === 0) {
    return { avg: 0, max: 0, current: 0 };
  }

  const values = data.map((d) => d[key] ?? 0);
  const sum = values.reduce((acc, v) => acc + v, 0);

  return {
    avg: sum / values.length,
    max: Math.max(...values),
    current: values[values.length - 1] ?? 0,
  };
}

/**
 * Calculate the average GPU utilization from the data.
 */
function calculateAverage(data: ResourceDataPoint[]): number {
  if (!data || data.length === 0) return 0;
  const sum = data.reduce((acc, d) => acc + (d.gpuUtil ?? 0), 0);
  return sum / data.length;
}

/**
 * Calculate the max GPU utilization from the data.
 */
function calculateMax(data: ResourceDataPoint[]): number {
  if (!data || data.length === 0) return 0;
  return Math.max(...data.map((d) => d.gpuUtil ?? 0));
}

/**
 * Custom tooltip for GPU chart.
 */
interface GPUTooltipProps extends TooltipProps<number, string> {
  mode: GPUChartMode;
}

const GPUTooltip: React.FC<GPUTooltipProps> = ({
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
        {(mode === 'memory' || mode === 'combined') && (
          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CHART_COLORS.memory.stroke }}
            />
            <span className="text-sm text-gray-600">Memory:</span>
            <span className={`text-sm font-medium ${getStatusColorClass(data.gpuMemory)}`}>
              {formatPercent(data.gpuMemory)}
            </span>
          </div>
        )}
        {(mode === 'utilization' || mode === 'combined') && (
          <div className="flex items-center gap-2">
            <span
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CHART_COLORS.utilization.stroke }}
            />
            <span className="text-sm text-gray-600">Utilization:</span>
            <span className={`text-sm font-medium ${getStatusColorClass(data.gpuUtil)}`}>
              {formatPercent(data.gpuUtil)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Loading skeleton for GPU chart.
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
          d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
        />
      </svg>
    </div>
  </div>
);

/**
 * Empty state when no data is available.
 */
const EmptyState: React.FC<{ height: number; message?: string }> = ({ height, message }) => (
  <div
    className="border border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center text-gray-400"
    style={{ height }}
  >
    <svg className="w-10 h-10 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
      />
    </svg>
    <p className="text-sm">{message ?? 'No GPU data available'}</p>
    <p className="text-xs mt-1">Connect to see GPU utilization</p>
  </div>
);

/**
 * GPUChart component for visualizing GPU memory and utilization over time.
 *
 * @example
 * ```tsx
 * <GPUChart
 *   data={resourceHistory}
 *   mode="combined"
 *   height={200}
 *   showThresholds
 * />
 * ```
 */
export const GPUChart: React.FC<GPUChartProps> = ({
  data,
  height = 200,
  mode = 'utilization',
  showGrid = true,
  loading = false,
  className = '',
  showThresholds = false,
  gpuIndex,
  title,
  onDataPointClick,
}) => {
  // Handle loading state
  if (loading) {
    return (
      <div className={className}>
        {title && (
          <h3 className="text-sm font-medium text-gray-700 mb-2">{title}</h3>
        )}
        <LoadingSkeleton height={height} />
      </div>
    );
  }

  // Handle empty data
  if (!data || data.length === 0) {
    const emptyMessage =
      mode === 'memory'
        ? 'No GPU memory data available'
        : mode === 'utilization'
        ? 'No GPU utilization data available'
        : gpuIndex !== undefined
        ? `No data for GPU ${gpuIndex}`
        : 'No GPU data available';

    return (
      <div className={className}>
        {title && (
          <h3 className="text-sm font-medium text-gray-700 mb-2">{title}</h3>
        )}
        <EmptyState height={height} message={emptyMessage} />
      </div>
    );
  }

  // Calculate stats based on mode
  const memoryStats = calculateStats(data, 'gpuMemory');
  const utilStats = calculateStats(data, 'gpuUtil');

  // Handle click on data point
  const handleClick = onDataPointClick
    ? (event: { activePayload?: Array<{ payload: ResourceDataPoint }> }) => {
        if (event.activePayload?.[0]?.payload) {
          onDataPointClick(event.activePayload[0].payload);
        }
      }
    : undefined;

  // Format chart data with time index
  const chartData = data.map((point, index) => ({
    ...point,
    time: index,
    timeLabel: new Date(point.timestamp).toLocaleTimeString(),
  }));

  // Use ComposedChart for combined mode, AreaChart otherwise
  const ChartComponent = mode === 'combined' ? ComposedChart : AreaChart;

  return (
    <div className={className}>
      {/* Stats bar */}
      <div className="flex items-center gap-4 mb-2 text-xs flex-wrap">
        {title && (
          <span className="font-medium text-gray-700">{title}</span>
        )}
        {(mode === 'memory' || mode === 'combined') && (
          <>
            <div className="flex items-center gap-1">
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: CHART_COLORS.memory.stroke }}
              />
              <span className="text-gray-500">Memory:</span>
              <span className={`font-medium ${getStatusColorClass(memoryStats.current)}`}>
                {formatPercent(memoryStats.current)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-gray-500">Avg:</span>
              <span className="font-medium text-gray-700">{formatPercent(memoryStats.avg)}</span>
            </div>
          </>
        )}
        {(mode === 'utilization' || mode === 'combined') && (
          <>
            <div className="flex items-center gap-1">
              <span
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: CHART_COLORS.utilization.stroke }}
              />
              <span className="text-gray-500">Util:</span>
              <span className={`font-medium ${getStatusColorClass(utilStats.current)}`}>
                {formatPercent(utilStats.current)}
              </span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-gray-500">Avg:</span>
              <span className="font-medium text-gray-700">{formatPercent(utilStats.avg)}</span>
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
            stroke="#6B7280"
            fontSize={12}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            domain={[0, 100]}
            tickFormatter={(value: number) => `${value}%`}
          />
          <Tooltip content={<GPUTooltip mode={mode} />} />
          {showThresholds && (
            <>
              <ReferenceLine
                y={75}
                stroke="#EAB308"
                strokeDasharray="5 5"
                label={{
                  value: 'Warning',
                  position: 'right',
                  fill: '#EAB308',
                  fontSize: 10,
                }}
              />
              <ReferenceLine
                y={90}
                stroke="#EF4444"
                strokeDasharray="5 5"
                label={{
                  value: 'Critical',
                  position: 'right',
                  fill: '#EF4444',
                  fontSize: 10,
                }}
              />
            </>
          )}

          {/* Memory area/line */}
          {(mode === 'memory' || mode === 'combined') && (
            <>
              {mode !== 'combined' && (
                <Area
                  type="monotone"
                  dataKey="gpuMemory"
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
                  dataKey="gpuMemory"
                  stroke={CHART_COLORS.memory.stroke}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: CHART_COLORS.memory.stroke }}
                  name={CHART_COLORS.memory.name}
                />
              )}
            </>
          )}

          {/* Utilization area/line */}
          {(mode === 'utilization' || mode === 'combined') && (
            <>
              {mode !== 'combined' && (
                <Area
                  type="monotone"
                  dataKey="gpuUtil"
                  stroke={CHART_COLORS.utilization.stroke}
                  fill={CHART_COLORS.utilization.fill}
                  fillOpacity={0.2}
                  strokeWidth={2}
                  name={CHART_COLORS.utilization.name}
                />
              )}
              {mode === 'combined' && (
                <Line
                  type="monotone"
                  dataKey="gpuUtil"
                  stroke={CHART_COLORS.utilization.stroke}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: CHART_COLORS.utilization.stroke }}
                  name={CHART_COLORS.utilization.name}
                />
              )}
            </>
          )}
        </ChartComponent>
      </ResponsiveContainer>
    </div>
  );
};

export default GPUChart;
