/**
 * AlignmentChart component - specialized chart for Alignment metric.
 *
 * Alignment measures the cosine similarity between sender and receiver
 * hidden states. Values range from 0 to 1 (or -1 to 1 in some cases).
 * Higher alignment indicates better semantic agreement.
 */
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  type TooltipProps,
} from 'recharts';
import type { MeasurementData, MeasurementEvent } from '@/types';

/**
 * AlignmentChart component props.
 */
export interface AlignmentChartProps {
  /** Measurement data to display */
  data: MeasurementData[] | MeasurementEvent[];
  /** Chart height in pixels */
  height?: number;
  /** Whether to show the area fill */
  showArea?: boolean;
  /** Whether to show the grid */
  showGrid?: boolean;
  /** Whether data is loading */
  loading?: boolean;
  /** Additional CSS class names */
  className?: string;
  /** Whether to show the average line */
  showAverage?: boolean;
  /** Whether to show threshold lines */
  showThresholds?: boolean;
  /** Callback when a data point is clicked */
  onDataPointClick?: (data: MeasurementData | MeasurementEvent) => void;
}

/**
 * Get alignment quality level.
 */
type AlignmentLevel = 'high' | 'moderate' | 'low' | 'poor';

function getAlignmentLevel(value: number): AlignmentLevel {
  if (value >= 0.8) return 'high';
  if (value >= 0.5) return 'moderate';
  if (value >= 0.2) return 'low';
  return 'poor';
}

/**
 * Alignment level colors.
 */
const LEVEL_COLORS: Record<AlignmentLevel, { bg: string; text: string }> = {
  high: { bg: 'bg-green-100', text: 'text-green-700' },
  moderate: { bg: 'bg-blue-100', text: 'text-blue-700' },
  low: { bg: 'bg-yellow-100', text: 'text-yellow-700' },
  poor: { bg: 'bg-red-100', text: 'text-red-700' },
};

/**
 * Calculate statistics for alignment values.
 */
function calculateStats(data: Array<MeasurementData | MeasurementEvent>): {
  avg: number;
  min: number;
  max: number;
  latest: number;
  level: AlignmentLevel;
} {
  if (!data || data.length === 0) {
    return { avg: 0, min: 0, max: 0, latest: 0, level: 'poor' };
  }

  const values = data.map((d) => d.alignment ?? 0);
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const latest = values[values.length - 1];
  const level = getAlignmentLevel(latest);

  return { avg, min, max, latest, level };
}

/**
 * Custom tooltip for Alignment chart.
 */
const AlignmentTooltip: React.FC<TooltipProps<number, string>> = ({
  active,
  payload,
  label,
}) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const value = payload[0]?.value as number;
  const level = getAlignmentLevel(value);
  const formattedValue = typeof value === 'number' ? (value * 100).toFixed(1) : 'N/A';
  const data = payload[0]?.payload as MeasurementData | MeasurementEvent;

  const levelLabels: Record<AlignmentLevel, string> = {
    high: 'High alignment',
    moderate: 'Moderate alignment',
    low: 'Low alignment',
    poor: 'Poor alignment',
  };

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 max-w-xs">
      <p className="text-sm font-medium text-gray-900">Turn {label}</p>
      <div className="mt-1 flex items-center gap-2">
        <span className="w-3 h-3 rounded-full bg-blue-500" />
        <span className="text-sm text-gray-600">Alignment:</span>
        <span className="text-sm font-medium text-gray-900">{formattedValue}%</span>
      </div>
      <div className="mt-1">
        <span
          className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium ${LEVEL_COLORS[level].bg} ${LEVEL_COLORS[level].text}`}
        >
          {levelLabels[level]}
        </span>
      </div>
      {data.senderName && data.receiverName && (
        <p className="mt-2 text-xs text-gray-500">
          {data.senderName} → {data.receiverName}
        </p>
      )}
      <p className="mt-1 text-xs text-gray-400">
        Cosine similarity between hidden states
      </p>
    </div>
  );
};

/**
 * Loading skeleton for Alignment chart.
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
          d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"
        />
      </svg>
    </div>
  </div>
);

/**
 * Empty state when no data is available.
 */
const EmptyState: React.FC<{ height: number }> = ({ height }) => (
  <div
    className="border border-dashed border-gray-300 rounded-lg flex flex-col items-center justify-center text-gray-400"
    style={{ height }}
  >
    <svg className="w-10 h-10 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={2}
        d="M4 5a1 1 0 011-1h14a1 1 0 011 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V5zM4 13a1 1 0 011-1h6a1 1 0 011 1v6a1 1 0 01-1 1H5a1 1 0 01-1-1v-6zM16 13a1 1 0 011-1h2a1 1 0 011 1v6a1 1 0 01-1 1h-2a1 1 0 01-1-1v-6z"
      />
    </svg>
    <p className="text-sm">No Alignment data available</p>
    <p className="text-xs mt-1">Start a session to see measurements</p>
  </div>
);

/**
 * AlignmentChart component for visualizing semantic alignment.
 *
 * @example
 * ```tsx
 * <AlignmentChart
 *   data={measurements}
 *   height={300}
 *   showArea
 *   showAverage
 * />
 * ```
 */
export const AlignmentChart: React.FC<AlignmentChartProps> = ({
  data,
  height = 256,
  showArea = true,
  showGrid = true,
  loading = false,
  className = '',
  showAverage = false,
  showThresholds = false,
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
    return (
      <div className={className}>
        <EmptyState height={height} />
      </div>
    );
  }

  const stats = calculateStats(data);

  // Handle click on data point
  const handleClick = onDataPointClick
    ? (event: { activePayload?: Array<{ payload: MeasurementData | MeasurementEvent }> }) => {
        if (event.activePayload?.[0]?.payload) {
          onDataPointClick(event.activePayload[0].payload);
        }
      }
    : undefined;

  return (
    <div className={className}>
      {/* Stats bar */}
      <div className="flex items-center gap-4 mb-2 text-xs">
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Current:</span>
          <span
            className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium ${LEVEL_COLORS[stats.level].bg} ${LEVEL_COLORS[stats.level].text}`}
          >
            {(stats.latest * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Avg:</span>
          <span className="font-medium text-gray-700">
            {(stats.avg * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Range:</span>
          <span className="font-medium text-gray-700">
            {(stats.min * 100).toFixed(0)}% - {(stats.max * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart
          data={data}
          margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
          onClick={handleClick}
        >
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />}
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
            domain={[0, 1]}
            tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`}
          />
          <Tooltip content={<AlignmentTooltip />} />

          {/* Threshold lines */}
          {showThresholds && (
            <>
              <ReferenceLine
                y={0.8}
                stroke="#22C55E"
                strokeDasharray="5 5"
                label={{
                  value: 'High (80%)',
                  position: 'right',
                  fill: '#22C55E',
                  fontSize: 9,
                }}
              />
              <ReferenceLine
                y={0.5}
                stroke="#3B82F6"
                strokeDasharray="5 5"
                label={{
                  value: 'Moderate (50%)',
                  position: 'right',
                  fill: '#3B82F6',
                  fontSize: 9,
                }}
              />
            </>
          )}

          {showAverage && (
            <ReferenceLine
              y={stats.avg}
              stroke="#60A5FA"
              strokeDasharray="5 5"
              label={{
                value: `Avg: ${(stats.avg * 100).toFixed(0)}%`,
                position: 'right',
                fill: '#60A5FA',
                fontSize: 10,
              }}
            />
          )}

          {showArea && (
            <Area
              type="monotone"
              dataKey="alignment"
              fill="#3B82F6"
              fillOpacity={0.2}
              stroke="#3B82F6"
              strokeWidth={2}
              dot={{ r: 3, fill: '#3B82F6' }}
              activeDot={{ r: 5, fill: '#3B82F6' }}
              name="Alignment"
            />
          )}
          {!showArea && (
            <Line
              type="monotone"
              dataKey="alignment"
              stroke="#3B82F6"
              strokeWidth={2}
              dot={{ r: 3, fill: '#3B82F6' }}
              activeDot={{ r: 5, fill: '#3B82F6' }}
              name="Alignment"
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default AlignmentChart;
