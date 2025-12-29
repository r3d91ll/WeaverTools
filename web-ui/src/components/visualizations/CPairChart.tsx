/**
 * CPairChart component - specialized chart for C_pair (Cross-pair Correlation).
 *
 * C_pair measures the bilateral conveyance score between agent pairs.
 * It captures how well semantic content is preserved in both directions
 * of agent communication. Values range from 0 to 1.
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
 * CPairChart component props.
 */
export interface CPairChartProps {
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
  /** Whether to show quality thresholds */
  showThresholds?: boolean;
  /** Callback when a data point is clicked */
  onDataPointClick?: (data: MeasurementData | MeasurementEvent) => void;
}

/**
 * C_pair quality levels.
 */
type CPairLevel = 'excellent' | 'good' | 'fair' | 'poor';

function getCPairLevel(value: number): CPairLevel {
  if (value >= 0.8) return 'excellent';
  if (value >= 0.6) return 'good';
  if (value >= 0.4) return 'fair';
  return 'poor';
}

/**
 * C_pair level colors.
 */
const LEVEL_COLORS: Record<CPairLevel, { bg: string; text: string; line: string }> = {
  excellent: { bg: 'bg-emerald-100', text: 'text-emerald-700', line: '#10B981' },
  good: { bg: 'bg-green-100', text: 'text-green-700', line: '#22C55E' },
  fair: { bg: 'bg-yellow-100', text: 'text-yellow-700', line: '#EAB308' },
  poor: { bg: 'bg-red-100', text: 'text-red-700', line: '#EF4444' },
};

/**
 * Calculate statistics for C_pair values.
 */
function calculateStats(data: Array<MeasurementData | MeasurementEvent>): {
  avg: number;
  min: number;
  max: number;
  latest: number;
  level: CPairLevel;
  trend: 'up' | 'down' | 'stable';
} {
  if (!data || data.length === 0) {
    return { avg: 0, min: 0, max: 0, latest: 0, level: 'poor', trend: 'stable' };
  }

  const values = data.map((d) => d.cpair ?? 0);
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const latest = values[values.length - 1];
  const level = getCPairLevel(latest);

  // Calculate trend (comparing last 5 points if available)
  let trend: 'up' | 'down' | 'stable' = 'stable';
  if (values.length >= 2) {
    const recentWindow = values.slice(-Math.min(5, values.length));
    const firstHalf = recentWindow.slice(0, Math.floor(recentWindow.length / 2));
    const secondHalf = recentWindow.slice(Math.floor(recentWindow.length / 2));
    const firstAvg = firstHalf.reduce((a, b) => a + b, 0) / firstHalf.length;
    const secondAvg = secondHalf.reduce((a, b) => a + b, 0) / secondHalf.length;
    const diff = secondAvg - firstAvg;
    if (diff > 0.05) trend = 'up';
    else if (diff < -0.05) trend = 'down';
  }

  return { avg, min, max, latest, level, trend };
}

/**
 * Trend indicator icon.
 */
const TrendIcon: React.FC<{ trend: 'up' | 'down' | 'stable' }> = ({ trend }) => {
  if (trend === 'up') {
    return (
      <svg className="w-3 h-3 text-green-500" fill="currentColor" viewBox="0 0 20 20">
        <path
          fillRule="evenodd"
          d="M5.293 9.707a1 1 0 010-1.414l4-4a1 1 0 011.414 0l4 4a1 1 0 01-1.414 1.414L11 7.414V15a1 1 0 11-2 0V7.414L6.707 9.707a1 1 0 01-1.414 0z"
          clipRule="evenodd"
        />
      </svg>
    );
  }
  if (trend === 'down') {
    return (
      <svg className="w-3 h-3 text-red-500" fill="currentColor" viewBox="0 0 20 20">
        <path
          fillRule="evenodd"
          d="M14.707 10.293a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 111.414-1.414L9 12.586V5a1 1 0 012 0v7.586l2.293-2.293a1 1 0 011.414 0z"
          clipRule="evenodd"
        />
      </svg>
    );
  }
  return (
    <svg className="w-3 h-3 text-gray-400" fill="currentColor" viewBox="0 0 20 20">
      <path
        fillRule="evenodd"
        d="M3 10a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1z"
        clipRule="evenodd"
      />
    </svg>
  );
};

/**
 * Custom tooltip for C_pair chart.
 */
const CPairTooltip: React.FC<TooltipProps<number, string>> = ({
  active,
  payload,
  label,
}) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const value = payload[0]?.value as number;
  const level = getCPairLevel(value);
  const formattedValue = typeof value === 'number' ? value.toFixed(4) : 'N/A';
  const data = payload[0]?.payload as MeasurementData | MeasurementEvent;

  const levelLabels: Record<CPairLevel, string> = {
    excellent: 'Excellent bilateral conveyance',
    good: 'Good bilateral conveyance',
    fair: 'Fair bilateral conveyance',
    poor: 'Poor bilateral conveyance',
  };

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 max-w-xs">
      <p className="text-sm font-medium text-gray-900">Turn {label}</p>
      <div className="mt-1 flex items-center gap-2">
        <span className="w-3 h-3 rounded-full bg-emerald-500" />
        <span className="text-sm text-gray-600">C_pair:</span>
        <span className="text-sm font-medium text-gray-900">{formattedValue}</span>
      </div>
      <div className="mt-1">
        <span
          className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium ${LEVEL_COLORS[level].bg} ${LEVEL_COLORS[level].text}`}
        >
          {levelLabels[level]}
        </span>
      </div>
      {data.sender && data.receiver && (
        <p className="mt-2 text-xs text-gray-500">
          {data.sender} ↔ {data.receiver}
        </p>
      )}
      {data.senderName && data.receiverName && (
        <p className="mt-2 text-xs text-gray-500">
          {data.senderName} ↔ {data.receiverName}
        </p>
      )}
      <p className="mt-1 text-xs text-gray-400">
        Bilateral semantic preservation score
      </p>
    </div>
  );
};

/**
 * Loading skeleton for C_pair chart.
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
          d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"
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
        d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4"
      />
    </svg>
    <p className="text-sm">No C_pair data available</p>
    <p className="text-xs mt-1">Start a session to see measurements</p>
  </div>
);

/**
 * CPairChart component for visualizing bilateral conveyance.
 *
 * @example
 * ```tsx
 * <CPairChart
 *   data={measurements}
 *   height={300}
 *   showArea
 *   showAverage
 * />
 * ```
 */
export const CPairChart: React.FC<CPairChartProps> = ({
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
            {stats.latest.toFixed(3)}
          </span>
          <TrendIcon trend={stats.trend} />
        </div>
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Avg:</span>
          <span className="font-medium text-gray-700">{stats.avg.toFixed(3)}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Range:</span>
          <span className="font-medium text-gray-700">
            {stats.min.toFixed(2)} - {stats.max.toFixed(2)}
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
            tickFormatter={(value: number) => value.toFixed(2)}
          />
          <Tooltip content={<CPairTooltip />} />

          {/* Threshold lines */}
          {showThresholds && (
            <>
              <ReferenceLine
                y={0.8}
                stroke={LEVEL_COLORS.excellent.line}
                strokeDasharray="5 5"
                label={{
                  value: 'Excellent',
                  position: 'right',
                  fill: LEVEL_COLORS.excellent.line,
                  fontSize: 9,
                }}
              />
              <ReferenceLine
                y={0.6}
                stroke={LEVEL_COLORS.good.line}
                strokeDasharray="5 5"
                label={{
                  value: 'Good',
                  position: 'right',
                  fill: LEVEL_COLORS.good.line,
                  fontSize: 9,
                }}
              />
              <ReferenceLine
                y={0.4}
                stroke={LEVEL_COLORS.fair.line}
                strokeDasharray="5 5"
                label={{
                  value: 'Fair',
                  position: 'right',
                  fill: LEVEL_COLORS.fair.line,
                  fontSize: 9,
                }}
              />
            </>
          )}

          {showAverage && (
            <ReferenceLine
              y={stats.avg}
              stroke="#6EE7B7"
              strokeDasharray="5 5"
              label={{
                value: `Avg: ${stats.avg.toFixed(2)}`,
                position: 'right',
                fill: '#6EE7B7',
                fontSize: 10,
              }}
            />
          )}

          {showArea && (
            <Area
              type="monotone"
              dataKey="cpair"
              fill="#10B981"
              fillOpacity={0.2}
              stroke="#10B981"
              strokeWidth={2}
              dot={{ r: 3, fill: '#10B981' }}
              activeDot={{ r: 5, fill: '#10B981' }}
              name="C_pair"
            />
          )}
          {!showArea && (
            <Line
              type="monotone"
              dataKey="cpair"
              stroke="#10B981"
              strokeWidth={2}
              dot={{ r: 3, fill: '#10B981' }}
              activeDot={{ r: 5, fill: '#10B981' }}
              name="C_pair"
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
};

export default CPairChart;
