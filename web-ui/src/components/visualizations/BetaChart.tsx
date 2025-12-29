/**
 * BetaChart component - specialized chart for Beta (Collapse Indicator).
 *
 * Beta indicates dimensional collapse in agent communications.
 * Lower values are better:
 * - 1.5-2.0: Optimal range
 * - 2.0-2.5: Monitor
 * - 2.5-3.0: Concerning
 * - 3.0+: Critical
 */
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  type TooltipProps,
} from 'recharts';
import {
  type MeasurementData,
  type MeasurementEvent,
  type BetaStatus,
  computeBetaStatus,
  BETA_STATUS_RANGES,
} from '@/types';

/**
 * BetaChart component props.
 */
export interface BetaChartProps {
  /** Measurement data to display */
  data: MeasurementData[] | MeasurementEvent[];
  /** Chart height in pixels */
  height?: number;
  /** Whether to show the status zones (colored bands) */
  showStatusZones?: boolean;
  /** Whether to show the grid */
  showGrid?: boolean;
  /** Whether data is loading */
  loading?: boolean;
  /** Additional CSS class names */
  className?: string;
  /** Whether to show the threshold lines */
  showThresholds?: boolean;
  /** Callback when a data point is clicked */
  onDataPointClick?: (data: MeasurementData | MeasurementEvent) => void;
  /** Callback when a critical value is detected */
  onCriticalValue?: (data: MeasurementData | MeasurementEvent) => void;
}

/**
 * Status color mapping.
 */
const STATUS_COLORS: Record<BetaStatus, { bg: string; line: string; text: string }> = {
  optimal: { bg: '#D1FAE5', line: '#10B981', text: 'text-green-700' },
  monitor: { bg: '#FEF3C7', line: '#F59E0B', text: 'text-yellow-700' },
  concerning: { bg: '#FED7AA', line: '#F97316', text: 'text-orange-700' },
  critical: { bg: '#FECACA', line: '#EF4444', text: 'text-red-700' },
  unknown: { bg: '#E5E7EB', line: '#6B7280', text: 'text-gray-700' },
};

/**
 * Get status badge for beta value.
 */
function getBetaStatusBadge(status: BetaStatus): React.ReactNode {
  const colors = STATUS_COLORS[status];
  const labels: Record<BetaStatus, string> = {
    optimal: 'Optimal',
    monitor: 'Monitor',
    concerning: 'Concerning',
    critical: 'Critical',
    unknown: 'Unknown',
  };

  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 rounded text-xs font-medium ${colors.text}`}
      style={{ backgroundColor: colors.bg }}
    >
      {labels[status]}
    </span>
  );
}

/**
 * Calculate statistics for beta values.
 */
function calculateStats(data: Array<MeasurementData | MeasurementEvent>): {
  avg: number;
  min: number;
  max: number;
  currentStatus: BetaStatus;
} {
  if (!data || data.length === 0) {
    return { avg: 0, min: 0, max: 0, currentStatus: 'unknown' };
  }

  const values = data.map((d) => d.beta ?? 0);
  const avg = values.reduce((a, b) => a + b, 0) / values.length;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const latest = data[data.length - 1];
  const currentStatus = computeBetaStatus(latest.beta ?? 0);

  return { avg, min, max, currentStatus };
}

/**
 * Custom tooltip for Beta chart.
 */
const BetaTooltip: React.FC<TooltipProps<number, string>> = ({
  active,
  payload,
  label,
}) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const value = payload[0]?.value as number;
  const status = computeBetaStatus(value);
  const formattedValue = typeof value === 'number' ? value.toFixed(4) : 'N/A';
  const data = payload[0]?.payload as MeasurementData | MeasurementEvent;

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 max-w-xs">
      <p className="text-sm font-medium text-gray-900">Turn {label}</p>
      <div className="mt-1 flex items-center gap-2">
        <span
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: STATUS_COLORS[status].line }}
        />
        <span className="text-sm text-gray-600">Beta:</span>
        <span className="text-sm font-medium text-gray-900">{formattedValue}</span>
        {getBetaStatusBadge(status)}
      </div>
      {data.senderName && data.receiverName && (
        <p className="mt-2 text-xs text-gray-500">
          {data.senderName} → {data.receiverName}
        </p>
      )}
      <p className="mt-1 text-xs text-gray-400">
        {status === 'optimal' && 'Dimensional structure is well preserved'}
        {status === 'monitor' && 'Slight compression detected, watch for drift'}
        {status === 'concerning' && 'Significant compression, consider intervention'}
        {status === 'critical' && 'Severe collapse, immediate action recommended'}
        {status === 'unknown' && 'Unable to assess dimensional quality'}
      </p>
    </div>
  );
};

/**
 * Loading skeleton for Beta chart.
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
          d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"
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
        d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6"
      />
    </svg>
    <p className="text-sm">No Beta data available</p>
    <p className="text-xs mt-1">Start a session to see measurements</p>
  </div>
);

/**
 * BetaChart component for visualizing collapse indicator.
 *
 * @example
 * ```tsx
 * <BetaChart
 *   data={measurements}
 *   height={300}
 *   showStatusZones
 *   showThresholds
 * />
 * ```
 */
export const BetaChart: React.FC<BetaChartProps> = ({
  data,
  height = 256,
  showStatusZones = true,
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

  // Determine Y-axis domain
  const yMax = Math.max(stats.max, BETA_STATUS_RANGES.critical.min + 1);

  return (
    <div className={className}>
      {/* Stats bar */}
      <div className="flex items-center gap-4 mb-2 text-xs">
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Current:</span>
          {getBetaStatusBadge(stats.currentStatus)}
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
        <ComposedChart
          data={data}
          margin={{ top: 5, right: 20, left: 10, bottom: 5 }}
          onClick={handleClick}
        >
          {showGrid && <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" />}

          {/* Status zone backgrounds */}
          {showStatusZones && (
            <>
              <ReferenceArea
                y1={BETA_STATUS_RANGES.optimal.min}
                y2={BETA_STATUS_RANGES.optimal.max}
                fill={STATUS_COLORS.optimal.bg}
                fillOpacity={0.5}
              />
              <ReferenceArea
                y1={BETA_STATUS_RANGES.monitor.min}
                y2={BETA_STATUS_RANGES.monitor.max}
                fill={STATUS_COLORS.monitor.bg}
                fillOpacity={0.5}
              />
              <ReferenceArea
                y1={BETA_STATUS_RANGES.concerning.min}
                y2={BETA_STATUS_RANGES.concerning.max}
                fill={STATUS_COLORS.concerning.bg}
                fillOpacity={0.5}
              />
              <ReferenceArea
                y1={BETA_STATUS_RANGES.critical.min}
                y2={yMax}
                fill={STATUS_COLORS.critical.bg}
                fillOpacity={0.5}
              />
            </>
          )}

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
            domain={[0, yMax]}
            tickFormatter={(value: number) => value.toFixed(1)}
          />
          <Tooltip content={<BetaTooltip />} />

          {/* Threshold lines */}
          {showThresholds && (
            <>
              <ReferenceLine
                y={BETA_STATUS_RANGES.optimal.min}
                stroke={STATUS_COLORS.optimal.line}
                strokeDasharray="5 5"
                label={{
                  value: 'Optimal',
                  position: 'right',
                  fill: STATUS_COLORS.optimal.line,
                  fontSize: 9,
                }}
              />
              <ReferenceLine
                y={BETA_STATUS_RANGES.monitor.min}
                stroke={STATUS_COLORS.monitor.line}
                strokeDasharray="5 5"
                label={{
                  value: 'Monitor',
                  position: 'right',
                  fill: STATUS_COLORS.monitor.line,
                  fontSize: 9,
                }}
              />
              <ReferenceLine
                y={BETA_STATUS_RANGES.critical.min}
                stroke={STATUS_COLORS.critical.line}
                strokeDasharray="5 5"
                label={{
                  value: 'Critical',
                  position: 'right',
                  fill: STATUS_COLORS.critical.line,
                  fontSize: 9,
                }}
              />
            </>
          )}

          <Area
            type="monotone"
            dataKey="beta"
            fill="#EF4444"
            fillOpacity={0.1}
            stroke="none"
          />
          <Line
            type="monotone"
            dataKey="beta"
            stroke="#EF4444"
            strokeWidth={2}
            dot={{ r: 3, fill: '#EF4444' }}
            activeDot={{ r: 5, fill: '#EF4444' }}
            name="Beta"
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BetaChart;
