/**
 * DEffChart component - specialized chart for Effective Dimension (D_eff).
 *
 * D_eff measures the effective dimensionality of semantic content in agent
 * communications. Higher values indicate more complex semantic structures.
 */
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Area,
  ComposedChart,
  type TooltipProps,
} from 'recharts';
import type { MeasurementData, MeasurementEvent } from '@/types';

/**
 * DEffChart component props.
 */
export interface DEffChartProps {
  /** Measurement data to display */
  data: MeasurementData[] | MeasurementEvent[];
  /** Chart height in pixels */
  height?: number;
  /** Whether to show the area fill under the line */
  showArea?: boolean;
  /** Whether to show the grid */
  showGrid?: boolean;
  /** Whether data is loading */
  loading?: boolean;
  /** Additional CSS class names */
  className?: string;
  /** Whether to show the average line */
  showAverage?: boolean;
  /** Callback when a data point is clicked */
  onDataPointClick?: (data: MeasurementData | MeasurementEvent) => void;
}

/**
 * Calculate the average D_eff from the data.
 */
function calculateAverage(data: Array<MeasurementData | MeasurementEvent>): number {
  if (!data || data.length === 0) return 0;
  const sum = data.reduce((acc, d) => acc + (d.deff ?? 0), 0);
  return sum / data.length;
}

/**
 * Calculate the max D_eff from the data.
 */
function calculateMax(data: Array<MeasurementData | MeasurementEvent>): number {
  if (!data || data.length === 0) return 0;
  return Math.max(...data.map((d) => d.deff ?? 0));
}

/**
 * Custom tooltip for D_eff chart.
 */
const DEffTooltip: React.FC<TooltipProps<number, string>> = ({
  active,
  payload,
  label,
}) => {
  if (!active || !payload || payload.length === 0) {
    return null;
  }

  const value = payload[0]?.value as number;
  const formattedValue = typeof value === 'number' ? value.toFixed(4) : 'N/A';
  const data = payload[0]?.payload as MeasurementData | MeasurementEvent;

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 p-3 max-w-xs">
      <p className="text-sm font-medium text-gray-900">Turn {label}</p>
      <div className="mt-1 flex items-center gap-2">
        <span className="w-3 h-3 rounded-full bg-purple-500" />
        <span className="text-sm text-gray-600">D_eff:</span>
        <span className="text-sm font-medium text-gray-900">{formattedValue}</span>
      </div>
      {data.senderName && data.receiverName && (
        <p className="mt-2 text-xs text-gray-500">
          {data.senderName} → {data.receiverName}
        </p>
      )}
      <p className="mt-1 text-xs text-gray-400">
        Effective dimensionality of semantic content
      </p>
    </div>
  );
};

/**
 * Loading skeleton for D_eff chart.
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
          d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"
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
        d="M7 12l3-3 3 3 4-4M8 21l4-4 4 4M3 4h18M4 4h16v12a1 1 0 01-1 1H5a1 1 0 01-1-1V4z"
      />
    </svg>
    <p className="text-sm">No D_eff data available</p>
    <p className="text-xs mt-1">Start a session to see measurements</p>
  </div>
);

/**
 * DEffChart component for visualizing effective dimension.
 *
 * @example
 * ```tsx
 * <DEffChart
 *   data={measurements}
 *   height={300}
 *   showArea
 *   showAverage
 * />
 * ```
 */
export const DEffChart: React.FC<DEffChartProps> = ({
  data,
  height = 256,
  showArea = false,
  showGrid = true,
  loading = false,
  className = '',
  showAverage = false,
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

  const average = calculateAverage(data);
  const max = calculateMax(data);
  const ChartComponent = showArea ? ComposedChart : LineChart;

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
          <span className="w-2 h-2 rounded-full bg-purple-500" />
          <span className="text-gray-500">Avg:</span>
          <span className="font-medium text-gray-700">{average.toFixed(3)}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Max:</span>
          <span className="font-medium text-gray-700">{max.toFixed(3)}</span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-gray-500">Points:</span>
          <span className="font-medium text-gray-700">{data.length}</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <ChartComponent
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
            domain={['auto', 'auto']}
            tickFormatter={(value: number) => value.toFixed(1)}
          />
          <Tooltip content={<DEffTooltip />} />
          {showAverage && (
            <ReferenceLine
              y={average}
              stroke="#A855F7"
              strokeDasharray="5 5"
              label={{
                value: `Avg: ${average.toFixed(2)}`,
                position: 'right',
                fill: '#A855F7',
                fontSize: 10,
              }}
            />
          )}
          {showArea && (
            <Area
              type="monotone"
              dataKey="deff"
              fill="#8B5CF6"
              fillOpacity={0.1}
              stroke="none"
            />
          )}
          <Line
            type="monotone"
            dataKey="deff"
            stroke="#8B5CF6"
            strokeWidth={2}
            dot={{ r: 3, fill: '#8B5CF6' }}
            activeDot={{ r: 5, fill: '#8B5CF6' }}
            name="D_eff"
          />
        </ChartComponent>
      </ResponsiveContainer>
    </div>
  );
};

export default DEffChart;
