/**
 * Resource Dashboard component for displaying GPU and system resource usage.
 *
 * Displays GPU memory, utilization, queue depth, and system resources
 * with real-time updates via WebSocket.
 */
import { useState, useEffect, useMemo, useCallback } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import type { GPUStatus, ResourceStatus } from '@/types/backend';
import { useResourceStatus, useWebSocket, type ResourceUpdateData } from '@/hooks/useWebSocket';
import { GPUChart } from './GPUChart';
import { MemoryChart } from './MemoryChart';

/**
 * Historical data point for resource charts.
 */
export interface ResourceDataPoint {
  timestamp: number;
  gpuMemory: number;
  gpuUtil: number;
  queueDepth: number;
  cpuPercent?: number;
  memoryPercent?: number;
}

/**
 * Props for ResourceDashboard component.
 */
export interface ResourceDashboardProps {
  /** Maximum number of data points to display in charts */
  maxDataPoints?: number;
  /** Height of charts in pixels */
  chartHeight?: number;
  /** Whether to show the queue depth chart */
  showQueueChart?: boolean;
  /** Whether to show GPU details */
  showGPUDetails?: boolean;
  /** External resource status (for testing/static data) */
  resourceStatus?: ResourceStatus | null;
  /** Callback when resource status updates */
  onResourceUpdate?: (data: ResourceUpdateData) => void;
}

/**
 * Format bytes to human-readable format.
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
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
 * Get background color based on utilization level.
 */
function getStatusBgColor(percent: number): string {
  if (percent >= 90) return 'bg-red-500';
  if (percent >= 75) return 'bg-yellow-500';
  return 'bg-green-500';
}

/**
 * Progress bar component for resource utilization.
 */
interface ProgressBarProps {
  value: number;
  max?: number;
  label?: string;
  showValue?: boolean;
  height?: 'sm' | 'md' | 'lg';
}

const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  max = 100,
  label,
  showValue = true,
  height = 'md',
}) => {
  const percent = Math.min((value / max) * 100, 100);
  const heightClass = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  }[height];

  return (
    <div className="w-full">
      {label && (
        <div className="flex items-center justify-between mb-1">
          <span className="text-sm text-gray-600">{label}</span>
          {showValue && (
            <span className={`text-sm font-medium ${getStatusColor(percent)}`}>
              {formatPercent(percent)}
            </span>
          )}
        </div>
      )}
      <div className={`w-full bg-gray-200 rounded-full overflow-hidden ${heightClass}`}>
        <div
          className={`${heightClass} rounded-full transition-all duration-300 ${getStatusBgColor(percent)}`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
};

/**
 * GPU Card component for individual GPU status.
 */
interface GPUCardProps {
  gpu: GPUStatus;
}

const GPUCard: React.FC<GPUCardProps> = ({ gpu }) => {
  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <div className={`w-2.5 h-2.5 rounded-full ${getStatusBgColor(gpu.memoryPercent)}`} />
          <h3 className="font-medium text-gray-900">GPU {gpu.index}</h3>
        </div>
        <span className="text-sm text-gray-500">{gpu.name}</span>
      </div>

      <div className="space-y-4">
        {/* Memory Usage */}
        <div>
          <ProgressBar
            value={gpu.memoryPercent}
            label="Memory"
          />
          <div className="flex justify-between text-xs text-gray-500 mt-1">
            <span>{formatBytes(gpu.memoryUsedMB * 1024 * 1024)}</span>
            <span>{formatBytes(gpu.memoryTotalMB * 1024 * 1024)}</span>
          </div>
        </div>

        {/* Utilization */}
        {gpu.utilization !== undefined && (
          <ProgressBar
            value={gpu.utilization}
            label="Utilization"
          />
        )}

        {/* Temperature */}
        {gpu.temperature !== undefined && (
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-600">Temperature</span>
            <span
              className={`font-medium ${
                gpu.temperature >= 80
                  ? 'text-red-600'
                  : gpu.temperature >= 70
                  ? 'text-yellow-600'
                  : 'text-green-600'
              }`}
            >
              {gpu.temperature}°C
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Stat Card component for summary statistics.
 */
interface StatCardProps {
  title: string;
  value: string;
  subtitle?: string;
  icon?: React.ReactNode;
  color?: 'default' | 'green' | 'yellow' | 'red';
}

const StatCard: React.FC<StatCardProps> = ({
  title,
  value,
  subtitle,
  icon,
  color = 'default',
}) => {
  const colorClasses = {
    default: 'text-weaver-600',
    green: 'text-green-600',
    yellow: 'text-yellow-600',
    red: 'text-red-600',
  }[color];

  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-sm font-medium text-gray-500">{title}</h3>
          <p className={`text-2xl font-bold mt-1 ${colorClasses}`}>{value}</p>
          {subtitle && (
            <p className="text-sm text-gray-500 mt-1">{subtitle}</p>
          )}
        </div>
        {icon && (
          <div className="text-gray-400">{icon}</div>
        )}
      </div>
    </div>
  );
};

/**
 * Connection Status component.
 */
const ConnectionStatus: React.FC<{ isConnected: boolean; state: string }> = ({
  isConnected,
  state,
}) => (
  <div className="flex items-center gap-2 text-sm">
    <span
      className={`w-2 h-2 rounded-full ${
        isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-300'
      }`}
    />
    <span className={isConnected ? 'text-green-600' : 'text-gray-500'}>
      {isConnected ? 'Live' : state === 'connecting' ? 'Connecting...' : 'Disconnected'}
    </span>
  </div>
);

/**
 * Resource Dashboard component.
 */
export const ResourceDashboard: React.FC<ResourceDashboardProps> = ({
  maxDataPoints = 60,
  chartHeight = 200,
  showQueueChart = true,
  showGPUDetails = true,
  resourceStatus: externalStatus,
  onResourceUpdate,
}) => {
  // WebSocket connection
  const { isConnected, state, subscribe, subscribedChannels } = useWebSocket();
  const wsStatus = useResourceStatus();

  // Historical data for charts
  const [history, setHistory] = useState<ResourceDataPoint[]>([]);

  // Subscribe to resources channel on mount
  useEffect(() => {
    if (isConnected && !subscribedChannels.includes('resources')) {
      subscribe('resources');
    }
  }, [isConnected, subscribedChannels, subscribe]);

  // Update history when resource status changes
  useEffect(() => {
    if (wsStatus.status) {
      const dataPoint: ResourceDataPoint = {
        timestamp: Date.now(),
        gpuMemory: wsStatus.gpuMemory,
        gpuUtil: wsStatus.gpuUtilization,
        queueDepth: wsStatus.queueDepth,
      };

      setHistory((prev) => {
        const next = [...prev, dataPoint];
        if (next.length > maxDataPoints) {
          return next.slice(-maxDataPoints);
        }
        return next;
      });

      onResourceUpdate?.(wsStatus.status);
    }
  }, [wsStatus.status, maxDataPoints, onResourceUpdate]);

  // Chart data with formatted time
  const chartData = useMemo(() => {
    return history.map((point, index) => ({
      ...point,
      time: index,
      timeLabel: new Date(point.timestamp).toLocaleTimeString(),
    }));
  }, [history]);

  // Calculate averages
  const averages = useMemo(() => {
    if (history.length === 0) {
      return { gpuMemory: 0, gpuUtil: 0, queueDepth: 0 };
    }
    const sum = history.reduce(
      (acc, point) => ({
        gpuMemory: acc.gpuMemory + point.gpuMemory,
        gpuUtil: acc.gpuUtil + point.gpuUtil,
        queueDepth: acc.queueDepth + point.queueDepth,
      }),
      { gpuMemory: 0, gpuUtil: 0, queueDepth: 0 }
    );
    return {
      gpuMemory: sum.gpuMemory / history.length,
      gpuUtil: sum.gpuUtil / history.length,
      queueDepth: sum.queueDepth / history.length,
    };
  }, [history]);

  // Current values
  const currentValues = {
    gpuMemory: wsStatus.gpuMemory,
    gpuUtil: wsStatus.gpuUtilization,
    queueDepth: wsStatus.queueDepth,
  };

  // Get status color for queue depth
  const getQueueColor = useCallback((depth: number): 'default' | 'green' | 'yellow' | 'red' => {
    if (depth === 0) return 'green';
    if (depth < 3) return 'default';
    if (depth < 10) return 'yellow';
    return 'red';
  }, []);

  // Clear history
  const handleClearHistory = useCallback(() => {
    setHistory([]);
  }, []);

  // Mock GPU data for display when not connected
  const gpuData: GPUStatus[] = externalStatus?.gpus ?? [
    {
      index: 0,
      name: 'NVIDIA GeForce RTX 4090',
      memoryUsedMB: currentValues.gpuMemory * 24000 / 100,
      memoryTotalMB: 24000,
      memoryPercent: currentValues.gpuMemory,
      utilization: currentValues.gpuUtil,
      temperature: 65,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          title="GPU Memory"
          value={formatPercent(currentValues.gpuMemory)}
          subtitle={`Avg: ${formatPercent(averages.gpuMemory)}`}
          color={currentValues.gpuMemory >= 90 ? 'red' : currentValues.gpuMemory >= 75 ? 'yellow' : 'green'}
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
              />
            </svg>
          }
        />
        <StatCard
          title="GPU Utilization"
          value={formatPercent(currentValues.gpuUtil)}
          subtitle={`Avg: ${formatPercent(averages.gpuUtil)}`}
          color={currentValues.gpuUtil >= 90 ? 'red' : currentValues.gpuUtil >= 75 ? 'yellow' : 'green'}
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          }
        />
        <StatCard
          title="Queue Depth"
          value={currentValues.queueDepth.toString()}
          subtitle={`Avg: ${averages.queueDepth.toFixed(1)}`}
          color={getQueueColor(currentValues.queueDepth)}
          icon={
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 10h16M4 14h16M4 18h16"
              />
            </svg>
          }
        />
      </div>

      {/* GPU Memory & Utilization Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* GPU Memory Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">GPU Memory</h2>
            <ConnectionStatus isConnected={isConnected} state={state} />
          </div>
          <GPUChart
            data={history}
            height={chartHeight}
            mode="memory"
            showThresholds
            loading={isConnected && history.length === 0}
          />
        </div>

        {/* GPU Utilization Chart */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-gray-900">GPU Utilization</h2>
            {history.length > 0 && (
              <button
                type="button"
                onClick={handleClearHistory}
                className="text-sm text-gray-500 hover:text-gray-700"
              >
                Clear
              </button>
            )}
          </div>
          <GPUChart
            data={history}
            height={chartHeight}
            mode="utilization"
            showThresholds
            loading={isConnected && history.length === 0}
          />
        </div>
      </div>

      {/* System Memory Chart (if data available) */}
      {history.some((d) => d.memoryPercent !== undefined || d.cpuPercent !== undefined) && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">System Resources</h2>
          <MemoryChart
            data={history}
            height={chartHeight}
            mode="combined"
            showThresholds
          />
        </div>
      )}

      {/* Queue Depth Chart */}
      {showQueueChart && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Inference Queue Depth</h2>
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={chartHeight}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis
                  dataKey="time"
                  tick={{ fill: '#6b7280', fontSize: 12 }}
                  axisLine={{ stroke: '#e5e7eb' }}
                />
                <YAxis
                  tick={{ fill: '#6b7280', fontSize: 12 }}
                  axisLine={{ stroke: '#e5e7eb' }}
                  allowDecimals={false}
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="bg-white p-2 shadow-lg rounded border text-sm">
                          <p className="text-gray-600">{data.timeLabel}</p>
                          <p className="text-amber-600 font-medium">
                            Queue: {data.queueDepth} requests
                          </p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Line
                  type="stepAfter"
                  dataKey="queueDepth"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          ) : (
            <div
              className="flex items-center justify-center text-gray-500"
              style={{ height: chartHeight }}
            >
              {isConnected ? 'Waiting for data...' : 'Connect to see queue depth'}
            </div>
          )}
        </div>
      )}

      {/* GPU Details */}
      {showGPUDetails && (
        <div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">GPU Details</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {gpuData.map((gpu) => (
              <GPUCard key={gpu.index} gpu={gpu} />
            ))}
            {gpuData.length === 0 && (
              <div className="col-span-full text-center py-8 text-gray-500">
                No GPUs detected. Ensure the backend is running with GPU support.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ResourceDashboard;
