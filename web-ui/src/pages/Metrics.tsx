/**
 * Metrics page - real-time visualization of conveyance metrics.
 *
 * Displays D_eff, Beta, Alignment, and C_pair metrics with charts.
 * Uses WebSocket connection for real-time measurement updates.
 * Supports both individual metric views and multi-metric comparison dashboard.
 */
import { useState, useMemo, useCallback, useEffect } from 'react';
import { useMeasurementEvents, useWebSocket } from '@/hooks/useWebSocket';
import { DEffChart } from '@/components/visualizations/DEffChart';
import { BetaChart } from '@/components/visualizations/BetaChart';
import { AlignmentChart } from '@/components/visualizations/AlignmentChart';
import { CPairChart } from '@/components/visualizations/CPairChart';
import { MetricsDashboard } from '@/components/visualizations/MetricsDashboard';
import { MetricSummaryGrid } from '@/components/visualizations/MetricSummary';
import { computeBetaStatus, type MeasurementData, type MeasurementEvent } from '@/types';

/** Convert MeasurementEvent to MeasurementData for charts */
function toMeasurementData(event: MeasurementEvent): MeasurementData {
  return {
    turn: event.turn,
    deff: event.deff,
    beta: event.beta,
    alignment: event.alignment,
    cpair: event.cpair,
    sender: event.senderName,
    receiver: event.receiverName,
  };
}

/** Calculate summary statistics from measurements */
function calculateSummary(measurements: MeasurementEvent[]): {
  latestDeff: number | null;
  latestBeta: number | null;
  latestAlignment: number | null;
  latestCpair: number | null;
  avgDeff: number | null;
  avgBeta: number | null;
  avgAlignment: number | null;
  avgCpair: number | null;
  count: number;
} {
  if (measurements.length === 0) {
    return {
      latestDeff: null,
      latestBeta: null,
      latestAlignment: null,
      latestCpair: null,
      avgDeff: null,
      avgBeta: null,
      avgAlignment: null,
      avgCpair: null,
      count: 0,
    };
  }

  const latest = measurements[measurements.length - 1];
  const count = measurements.length;

  const sum = measurements.reduce(
    (acc, m) => ({
      deff: acc.deff + m.deff,
      beta: acc.beta + m.beta,
      alignment: acc.alignment + m.alignment,
      cpair: acc.cpair + m.cpair,
    }),
    { deff: 0, beta: 0, alignment: 0, cpair: 0 }
  );

  return {
    latestDeff: latest.deff,
    latestBeta: latest.beta,
    latestAlignment: latest.alignment,
    latestCpair: latest.cpair,
    avgDeff: sum.deff / count,
    avgBeta: sum.beta / count,
    avgAlignment: sum.alignment / count,
    avgCpair: sum.cpair / count,
    count,
  };
}

/** Get the status color for beta value */
function getBetaStatusColor(beta: number | null): string {
  if (beta === null) return 'text-gray-400';
  const status = computeBetaStatus(beta);
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
}

/**
 * MetricsSummaryCard displays a summary of a single metric.
 */
interface MetricsSummaryCardProps {
  title: string;
  value: number | null;
  avgValue: number | null;
  count: number;
  colorClass?: string;
  loading?: boolean;
  format?: (v: number) => string;
}

const MetricsSummaryCard: React.FC<MetricsSummaryCardProps> = ({
  title,
  value,
  avgValue,
  count,
  colorClass = 'text-weaver-600',
  loading = false,
  format = (v) => v.toFixed(4),
}) => {
  if (loading) {
    return (
      <div className="card animate-pulse">
        <div className="h-4 bg-gray-200 rounded w-1/2 mb-2" />
        <div className="h-8 bg-gray-200 rounded w-1/3 mb-2" />
        <div className="h-3 bg-gray-200 rounded w-1/4" />
      </div>
    );
  }

  return (
    <div className="card">
      <h3 className="text-sm font-medium text-gray-500">{title}</h3>
      <p className={`text-2xl font-bold ${colorClass}`}>
        {value !== null ? format(value) : '--'}
      </p>
      <p className="text-xs text-gray-400 mt-1">
        {count > 0
          ? `Avg: ${avgValue !== null ? format(avgValue) : '--'} (${count} pts)`
          : 'No data'}
      </p>
    </div>
  );
};

/**
 * ConnectionStatus shows the WebSocket connection state.
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

/** View mode for the metrics page */
type MetricsViewMode = 'dashboard' | 'grid';

export const Metrics: React.FC = () => {
  const { isConnected, state, subscribe, subscribedChannels } = useWebSocket();
  const { measurements, clear } = useMeasurementEvents(100);
  const [selectedSession, setSelectedSession] = useState<string>('');
  const [isLoading, setIsLoading] = useState(false);
  const [viewMode, setViewMode] = useState<MetricsViewMode>('dashboard');

  // Subscribe to measurements channel on mount
  useEffect(() => {
    if (isConnected && !subscribedChannels.includes('measurements')) {
      subscribe('measurements');
    }
  }, [isConnected, subscribedChannels, subscribe]);

  // Convert measurements to chart data format
  const chartData = useMemo(
    () => measurements.map(toMeasurementData),
    [measurements]
  );

  // Calculate summary statistics
  const summary = useMemo(() => calculateSummary(measurements), [measurements]);

  // Handle session selection (for future session-specific data loading)
  const handleSessionChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedSession(e.target.value);
  }, []);

  // Handle clear data
  const handleClearData = useCallback(() => {
    clear();
  }, [clear]);

  // Handle load metrics (placeholder for API integration)
  const handleLoadMetrics = useCallback(() => {
    if (!selectedSession) return;
    setIsLoading(true);
    // TODO: Implement session-specific metrics loading via API
    setTimeout(() => setIsLoading(false), 1000);
  }, [selectedSession]);

  // Get beta status color
  const betaColor = getBetaStatusColor(summary.latestBeta);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Metrics</h1>
          <p className="mt-2 text-gray-600">
            Real-time conveyance metrics visualization (D_eff, Beta, Alignment, C_pair)
          </p>
        </div>
        <div className="flex items-center gap-4">
          {/* View Mode Toggle */}
          <div className="flex items-center gap-1 bg-gray-100 rounded-lg p-1">
            <button
              onClick={() => setViewMode('dashboard')}
              className={`
                flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors
                ${viewMode === 'dashboard'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
                }
              `}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              Dashboard
            </button>
            <button
              onClick={() => setViewMode('grid')}
              className={`
                flex items-center gap-1.5 px-3 py-1.5 rounded-md text-sm font-medium transition-colors
                ${viewMode === 'grid'
                  ? 'bg-white text-gray-900 shadow-sm'
                  : 'text-gray-600 hover:text-gray-900'
                }
              `}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
              Grid
            </button>
          </div>
          <ConnectionStatus isConnected={isConnected} state={state} />
          {measurements.length > 0 && (
            <button
              onClick={handleClearData}
              className="text-sm text-gray-500 hover:text-gray-700"
            >
              Clear Data
            </button>
          )}
        </div>
      </div>

      {/* Dashboard View */}
      {viewMode === 'dashboard' && (
        <MetricsDashboard
          data={chartData}
          loading={isLoading}
          showViewTabs
          showSummaryCards
          allowMetricSelection
          chartHeight={300}
        />
      )}

      {/* Grid View - Individual Charts */}
      {viewMode === 'grid' && (
        <>
          {/* Metrics Summary Cards */}
          <MetricSummaryGrid
            data={chartData}
            loading={isLoading}
            showSparklines
          />

          {/* Chart Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* D_eff Chart */}
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Effective Dimension (D_eff)
              </h2>
              <DEffChart
                data={chartData}
                height={256}
                loading={isLoading}
                showArea
                showAverage={chartData.length > 1}
              />
            </div>

            {/* Beta Chart */}
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Convergence (Beta)
              </h2>
              <BetaChart
                data={chartData}
                height={256}
                loading={isLoading}
                showStatusZones
                showThresholds
              />
            </div>

            {/* Alignment Chart */}
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Alignment
              </h2>
              <AlignmentChart
                data={chartData}
                height={256}
                loading={isLoading}
                showArea
                showAverage={chartData.length > 1}
              />
            </div>

            {/* C_pair Chart */}
            <div className="card">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Cross-pair Correlation (C_pair)
              </h2>
              <CPairChart
                data={chartData}
                height={256}
                loading={isLoading}
                showArea
                showAverage={chartData.length > 1}
              />
            </div>
          </div>
        </>
      )}

      {/* Session Selector */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Session Data
        </h2>
        <div className="flex items-center space-x-4">
          <select
            className="input flex-1"
            value={selectedSession}
            onChange={handleSessionChange}
          >
            <option value="" disabled>
              Select a session to view metrics
            </option>
            {/* Sessions will be populated via API */}
          </select>
          <button
            className="btn-primary"
            onClick={handleLoadMetrics}
            disabled={!selectedSession || isLoading}
          >
            {isLoading ? 'Loading...' : 'Load Metrics'}
          </button>
        </div>
        <p className="text-sm text-gray-500 mt-2">
          {isConnected
            ? 'Connected - Metrics update in real-time via WebSocket.'
            : 'Connect to see real-time metric updates.'}
        </p>
      </div>
    </div>
  );
};

export default Metrics;
