import { useState, useEffect, useCallback, useRef } from 'react';

/**
 * Default dashboard URL - Dash server runs on port 8050.
 */
const DEFAULT_DASHBOARD_URL = 'http://localhost:8050';

/**
 * Auto-refresh interval in milliseconds (5 seconds).
 * Dashboard content updates via Dash's internal mechanisms,
 * but we check connection status periodically.
 */
const CONNECTION_CHECK_INTERVAL = 5000;

/**
 * Connection status states for the dashboard.
 */
type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'error';

/**
 * Props for the ConnectionStatus indicator.
 */
interface ConnectionStatusProps {
  state: ConnectionState;
  lastCheck: Date | null;
  onRefresh: () => void;
}

/**
 * ConnectionStatus shows the dashboard connection state.
 */
const ConnectionStatus: React.FC<ConnectionStatusProps> = ({
  state,
  lastCheck,
  onRefresh,
}) => {
  const stateConfig = {
    connecting: { color: 'bg-yellow-500', pulse: true, text: 'Connecting...' },
    connected: { color: 'bg-green-500', pulse: true, text: 'Live' },
    disconnected: { color: 'bg-gray-300', pulse: false, text: 'Disconnected' },
    error: { color: 'bg-red-500', pulse: false, text: 'Error' },
  };

  const config = stateConfig[state];

  return (
    <div className="flex items-center gap-4">
      <div className="flex items-center gap-2 text-sm">
        <span
          className={`w-2 h-2 rounded-full ${config.color} ${
            config.pulse ? 'animate-pulse' : ''
          }`}
        />
        <span
          className={
            state === 'connected'
              ? 'text-green-600'
              : state === 'error'
              ? 'text-red-600'
              : 'text-gray-500'
          }
        >
          {config.text}
        </span>
      </div>
      {lastCheck && (
        <span className="text-xs text-gray-400">
          Last check: {lastCheck.toLocaleTimeString()}
        </span>
      )}
      <button
        onClick={onRefresh}
        className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
        title="Refresh dashboard"
      >
        <svg
          className="w-4 h-4"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
          />
        </svg>
        Refresh
      </button>
    </div>
  );
};

/**
 * Props for the DashboardIframe component.
 */
interface DashboardIframeProps {
  url: string;
  onLoad: () => void;
  onError: () => void;
  refreshKey: number;
}

/**
 * DashboardIframe wraps the Dash dashboard in a responsive iframe.
 */
const DashboardIframe: React.FC<DashboardIframeProps> = ({
  url,
  onLoad,
  onError,
  refreshKey,
}) => {
  const iframeRef = useRef<HTMLIFrameElement>(null);

  return (
    <div className="relative w-full h-full min-h-[600px] bg-gray-50 rounded-lg overflow-hidden border border-gray-200">
      <iframe
        ref={iframeRef}
        key={refreshKey}
        src={url}
        title="Atlas Training Dashboard"
        className="absolute inset-0 w-full h-full"
        onLoad={onLoad}
        onError={onError}
        sandbox="allow-scripts allow-same-origin allow-forms"
        loading="eager"
      />
    </div>
  );
};

/**
 * Props for the DashboardError component.
 */
interface DashboardErrorProps {
  url: string;
  onRetry: () => void;
}

/**
 * DashboardError displays when the dashboard is unavailable.
 */
const DashboardError: React.FC<DashboardErrorProps> = ({ url, onRetry }) => (
  <div className="flex flex-col items-center justify-center min-h-[600px] bg-gray-50 rounded-lg border border-gray-200 p-8">
    <div className="p-4 bg-red-100 rounded-full mb-4">
      <svg
        className="w-8 h-8 text-red-500"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
        />
      </svg>
    </div>
    <h3 className="text-lg font-semibold text-gray-900 mb-2">
      Dashboard Unavailable
    </h3>
    <p className="text-gray-600 text-center max-w-md mb-4">
      Unable to connect to the Atlas training dashboard. Please ensure the
      dashboard server is running.
    </p>
    <div className="bg-gray-100 rounded-lg p-4 mb-4 font-mono text-sm text-gray-700">
      <p>Start the dashboard:</p>
      <code className="block mt-2 text-weaver-600">
        cd TheLoom/the-loom && poetry run python -m src.training.dashboard
      </code>
    </div>
    <p className="text-xs text-gray-500 mb-4">
      Dashboard URL: <code className="bg-gray-100 px-1 rounded">{url}</code>
    </p>
    <button
      onClick={onRetry}
      className="btn-primary flex items-center gap-2"
    >
      <svg
        className="w-4 h-4"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
        />
      </svg>
      Retry Connection
    </button>
  </div>
);

/**
 * DashboardLoading displays while the dashboard is loading.
 */
const DashboardLoading: React.FC = () => (
  <div className="absolute inset-0 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200">
    <div className="flex flex-col items-center gap-4">
      <svg
        className="animate-spin h-8 w-8 text-weaver-600"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
          fill="none"
        />
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        />
      </svg>
      <span className="text-gray-600">Loading Atlas Dashboard...</span>
    </div>
  </div>
);

/**
 * AtlasDashboard page - embeds the Dash live training dashboard.
 *
 * Features:
 * - Responsive iframe embedding of Dash dashboard (port 8050)
 * - Connection status monitoring with auto-retry
 * - Graceful error handling when dashboard is unavailable
 * - Manual refresh capability
 * - Mobile and desktop responsive layout
 *
 * The dashboard displays:
 * - Loss/Perplexity curves
 * - Learning rate schedule
 * - GPU memory and throughput stats
 * - Memory matrix norm evolution
 */
export const AtlasDashboard: React.FC = () => {
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting');
  const [lastCheck, setLastCheck] = useState<Date | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [isLoading, setIsLoading] = useState(true);
  const checkIntervalRef = useRef<number | null>(null);

  // Dashboard URL - could be made configurable in the future
  const dashboardUrl = DEFAULT_DASHBOARD_URL;

  /**
   * Check if the dashboard server is reachable.
   * Uses a HEAD request to avoid loading the full page.
   * Uses functional state updates to avoid depending on connectionState.
   */
  const checkConnection = useCallback(async () => {
    try {
      // Attempt to fetch just the headers to check if server is up
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 3000);

      await fetch(dashboardUrl, {
        method: 'HEAD',
        mode: 'no-cors', // Dashboard may not have CORS headers for HEAD
        signal: controller.signal,
      });

      clearTimeout(timeoutId);
      setLastCheck(new Date());

      // In no-cors mode, we can't check response.ok
      // A successful fetch (no abort/error) means the server is reachable
      // Use functional update to avoid depending on connectionState
      setConnectionState((prev) => prev !== 'connected' ? 'connected' : prev);
    } catch (err) {
      setLastCheck(new Date());
      // Use functional update to avoid depending on connectionState
      setConnectionState((prev) => {
        if (prev === 'connected') return 'disconnected';
        if (prev === 'connecting') return 'error';
        return prev;
      });
    }
  }, [dashboardUrl]);

  /**
   * Handle iframe load event - indicates dashboard loaded successfully.
   */
  const handleIframeLoad = useCallback(() => {
    setIsLoading(false);
    setConnectionState('connected');
    setLastCheck(new Date());
  }, []);

  /**
   * Handle iframe error event.
   */
  const handleIframeError = useCallback(() => {
    setIsLoading(false);
    setConnectionState('error');
    setLastCheck(new Date());
  }, []);

  /**
   * Force refresh the dashboard iframe.
   */
  const handleRefresh = useCallback(() => {
    setIsLoading(true);
    setConnectionState('connecting');
    setRefreshKey((prev) => prev + 1);
  }, []);

  // Set up periodic connection checking
  useEffect(() => {
    // Initial check
    checkConnection();

    // Set up interval for periodic checks
    checkIntervalRef.current = window.setInterval(
      checkConnection,
      CONNECTION_CHECK_INTERVAL
    );

    return () => {
      if (checkIntervalRef.current !== null) {
        clearInterval(checkIntervalRef.current);
      }
    };
  }, [checkConnection]);

  return (
    <div className="space-y-6 h-full flex flex-col">
      {/* Page Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Atlas Dashboard</h1>
          <p className="mt-2 text-gray-600">
            Live training metrics for Atlas model interpretability
          </p>
        </div>
        <ConnectionStatus
          state={connectionState}
          lastCheck={lastCheck}
          onRefresh={handleRefresh}
        />
      </div>

      {/* Info Banner */}
      <div className="bg-weaver-50 border border-weaver-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <div className="p-1 bg-weaver-100 rounded text-weaver-600">
            <svg
              className="w-5 h-5"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
          </div>
          <div className="text-sm">
            <p className="font-medium text-weaver-800">
              Real-time Training Monitoring
            </p>
            <p className="text-weaver-600 mt-1">
              The dashboard updates at 1 Hz showing loss, perplexity, learning rate,
              GPU stats, and memory matrix evolution. Start training with{' '}
              <code className="bg-weaver-100 px-1 rounded text-xs">
                poetry run python -m src.training.atlas_trainer --dashboard
              </code>
            </p>
          </div>
        </div>
      </div>

      {/* Dashboard Container - Flexible height */}
      <div className="flex-1 relative min-h-[600px]">
        {connectionState === 'error' || connectionState === 'disconnected' ? (
          <DashboardError url={dashboardUrl} onRetry={handleRefresh} />
        ) : (
          <>
            {isLoading && <DashboardLoading />}
            <DashboardIframe
              url={dashboardUrl}
              onLoad={handleIframeLoad}
              onError={handleIframeError}
              refreshKey={refreshKey}
            />
          </>
        )}
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="card">
          <h3 className="font-medium text-gray-900 mb-1">Concept Landscapes</h3>
          <p className="text-sm text-gray-500">
            View 3D PCA visualizations of concept evolution across epochs.
          </p>
        </div>
        <div className="card">
          <h3 className="font-medium text-gray-900 mb-1">Memory Tracing</h3>
          <p className="text-sm text-gray-500">
            Analyze M/S matrix statistics with magnitude and sparsity metrics.
          </p>
        </div>
        <div className="card">
          <h3 className="font-medium text-gray-900 mb-1">Checkpoint Analysis</h3>
          <p className="text-sm text-gray-500">
            Validate and explore Atlas model checkpoints with aligned PCA.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AtlasDashboard;
