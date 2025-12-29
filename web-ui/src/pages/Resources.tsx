/**
 * Resources page - GPU and system resource monitoring.
 *
 * Displays real-time GPU memory, utilization, queue depth, and system resources.
 * Uses WebSocket connection for live updates.
 */
import { useState, useCallback, useEffect } from 'react';
import { useWebSocket, useResourceStatus, useBetaAlerts } from '@/hooks/useWebSocket';
import { ResourceDashboard } from '@/components/resources';

/**
 * Alert Notification component.
 */
interface AlertNotificationProps {
  count: number;
  hasUnread: boolean;
  onViewAlerts: () => void;
  onDismiss: () => void;
}

const AlertNotification: React.FC<AlertNotificationProps> = ({
  count,
  hasUnread,
  onViewAlerts,
  onDismiss,
}) => {
  if (count === 0) return null;

  return (
    <div
      className={`
        flex items-center gap-3 p-3 rounded-lg
        ${hasUnread ? 'bg-yellow-50 border border-yellow-200' : 'bg-gray-50 border border-gray-200'}
      `}
    >
      <div
        className={`
          flex items-center justify-center w-8 h-8 rounded-full
          ${hasUnread ? 'bg-yellow-100 text-yellow-600' : 'bg-gray-100 text-gray-500'}
        `}
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
          />
        </svg>
      </div>
      <div className="flex-1">
        <p className={hasUnread ? 'text-yellow-800 font-medium' : 'text-gray-600'}>
          {count} {count === 1 ? 'alert' : 'alerts'} received
        </p>
        <p className="text-sm text-gray-500">
          Beta values reached concerning or critical levels
        </p>
      </div>
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onViewAlerts}
          className="text-sm text-weaver-600 hover:text-weaver-700 font-medium"
        >
          View
        </button>
        <button
          type="button"
          onClick={onDismiss}
          className="text-sm text-gray-500 hover:text-gray-700"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
};

/**
 * Connection Status component.
 */
const ConnectionStatus: React.FC<{
  isConnected: boolean;
  state: string;
  onConnect: () => void;
  onDisconnect: () => void;
}> = ({ isConnected, state, onConnect, onDisconnect }) => (
  <div className="flex items-center gap-3">
    <div className="flex items-center gap-2 text-sm">
      <span
        className={`w-2 h-2 rounded-full ${
          isConnected ? 'bg-green-500 animate-pulse' : 'bg-gray-300'
        }`}
      />
      <span className={isConnected ? 'text-green-600' : 'text-gray-500'}>
        {isConnected ? 'Connected' : state === 'connecting' ? 'Connecting...' : 'Disconnected'}
      </span>
    </div>
    {isConnected ? (
      <button
        type="button"
        onClick={onDisconnect}
        className="text-sm text-gray-500 hover:text-gray-700"
      >
        Disconnect
      </button>
    ) : (
      <button
        type="button"
        onClick={onConnect}
        className="text-sm text-weaver-600 hover:text-weaver-700 font-medium"
      >
        Connect
      </button>
    )}
  </div>
);

/**
 * System Info component for displaying backend status.
 */
const SystemInfo: React.FC = () => {
  return (
    <div className="card">
      <h2 className="text-lg font-semibold text-gray-900 mb-4">System Information</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-500">Platform</p>
          <p className="text-gray-900 font-medium">WeaverTools</p>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-500">API Endpoint</p>
          <p className="text-gray-900 font-medium font-mono text-sm">
            {import.meta.env.VITE_API_URL || 'http://localhost:8081'}
          </p>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-500">WebSocket</p>
          <p className="text-gray-900 font-medium font-mono text-sm">
            {import.meta.env.VITE_WS_URL || 'ws://localhost:8081/ws'}
          </p>
        </div>
        <div className="p-3 bg-gray-50 rounded-lg">
          <p className="text-sm text-gray-500">Update Interval</p>
          <p className="text-gray-900 font-medium">Real-time (WebSocket)</p>
        </div>
      </div>
    </div>
  );
};

/**
 * Resources page component.
 */
export const Resources: React.FC = () => {
  const { isConnected, state, connect, disconnect, subscribe, subscribedChannels } = useWebSocket();
  const { gpuMemory, gpuUtilization, queueDepth } = useResourceStatus();
  const { alerts, hasUnread, markRead, clear: clearAlerts } = useBetaAlerts(50);

  // Alert modal state
  const [showAlerts, setShowAlerts] = useState(false);

  // Subscribe to resources channel on mount
  useEffect(() => {
    if (isConnected && !subscribedChannels.includes('resources')) {
      subscribe('resources');
    }
  }, [isConnected, subscribedChannels, subscribe]);

  // Handle connect
  const handleConnect = useCallback(() => {
    connect();
  }, [connect]);

  // Handle disconnect
  const handleDisconnect = useCallback(() => {
    disconnect();
  }, [disconnect]);

  // Handle view alerts
  const handleViewAlerts = useCallback(() => {
    setShowAlerts(true);
    markRead();
  }, [markRead]);

  // Handle dismiss alerts
  const handleDismissAlerts = useCallback(() => {
    clearAlerts();
  }, [clearAlerts]);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Resources</h1>
          <p className="mt-2 text-gray-600">
            Monitor GPU memory, utilization, and inference queue status
          </p>
        </div>
        <ConnectionStatus
          isConnected={isConnected}
          state={state}
          onConnect={handleConnect}
          onDisconnect={handleDisconnect}
        />
      </div>

      {/* Alert Notification */}
      <AlertNotification
        count={alerts.length}
        hasUnread={hasUnread}
        onViewAlerts={handleViewAlerts}
        onDismiss={handleDismissAlerts}
      />

      {/* Resource Dashboard */}
      <ResourceDashboard
        maxDataPoints={60}
        chartHeight={200}
        showQueueChart={true}
        showGPUDetails={true}
      />

      {/* System Information */}
      <SystemInfo />

      {/* Help Card */}
      <div className="card bg-blue-50 border-blue-200">
        <div className="flex gap-4">
          <div className="flex-shrink-0">
            <svg
              className="w-6 h-6 text-blue-600"
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
          <div>
            <h3 className="text-lg font-medium text-blue-900">Resource Monitoring</h3>
            <p className="mt-1 text-blue-700">
              This dashboard shows real-time resource usage from the Weaver backend.
              GPU memory, utilization, and inference queue depth are updated via WebSocket.
            </p>
            <ul className="mt-2 text-sm text-blue-600 space-y-1">
              <li>• <strong>GPU Memory:</strong> Current VRAM usage across all GPUs</li>
              <li>• <strong>GPU Utilization:</strong> Processing load on GPU compute cores</li>
              <li>• <strong>Queue Depth:</strong> Number of pending inference requests</li>
            </ul>
            <p className="mt-2 text-sm text-blue-600">
              Alerts are triggered when beta values reach concerning or critical levels during experiments.
            </p>
          </div>
        </div>
      </div>

      {/* Alert Modal */}
      {showAlerts && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] flex flex-col">
            <div className="flex items-center justify-between p-4 border-b">
              <h2 className="text-lg font-semibold text-gray-900">Resource Alerts</h2>
              <button
                type="button"
                onClick={() => setShowAlerts(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M6 18L18 6M6 6l12 12"
                  />
                </svg>
              </button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {alerts.length > 0 ? (
                <div className="space-y-3">
                  {alerts.map((alert, index) => (
                    <div
                      key={`${alert.timestamp}-${index}`}
                      className={`
                        p-3 rounded-lg border
                        ${
                          alert.status === 'critical'
                            ? 'bg-red-50 border-red-200'
                            : 'bg-yellow-50 border-yellow-200'
                        }
                      `}
                    >
                      <div className="flex items-start justify-between">
                        <div>
                          <span
                            className={`
                              text-xs font-medium px-2 py-0.5 rounded
                              ${
                                alert.status === 'critical'
                                  ? 'bg-red-100 text-red-700'
                                  : 'bg-yellow-100 text-yellow-700'
                              }
                            `}
                          >
                            {alert.status.toUpperCase()}
                          </span>
                          <p className="mt-1 text-sm text-gray-900">
                            Beta value: <strong>{alert.beta.toFixed(4)}</strong>
                          </p>
                          <p className="text-xs text-gray-500 mt-1">
                            Threshold: {alert.threshold.toFixed(4)}
                          </p>
                        </div>
                        <span className="text-xs text-gray-500">
                          {new Date(alert.timestamp).toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  No alerts recorded
                </div>
              )}
            </div>
            <div className="p-4 border-t bg-gray-50 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => {
                  clearAlerts();
                  setShowAlerts(false);
                }}
                className="btn-secondary"
              >
                Clear All
              </button>
              <button
                type="button"
                onClick={() => setShowAlerts(false)}
                className="btn-primary"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Resources;
