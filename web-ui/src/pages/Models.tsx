/**
 * Models page - model browser with load/unload capabilities.
 *
 * Displays available models, their status, and controls for loading/unloading.
 * Uses WebSocket for real-time model and backend status updates.
 */
import { useState, useCallback, useEffect } from 'react';
import type { ModelInfo } from '@/services/modelApi';
import { formatModelSize } from '@/services/modelApi';
import { listBackends } from '@/services/backendApi';
import type { BackendStatus } from '@/services/backendApi';
import { ModelList } from '@/components/models';
import { useBackendStatus, useModelStatus } from '@/hooks/useWebSocket';

/**
 * Backend status display component.
 */
interface BackendStatusRowProps {
  name: string;
  available: boolean;
  type?: string;
}

const BackendStatusRow: React.FC<BackendStatusRowProps> = ({
  name,
  available,
  type,
}) => {
  return (
    <div className="flex items-center justify-between py-2 border-b border-gray-100 last:border-0">
      <div className="flex items-center space-x-3">
        <span
          className={`w-3 h-3 rounded-full ${
            available ? 'bg-green-500' : 'bg-gray-300'
          }`}
        />
        <span className="font-medium">{name}</span>
        {type && (
          <span className="text-xs text-gray-500 bg-gray-100 px-2 py-0.5 rounded">
            {type}
          </span>
        )}
      </div>
      <span
        className={`text-sm ${
          available ? 'text-green-600' : 'text-gray-500'
        }`}
      >
        {available ? 'Connected' : 'Not connected'}
      </span>
    </div>
  );
};

/**
 * Models page component.
 */
export const Models: React.FC = () => {
  // State for search
  const [searchQuery, setSearchQuery] = useState('');

  // State for model stats
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [backends, setBackends] = useState<BackendStatus[]>([]);
  const [isLoadingBackends, setIsLoadingBackends] = useState(true);

  // WebSocket hooks for real-time updates
  const { getBackend: getWsBackend } = useBackendStatus();
  // Note: useModelStatus() can be used for real-time model status updates
  useModelStatus(); // Subscribe to model status updates

  // Calculate stats from models
  const totalModels = models.length;
  const loadedCount = models.filter((m) => m.loaded).length;
  const totalMemoryUsed = models
    .filter((m) => m.loaded)
    .reduce((sum, m) => sum + (m.memoryUsed || 0), 0);

  // Load backends on mount
  useEffect(() => {
    const fetchBackends = async () => {
      try {
        const result = await listBackends();
        setBackends(result);
      } catch {
        // Backend list may fail if not connected
      } finally {
        setIsLoadingBackends(false);
      }
    };
    fetchBackends();
  }, []);

  // Update backend availability from WebSocket
  const getBackendAvailability = useCallback(
    (name: string): boolean => {
      // Check WebSocket status first
      const wsBackend = getWsBackend(name);
      if (wsBackend) {
        return wsBackend.available;
      }
      // Fall back to loaded backends
      const backend = backends.find((b) => b.name === name);
      return backend?.available ?? false;
    },
    [getWsBackend, backends]
  );

  // Handle model list changes
  const handleModelsChange = useCallback((newModels: ModelInfo[]) => {
    setModels(newModels);
  }, []);

  // Handle model loaded
  const handleModelLoaded = useCallback((model: ModelInfo) => {
    setModels((prev) =>
      prev.map((m) =>
        m.name === model.name ? { ...m, ...model, loaded: true } : m
      )
    );
  }, []);

  // Handle model unloaded
  const handleModelUnloaded = useCallback((name: string) => {
    setModels((prev) =>
      prev.map((m) =>
        m.name === name ? { ...m, loaded: false, memoryUsed: 0 } : m
      )
    );
  }, []);

  // Check backend connection
  const handleCheckConnection = useCallback(async () => {
    setIsLoadingBackends(true);
    try {
      const result = await listBackends();
      setBackends(result);
    } catch {
      // Connection check may fail
    } finally {
      setIsLoadingBackends(false);
    }
  }, []);

  // Get unique backends from models and loaded backends
  const allBackends = useCallback(() => {
    const backendNames = new Set<string>();
    models.forEach((m) => m.backend && backendNames.add(m.backend));
    backends.forEach((b) => backendNames.add(b.name));
    return Array.from(backendNames);
  }, [models, backends]);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Models</h1>
        <p className="mt-2 text-gray-600">
          View and manage available AI models
        </p>
      </div>

      {/* Status Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Total Models</h3>
          <p className="text-2xl font-bold text-weaver-600">{totalModels}</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Loaded</h3>
          <p className="text-2xl font-bold text-green-600">{loadedCount}</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Memory Usage</h3>
          <p className="text-2xl font-bold text-weaver-600">
            {totalMemoryUsed > 0 ? formatModelSize(totalMemoryUsed) : '0 GB'}
          </p>
        </div>
      </div>

      {/* Search Filter */}
      <div className="card">
        <div className="flex items-center space-x-4">
          <div className="relative flex-1">
            <svg
              className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
            <input
              type="text"
              placeholder="Search models by name, backend, or quantization..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input pl-10 w-full"
            />
          </div>
        </div>
      </div>

      {/* Models List */}
      <ModelList
        onModelsChange={handleModelsChange}
        onModelLoaded={handleModelLoaded}
        onModelUnloaded={handleModelUnloaded}
        searchQuery={searchQuery}
        showHeader={true}
        title="Available Models"
      />

      {/* Backend Status */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">
            Backend Status
          </h2>
          <button
            type="button"
            onClick={handleCheckConnection}
            disabled={isLoadingBackends}
            className="btn-secondary text-sm py-1.5 px-3"
          >
            {isLoadingBackends ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
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
                Checking...
              </span>
            ) : (
              'Check Connection'
            )}
          </button>
        </div>
        <div className="space-y-1">
          {/* Show backends from API */}
          {backends.length > 0 ? (
            backends.map((backend) => (
              <BackendStatusRow
                key={backend.name}
                name={backend.name}
                available={getBackendAvailability(backend.name)}
                type={backend.type}
              />
            ))
          ) : (
            <>
              {/* Default backend entries when not connected */}
              <BackendStatusRow
                name="TheLoom"
                available={getBackendAvailability('loom')}
                type="loom"
              />
              <BackendStatusRow
                name="Claude Code"
                available={getBackendAvailability('claudecode')}
                type="claudecode"
              />
            </>
          )}
        </div>
        {backends.length === 0 && !isLoadingBackends && (
          <p className="text-sm text-gray-500 mt-4">
            Unable to fetch backend status. Ensure the Weaver API server is
            running.
          </p>
        )}
      </div>
    </div>
  );
};

export default Models;
