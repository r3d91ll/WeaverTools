/**
 * React hook for managing models with load/unload operations.
 *
 * Provides comprehensive model management including:
 * - Model listing and filtering
 * - Load/unload operations with state tracking
 * - Real-time WebSocket updates
 * - Statistics computation
 */

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import type { ModelInfo, ModelLoadOptions } from '@/services/modelApi';
import { listModels, loadModel, unloadModel, formatModelSize } from '@/services/modelApi';
import { useWebSocketEvent, useModelStatus, getWebSocketService } from './useWebSocket';

/** Filter type for model list. */
export type ModelFilter = 'all' | 'loaded' | 'available';

/** Sort options for model list. */
export type ModelSortBy = 'name' | 'size' | 'backend' | 'memory';
export type ModelSortOrder = 'asc' | 'desc';

/** Action state for individual model operations. */
export interface ModelActionState {
  /** Whether the model is currently being loaded */
  isLoading: boolean;
  /** Whether the model is currently being unloaded */
  isUnloading: boolean;
  /** Error message if the last operation failed */
  error: string | null;
  /** Timestamp when the action started */
  startedAt: Date | null;
}

/** Options for useModels hook. */
export interface UseModelsOptions {
  /** Auto-fetch models on mount (default: true) */
  autoFetch?: boolean;
  /** Enable WebSocket updates (default: true) */
  enableWebSocket?: boolean;
  /** Initial filter (default: 'all') */
  initialFilter?: ModelFilter;
  /** Initial sort field (default: 'name') */
  initialSortBy?: ModelSortBy;
  /** Initial sort order (default: 'asc') */
  initialSortOrder?: ModelSortOrder;
  /** Callback when models are loaded */
  onModelsChange?: (models: ModelInfo[]) => void;
  /** Callback when a model is loaded successfully */
  onModelLoaded?: (model: ModelInfo, loadTime: number) => void;
  /** Callback when a model is unloaded successfully */
  onModelUnloaded?: (name: string, memoryFreed: number) => void;
  /** Callback when a load/unload operation fails */
  onOperationError?: (name: string, error: Error) => void;
}

/** Model statistics. */
export interface ModelStats {
  /** Total number of models */
  total: number;
  /** Number of loaded models */
  loaded: number;
  /** Number of available (not loaded) models */
  available: number;
  /** Total memory used by loaded models (bytes) */
  totalMemoryUsed: number;
  /** Total size of all models (bytes) */
  totalSize: number;
  /** Models grouped by backend */
  byBackend: Record<string, number>;
}

/** Return type for useModels hook. */
export interface UseModelsReturn {
  /** All models */
  models: ModelInfo[];
  /** Filtered and sorted models */
  filteredModels: ModelInfo[];
  /** Current filter */
  filter: ModelFilter;
  /** Set the filter */
  setFilter: (filter: ModelFilter) => void;
  /** Current sort field */
  sortBy: ModelSortBy;
  /** Current sort order */
  sortOrder: ModelSortOrder;
  /** Set sort options */
  setSort: (sortBy: ModelSortBy, sortOrder?: ModelSortOrder) => void;
  /** Search query */
  searchQuery: string;
  /** Set search query */
  setSearchQuery: (query: string) => void;
  /** Whether models are loading (initial fetch) */
  isLoading: boolean;
  /** Whether models are refreshing */
  isRefreshing: boolean;
  /** Error from last fetch */
  error: string | null;
  /** Fetch/refresh models */
  refresh: (showLoading?: boolean) => Promise<void>;
  /** Get a specific model by name */
  getModel: (name: string) => ModelInfo | undefined;
  /** Load a model */
  load: (name: string, options?: ModelLoadOptions) => Promise<void>;
  /** Unload a model */
  unload: (name: string) => Promise<void>;
  /** Get action state for a model */
  getActionState: (name: string) => ModelActionState;
  /** Clear action error for a model */
  clearError: (name: string) => void;
  /** Model statistics */
  stats: ModelStats;
  /** Whether any models are currently being loaded/unloaded */
  hasActiveOperations: boolean;
  /** List of models with active operations */
  activeOperations: string[];
}

/** Default action state. */
const DEFAULT_ACTION_STATE: ModelActionState = {
  isLoading: false,
  isUnloading: false,
  error: null,
  startedAt: null,
};

/**
 * Sort models based on sort options.
 */
function sortModels(
  models: ModelInfo[],
  sortBy: ModelSortBy,
  sortOrder: ModelSortOrder
): ModelInfo[] {
  return [...models].sort((a, b) => {
    let comparison = 0;

    switch (sortBy) {
      case 'name':
        comparison = a.name.localeCompare(b.name);
        break;
      case 'size':
        comparison = (a.size || 0) - (b.size || 0);
        break;
      case 'backend':
        comparison = a.backend.localeCompare(b.backend);
        break;
      case 'memory':
        comparison = (a.memoryUsed || 0) - (b.memoryUsed || 0);
        break;
      default:
        comparison = a.name.localeCompare(b.name);
    }

    return sortOrder === 'asc' ? comparison : -comparison;
  });
}

/**
 * Filter models based on filter option and search query.
 */
function filterModels(
  models: ModelInfo[],
  filter: ModelFilter,
  searchQuery: string
): ModelInfo[] {
  let filtered = models;

  // Apply filter
  if (filter === 'loaded') {
    filtered = filtered.filter((m) => m.loaded);
  } else if (filter === 'available') {
    filtered = filtered.filter((m) => !m.loaded);
  }

  // Apply search
  if (searchQuery.trim()) {
    const query = searchQuery.toLowerCase().trim();
    filtered = filtered.filter(
      (m) =>
        m.name.toLowerCase().includes(query) ||
        m.backend.toLowerCase().includes(query) ||
        (m.quantization && m.quantization.toLowerCase().includes(query)) ||
        (m.path && m.path.toLowerCase().includes(query))
    );
  }

  return filtered;
}

/**
 * Compute model statistics.
 */
function computeStats(models: ModelInfo[]): ModelStats {
  const loaded = models.filter((m) => m.loaded);
  const byBackend: Record<string, number> = {};

  for (const model of models) {
    byBackend[model.backend] = (byBackend[model.backend] || 0) + 1;
  }

  return {
    total: models.length,
    loaded: loaded.length,
    available: models.length - loaded.length,
    totalMemoryUsed: loaded.reduce((sum, m) => sum + (m.memoryUsed || 0), 0),
    totalSize: models.reduce((sum, m) => sum + (m.size || 0), 0),
    byBackend,
  };
}

/**
 * Hook for managing models with load/unload operations.
 *
 * @param options - Hook configuration options
 * @returns Model data and control functions
 *
 * @example
 * ```tsx
 * function ModelBrowser() {
 *   const {
 *     filteredModels,
 *     filter,
 *     setFilter,
 *     load,
 *     unload,
 *     getActionState,
 *     stats,
 *   } = useModels({
 *     onModelLoaded: (model, time) => console.log(`Loaded ${model.name} in ${time}ms`),
 *   });
 *
 *   return (
 *     <div>
 *       <p>{stats.loaded} / {stats.total} models loaded</p>
 *       {filteredModels.map((model) => (
 *         <ModelCard
 *           key={model.name}
 *           model={model}
 *           onLoad={() => load(model.name)}
 *           onUnload={() => unload(model.name)}
 *           {...getActionState(model.name)}
 *         />
 *       ))}
 *     </div>
 *   );
 * }
 * ```
 */
export function useModels(options: UseModelsOptions = {}): UseModelsReturn {
  const {
    autoFetch = true,
    enableWebSocket = true,
    initialFilter = 'all',
    initialSortBy = 'name',
    initialSortOrder = 'asc',
    onModelsChange,
    onModelLoaded,
    onModelUnloaded,
    onOperationError,
  } = options;

  // State
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [filter, setFilter] = useState<ModelFilter>(initialFilter);
  const [sortBy, setSortBy] = useState<ModelSortBy>(initialSortBy);
  const [sortOrder, setSortOrder] = useState<ModelSortOrder>(initialSortOrder);
  const [searchQuery, setSearchQuery] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [actionStates, setActionStates] = useState<Record<string, ModelActionState>>({});

  // Refs for callbacks
  const onModelsChangeRef = useRef(onModelsChange);
  onModelsChangeRef.current = onModelsChange;

  const onModelLoadedRef = useRef(onModelLoaded);
  onModelLoadedRef.current = onModelLoaded;

  const onModelUnloadedRef = useRef(onModelUnloaded);
  onModelUnloadedRef.current = onModelUnloaded;

  const onOperationErrorRef = useRef(onOperationError);
  onOperationErrorRef.current = onOperationError;

  /**
   * Fetch models from API.
   */
  const refresh = useCallback(async (showLoading = true) => {
    if (showLoading) {
      setIsLoading(true);
    } else {
      setIsRefreshing(true);
    }
    setError(null);

    try {
      const result = await listModels();
      setModels(result);
      onModelsChangeRef.current?.(result);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load models';
      setError(message);
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  }, []);

  // Auto-fetch on mount
  useEffect(() => {
    if (autoFetch) {
      refresh();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Subscribe to WebSocket model status updates
  useWebSocketEvent<ModelInfo>(
    'model_status',
    (data) => {
      if (!enableWebSocket) return;

      setModels((prev) => {
        const existing = prev.find((m) => m.name === data.name);
        if (existing) {
          // Update existing model
          return prev.map((m) => (m.name === data.name ? { ...m, ...data } : m));
        } else {
          // Add new model
          return [...prev, data];
        }
      });
    },
    [enableWebSocket]
  );

  // Subscribe to status channel for model updates
  useEffect(() => {
    if (!enableWebSocket) return;

    const service = getWebSocketService();
    service.subscribe('status');

    return () => {
      service.unsubscribe('status');
    };
  }, [enableWebSocket]);

  /**
   * Get action state for a model.
   */
  const getActionState = useCallback(
    (name: string): ModelActionState => {
      return actionStates[name] ?? DEFAULT_ACTION_STATE;
    },
    [actionStates]
  );

  /**
   * Update action state for a model.
   */
  const updateActionState = useCallback(
    (name: string, update: Partial<ModelActionState>) => {
      setActionStates((prev) => ({
        ...prev,
        [name]: { ...(prev[name] ?? DEFAULT_ACTION_STATE), ...update },
      }));
    },
    []
  );

  /**
   * Clear action state for a model.
   */
  const clearActionState = useCallback((name: string) => {
    setActionStates((prev) => {
      const { [name]: _, ...rest } = prev;
      return rest;
    });
  }, []);

  /**
   * Clear error for a model.
   */
  const clearError = useCallback((name: string) => {
    updateActionState(name, { error: null });
  }, [updateActionState]);

  /**
   * Load a model.
   */
  const load = useCallback(
    async (name: string, loadOptions?: ModelLoadOptions): Promise<void> => {
      updateActionState(name, {
        isLoading: true,
        isUnloading: false,
        error: null,
        startedAt: new Date(),
      });

      try {
        const { model, loadTime } = await loadModel(name, loadOptions);

        // Update local state
        setModels((prev) =>
          prev.map((m) => (m.name === name ? { ...m, ...model, loaded: true } : m))
        );

        clearActionState(name);
        onModelLoadedRef.current?.(model, loadTime);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to load model';
        updateActionState(name, {
          isLoading: false,
          error: message,
          startedAt: null,
        });
        onOperationErrorRef.current?.(name, err instanceof Error ? err : new Error(message));
      }
    },
    [updateActionState, clearActionState]
  );

  /**
   * Unload a model.
   */
  const unload = useCallback(
    async (name: string): Promise<void> => {
      updateActionState(name, {
        isLoading: false,
        isUnloading: true,
        error: null,
        startedAt: new Date(),
      });

      try {
        const memoryFreed = await unloadModel(name);

        // Update local state
        setModels((prev) =>
          prev.map((m) =>
            m.name === name ? { ...m, loaded: false, memoryUsed: 0 } : m
          )
        );

        clearActionState(name);
        onModelUnloadedRef.current?.(name, memoryFreed);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to unload model';
        updateActionState(name, {
          isUnloading: false,
          error: message,
          startedAt: null,
        });
        onOperationErrorRef.current?.(name, err instanceof Error ? err : new Error(message));
      }
    },
    [updateActionState, clearActionState]
  );

  /**
   * Get a model by name.
   */
  const getModel = useCallback(
    (name: string): ModelInfo | undefined => {
      return models.find((m) => m.name === name);
    },
    [models]
  );

  /**
   * Set sort options.
   */
  const setSort = useCallback((newSortBy: ModelSortBy, newSortOrder?: ModelSortOrder) => {
    if (newSortOrder !== undefined) {
      setSortBy(newSortBy);
      setSortOrder(newSortOrder);
    } else if (newSortBy === sortBy) {
      // Toggle order if same field
      setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortBy(newSortBy);
      setSortOrder('asc');
    }
  }, [sortBy]);

  // Compute filtered and sorted models
  const filteredModels = useMemo(() => {
    const filtered = filterModels(models, filter, searchQuery);
    return sortModels(filtered, sortBy, sortOrder);
  }, [models, filter, searchQuery, sortBy, sortOrder]);

  // Compute stats
  const stats = useMemo(() => computeStats(models), [models]);

  // Compute active operations
  const activeOperations = useMemo(() => {
    return Object.entries(actionStates)
      .filter(([_, state]) => state.isLoading || state.isUnloading)
      .map(([name]) => name);
  }, [actionStates]);

  const hasActiveOperations = activeOperations.length > 0;

  return {
    models,
    filteredModels,
    filter,
    setFilter,
    sortBy,
    sortOrder,
    setSort,
    searchQuery,
    setSearchQuery,
    isLoading,
    isRefreshing,
    error,
    refresh,
    getModel,
    load,
    unload,
    getActionState,
    clearError,
    stats,
    hasActiveOperations,
    activeOperations,
  };
}

/**
 * Lightweight hook for just tracking model load operations.
 * Use when you only need to perform a single load/unload operation.
 *
 * @param modelName - Name of the model to track
 * @returns Loading state and control functions
 *
 * @example
 * ```tsx
 * function LoadButton({ modelName }: { modelName: string }) {
 *   const { isLoading, error, load, clearError } = useModelLoad(modelName);
 *
 *   return (
 *     <div>
 *       <button onClick={load} disabled={isLoading}>
 *         {isLoading ? 'Loading...' : 'Load'}
 *       </button>
 *       {error && <span>{error}</span>}
 *     </div>
 *   );
 * }
 * ```
 */
export function useModelLoad(modelName: string): {
  isLoading: boolean;
  error: string | null;
  load: (options?: ModelLoadOptions) => Promise<void>;
  clearError: () => void;
} {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (options?: ModelLoadOptions): Promise<void> => {
      setIsLoading(true);
      setError(null);

      try {
        await loadModel(modelName, options);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load model');
      } finally {
        setIsLoading(false);
      }
    },
    [modelName]
  );

  const clearError = useCallback(() => setError(null), []);

  return { isLoading, error, load, clearError };
}

/**
 * Lightweight hook for just tracking model unload operations.
 * Use when you only need to perform a single unload operation.
 *
 * @param modelName - Name of the model to track
 * @returns Unloading state and control functions
 */
export function useModelUnload(modelName: string): {
  isUnloading: boolean;
  error: string | null;
  unload: () => Promise<void>;
  clearError: () => void;
} {
  const [isUnloading, setIsUnloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const unload = useCallback(async (): Promise<void> => {
    setIsUnloading(true);
    setError(null);

    try {
      await unloadModel(modelName);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to unload model');
    } finally {
      setIsUnloading(false);
    }
  }, [modelName]);

  const clearError = useCallback(() => setError(null), []);

  return { isUnloading, error, unload, clearError };
}

/**
 * Hook for formatting model size values.
 * Provides consistent formatting utilities.
 */
export function useModelFormatters(): {
  formatSize: (bytes: number) => string;
  formatMemory: (bytes: number) => string;
  formatParameters: (params: number) => string;
} {
  return useMemo(
    () => ({
      formatSize: formatModelSize,
      formatMemory: formatModelSize,
      formatParameters: (params: number) => {
        if (params >= 1e12) return `${(params / 1e12).toFixed(1)}T`;
        if (params >= 1e9) return `${(params / 1e9).toFixed(1)}B`;
        if (params >= 1e6) return `${(params / 1e6).toFixed(1)}M`;
        return String(params);
      },
    }),
    []
  );
}

// Default export
export default useModels;
