/**
 * ModelList component for displaying and managing models.
 *
 * Provides a list view of models with filtering, sorting, and actions
 * for loading/unloading models.
 */
import { useState, useEffect, useCallback } from 'react';
import type { ModelInfo } from '@/services/modelApi';
import { listModels, loadModel, unloadModel, formatModelSize } from '@/services/modelApi';
import { ModelCard } from './ModelCard';

/**
 * Filter type for model list.
 */
export type ModelFilter = 'all' | 'loaded' | 'available';

/**
 * Sort options for model list.
 */
export type ModelSortBy = 'name' | 'size' | 'backend';
export type ModelSortOrder = 'asc' | 'desc';

/**
 * ModelList component props.
 */
export interface ModelListProps {
  /** Initial filter to apply */
  initialFilter?: ModelFilter;
  /** Maximum number of models to show (0 for unlimited) */
  limit?: number;
  /** Whether to show the header with filters */
  showHeader?: boolean;
  /** Callback when model list changes */
  onModelsChange?: (models: ModelInfo[]) => void;
  /** Callback when a model is loaded */
  onModelLoaded?: (model: ModelInfo) => void;
  /** Callback when a model is unloaded */
  onModelUnloaded?: (name: string) => void;
  /** Whether to use compact card view */
  compact?: boolean;
  /** Title for the section */
  title?: string;
  /** Search query to filter models */
  searchQuery?: string;
}

/**
 * Loading state type.
 */
type LoadingState = 'idle' | 'loading' | 'refreshing';

/**
 * Action state for individual models.
 */
interface ModelActionState {
  isLoading: boolean;
  isUnloading: boolean;
  error: string | null;
}

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
  searchQuery?: string
): ModelInfo[] {
  let filtered = models;

  // Apply filter
  if (filter === 'loaded') {
    filtered = filtered.filter((m) => m.loaded);
  } else if (filter === 'available') {
    filtered = filtered.filter((m) => !m.loaded);
  }

  // Apply search
  if (searchQuery && searchQuery.trim()) {
    const query = searchQuery.toLowerCase().trim();
    filtered = filtered.filter(
      (m) =>
        m.name.toLowerCase().includes(query) ||
        m.backend.toLowerCase().includes(query) ||
        (m.quantization && m.quantization.toLowerCase().includes(query))
    );
  }

  return filtered;
}

/**
 * Model list component with filtering and management.
 */
export const ModelList: React.FC<ModelListProps> = ({
  initialFilter = 'all',
  limit = 0,
  showHeader = true,
  onModelsChange,
  onModelLoaded,
  onModelUnloaded,
  compact = false,
  title = 'Available Models',
  searchQuery: externalSearchQuery,
}) => {
  // State
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [filter, setFilter] = useState<ModelFilter>(initialFilter);
  const [sortBy, setSortBy] = useState<ModelSortBy>('name');
  const [sortOrder, setSortOrder] = useState<ModelSortOrder>('asc');
  const [loadingState, setLoadingState] = useState<LoadingState>('loading');
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [actionStates, setActionStates] = useState<Record<string, ModelActionState>>({});

  // Use external search query if provided
  const effectiveSearchQuery = externalSearchQuery ?? searchQuery;

  /** Load models from API */
  const loadModels = useCallback(async (showLoading = true) => {
    if (showLoading) {
      setLoadingState('loading');
    } else {
      setLoadingState('refreshing');
    }
    setError(null);

    try {
      const result = await listModels();
      setModels(result);
      onModelsChange?.(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models');
    } finally {
      setLoadingState('idle');
    }
  }, [onModelsChange]);

  /** Load models on mount */
  useEffect(() => {
    loadModels();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /** Get action state for a model */
  const getActionState = useCallback(
    (name: string): ModelActionState => {
      return actionStates[name] ?? { isLoading: false, isUnloading: false, error: null };
    },
    [actionStates]
  );

  /** Update action state for a model */
  const updateActionState = useCallback(
    (name: string, update: Partial<ModelActionState>) => {
      setActionStates((prev) => ({
        ...prev,
        [name]: { ...getActionState(name), ...update },
      }));
    },
    [getActionState]
  );

  /** Clear action state for a model */
  const clearActionState = useCallback((name: string) => {
    setActionStates((prev) => {
      const { [name]: removed, ...rest } = prev;
      return rest;
    });
  }, []);

  /** Handle load model */
  const handleLoad = useCallback(
    async (name: string) => {
      updateActionState(name, { isLoading: true, error: null });

      try {
        const { model } = await loadModel(name);
        // Update local state
        setModels((prev) =>
          prev.map((m) => (m.name === name ? { ...m, ...model, loaded: true } : m))
        );
        clearActionState(name);
        onModelLoaded?.(model);
      } catch (err) {
        updateActionState(name, {
          isLoading: false,
          error: err instanceof Error ? err.message : 'Failed to load model',
        });
      }
    },
    [updateActionState, clearActionState, onModelLoaded]
  );

  /** Handle unload model */
  const handleUnload = useCallback(
    async (name: string) => {
      updateActionState(name, { isUnloading: true, error: null });

      try {
        await unloadModel(name);
        // Update local state
        setModels((prev) =>
          prev.map((m) =>
            m.name === name ? { ...m, loaded: false, memoryUsed: 0 } : m
          )
        );
        clearActionState(name);
        onModelUnloaded?.(name);
      } catch (err) {
        updateActionState(name, {
          isUnloading: false,
          error: err instanceof Error ? err.message : 'Failed to unload model',
        });
      }
    },
    [updateActionState, clearActionState, onModelUnloaded]
  );

  /** Handle filter change */
  const handleFilterChange = useCallback((newFilter: ModelFilter) => {
    setFilter(newFilter);
  }, []);

  /** Handle sort change */
  const handleSortChange = useCallback(
    (newSortBy: ModelSortBy) => {
      if (newSortBy === sortBy) {
        // Toggle order if same field
        setSortOrder((prev) => (prev === 'asc' ? 'desc' : 'asc'));
      } else {
        setSortBy(newSortBy);
        setSortOrder('asc');
      }
    },
    [sortBy]
  );

  // Process models
  const filteredModels = filterModels(models, filter, effectiveSearchQuery);
  const sortedModels = sortModels(filteredModels, sortBy, sortOrder);
  const displayedModels = limit > 0 ? sortedModels.slice(0, limit) : sortedModels;

  // Calculate stats
  const totalModels = models.length;
  const loadedCount = models.filter((m) => m.loaded).length;
  const totalMemoryUsed = models
    .filter((m) => m.loaded)
    .reduce((sum, m) => sum + (m.memoryUsed || 0), 0);

  const isLoading = loadingState === 'loading';
  const isRefreshing = loadingState === 'refreshing';

  return (
    <div className="space-y-4">
      {/* Header with filters and actions */}
      {showHeader && (
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center gap-4">
            <h2 className="text-xl font-semibold text-gray-900">{title}</h2>
            {isRefreshing && (
              <svg
                className="animate-spin h-4 w-4 text-weaver-600"
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
            )}
            <span className="text-sm text-gray-500">
              {loadedCount}/{totalModels} loaded
              {totalMemoryUsed > 0 && ` (${formatModelSize(totalMemoryUsed)} used)`}
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Search - only show if external search not provided */}
            {externalSearchQuery === undefined && (
              <input
                type="text"
                placeholder="Search models..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input text-sm py-1.5 w-48"
              />
            )}

            {/* Filter Buttons */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              {(['all', 'loaded', 'available'] as ModelFilter[]).map((filterOption) => (
                <button
                  key={filterOption}
                  type="button"
                  onClick={() => handleFilterChange(filterOption)}
                  className={`px-3 py-1 text-sm font-medium rounded-md transition-colors ${
                    filter === filterOption
                      ? 'bg-white text-gray-900 shadow-sm'
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  {filterOption === 'all'
                    ? 'All'
                    : filterOption === 'loaded'
                    ? 'Loaded'
                    : 'Available'}
                </button>
              ))}
            </div>

            {/* Sort Dropdown */}
            <select
              value={`${sortBy}_${sortOrder}`}
              onChange={(e) => {
                const [newSortBy, newSortOrder] = e.target.value.split('_') as [
                  ModelSortBy,
                  ModelSortOrder
                ];
                setSortBy(newSortBy);
                setSortOrder(newSortOrder);
              }}
              className="input text-sm py-1.5"
            >
              <option value="name_asc">Name A-Z</option>
              <option value="name_desc">Name Z-A</option>
              <option value="size_desc">Largest First</option>
              <option value="size_asc">Smallest First</option>
              <option value="backend_asc">Backend A-Z</option>
            </select>

            {/* Refresh Button */}
            <button
              type="button"
              onClick={() => loadModels(false)}
              disabled={isLoading}
              className="btn-secondary p-2"
              title="Refresh"
            >
              <svg
                className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`}
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
            </button>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex">
            <svg
              className="w-5 h-5 text-red-400 flex-shrink-0"
              viewBox="0 0 20 20"
              fill="currentColor"
            >
              <path
                fillRule="evenodd"
                d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
                clipRule="evenodd"
              />
            </svg>
            <div className="ml-3">
              <p className="text-sm font-medium text-red-800">{error}</p>
              <button
                type="button"
                onClick={() => loadModels()}
                className="mt-2 text-sm text-red-600 hover:text-red-700 underline"
              >
                Try again
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Loading State */}
      {isLoading && (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card animate-pulse">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="h-5 w-48 bg-gray-200 rounded" />
                  <div className="mt-2 h-4 w-72 bg-gray-100 rounded" />
                </div>
                <div className="h-6 w-20 bg-gray-100 rounded-full" />
              </div>
              <div className="mt-4 grid grid-cols-4 gap-4">
                <div className="h-4 w-16 bg-gray-100 rounded" />
                <div className="h-4 w-16 bg-gray-100 rounded" />
                <div className="h-4 w-16 bg-gray-100 rounded" />
                <div className="h-4 w-16 bg-gray-100 rounded" />
              </div>
              <div className="mt-4 pt-3 border-t border-gray-100 flex justify-end">
                <div className="h-8 w-20 bg-gray-200 rounded" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Model List */}
      {!isLoading && displayedModels.length > 0 && (
        <div className="space-y-4">
          {displayedModels.map((model) => {
            const actionState = getActionState(model.name);
            return (
              <ModelCard
                key={model.name}
                model={model}
                onLoad={handleLoad}
                onUnload={handleUnload}
                isLoading={actionState.isLoading}
                isUnloading={actionState.isUnloading}
                error={actionState.error}
                compact={compact}
              />
            );
          })}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && displayedModels.length === 0 && (
        <div className="card text-center py-12">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
            />
          </svg>
          <h3 className="mt-4 text-lg font-medium text-gray-900">
            {effectiveSearchQuery
              ? 'No models match your search'
              : filter === 'all'
              ? 'No models available'
              : filter === 'loaded'
              ? 'No models loaded'
              : 'No models available to load'}
          </h3>
          <p className="mt-2 text-sm text-gray-500 max-w-sm mx-auto">
            {effectiveSearchQuery
              ? 'Try adjusting your search terms or filters.'
              : filter === 'all'
              ? 'Connect to a backend to see available models. Ensure TheLoom is running.'
              : filter === 'loaded'
              ? 'Load a model to start using it for inference.'
              : 'All available models are currently loaded.'}
          </p>
          {filter !== 'all' && (
            <button
              type="button"
              onClick={() => setFilter('all')}
              className="mt-4 btn-secondary"
            >
              Show All Models
            </button>
          )}
        </div>
      )}

      {/* Show more indicator */}
      {!isLoading && limit > 0 && sortedModels.length > limit && (
        <div className="text-center pt-2">
          <span className="text-sm text-gray-500">
            Showing {limit} of {sortedModels.length} models
          </span>
        </div>
      )}
    </div>
  );
};

export default ModelList;
