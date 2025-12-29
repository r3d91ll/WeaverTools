/**
 * ConceptList component for displaying and managing concepts.
 *
 * Provides a list view of concepts with filtering, sorting, and actions.
 */
import { useState, useEffect, useCallback } from 'react';
import { ConceptCard } from './ConceptCard';
import { listConcepts, deleteConcept } from '@/services/conceptApi';
import type { ConceptStats } from '@/types/concept';

/**
 * Filter type for concept list.
 */
export type ConceptFilter = 'all' | 'healthy' | 'issues';

/**
 * Sort options for concept list.
 */
export type ConceptSortBy = 'name' | 'samples' | 'updated';
export type ConceptSortOrder = 'asc' | 'desc';

/**
 * ConceptList component props.
 */
export interface ConceptListProps {
  /** Initial filter to apply */
  initialFilter?: ConceptFilter;
  /** Maximum number of concepts to show (0 for unlimited) */
  limit?: number;
  /** Whether to show the header with filters */
  showHeader?: boolean;
  /** Callback when concept list changes */
  onConceptsChange?: (concepts: ConceptStats[]) => void;
  /** Callback when a concept is deleted */
  onConceptDeleted?: (name: string) => void;
  /** Callback when view details is requested */
  onViewDetails?: (name: string) => void;
  /** Whether to use compact card view */
  compact?: boolean;
  /** Title for the section */
  title?: string;
  /** Search query to filter concepts */
  searchQuery?: string;
}

/**
 * Loading state type.
 */
type LoadingState = 'idle' | 'loading' | 'refreshing';

/**
 * Sort concepts based on sort options.
 */
function sortConcepts(
  concepts: ConceptStats[],
  sortBy: ConceptSortBy,
  sortOrder: ConceptSortOrder
): ConceptStats[] {
  return [...concepts].sort((a, b) => {
    let comparison = 0;

    switch (sortBy) {
      case 'name':
        comparison = a.name.localeCompare(b.name);
        break;
      case 'samples':
        comparison = a.sampleCount - b.sampleCount;
        break;
      case 'updated':
        comparison = new Date(a.updatedAt).getTime() - new Date(b.updatedAt).getTime();
        break;
      default:
        comparison = a.name.localeCompare(b.name);
    }

    return sortOrder === 'asc' ? comparison : -comparison;
  });
}

/**
 * Filter concepts based on filter option and search query.
 */
function filterConcepts(
  concepts: ConceptStats[],
  filter: ConceptFilter,
  searchQuery?: string
): ConceptStats[] {
  let filtered = concepts;

  // Apply filter
  if (filter === 'healthy') {
    filtered = filtered.filter(
      (c) => !c.mismatchedIds || c.mismatchedIds.length === 0
    );
  } else if (filter === 'issues') {
    filtered = filtered.filter(
      (c) => c.mismatchedIds && c.mismatchedIds.length > 0
    );
  }

  // Apply search
  if (searchQuery && searchQuery.trim()) {
    const query = searchQuery.toLowerCase().trim();
    filtered = filtered.filter(
      (c) =>
        c.name.toLowerCase().includes(query) ||
        c.models?.some((m) => m.toLowerCase().includes(query))
    );
  }

  return filtered;
}

/**
 * Concept list component with filtering and management.
 */
export const ConceptList: React.FC<ConceptListProps> = ({
  initialFilter = 'all',
  limit = 0,
  showHeader = true,
  onConceptsChange,
  onConceptDeleted,
  onViewDetails,
  compact = false,
  title = 'Concepts',
  searchQuery: externalSearchQuery,
}) => {
  // State
  const [concepts, setConcepts] = useState<ConceptStats[]>([]);
  const [filter, setFilter] = useState<ConceptFilter>(initialFilter);
  const [sortBy, setSortBy] = useState<ConceptSortBy>('name');
  const [sortOrder, setSortOrder] = useState<ConceptSortOrder>('asc');
  const [loadingState, setLoadingState] = useState<LoadingState>('loading');
  const [error, setError] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [deletingConcept, setDeletingConcept] = useState<string | null>(null);

  // Use external search query if provided
  const effectiveSearchQuery = externalSearchQuery ?? searchQuery;

  /** Load concepts from API */
  const loadConcepts = useCallback(
    async (showLoading = true) => {
      if (showLoading) {
        setLoadingState('loading');
      } else {
        setLoadingState('refreshing');
      }
      setError(null);

      try {
        const result = await listConcepts();
        setConcepts(result);
        onConceptsChange?.(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load concepts');
      } finally {
        setLoadingState('idle');
      }
    },
    [onConceptsChange]
  );

  /** Load concepts on mount */
  useEffect(() => {
    loadConcepts();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /** Handle delete concept */
  const handleDelete = useCallback(
    async (name: string) => {
      setDeletingConcept(name);

      try {
        await deleteConcept(name);
        setConcepts((prev) => prev.filter((c) => c.name !== name));
        onConceptDeleted?.(name);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to delete concept');
      } finally {
        setDeletingConcept(null);
      }
    },
    [onConceptDeleted]
  );

  /** Handle filter change */
  const handleFilterChange = useCallback((newFilter: ConceptFilter) => {
    setFilter(newFilter);
  }, []);

  // Process concepts
  const filteredConcepts = filterConcepts(concepts, filter, effectiveSearchQuery);
  const sortedConcepts = sortConcepts(filteredConcepts, sortBy, sortOrder);
  const displayedConcepts = limit > 0 ? sortedConcepts.slice(0, limit) : sortedConcepts;

  // Calculate stats
  const totalConcepts = concepts.length;
  const healthyCount = concepts.filter(
    (c) => !c.mismatchedIds || c.mismatchedIds.length === 0
  ).length;
  const totalSamples = concepts.reduce((sum, c) => sum + c.sampleCount, 0);

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
              {healthyCount}/{totalConcepts} healthy ({totalSamples} total samples)
            </span>
          </div>

          <div className="flex items-center gap-3">
            {/* Search - only show if external search not provided */}
            {externalSearchQuery === undefined && (
              <input
                type="text"
                placeholder="Search concepts..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="input text-sm py-1.5 w-48"
              />
            )}

            {/* Filter Buttons */}
            <div className="flex bg-gray-100 rounded-lg p-1">
              {(['all', 'healthy', 'issues'] as ConceptFilter[]).map(
                (filterOption) => (
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
                      : filterOption === 'healthy'
                      ? 'Healthy'
                      : 'Issues'}
                  </button>
                )
              )}
            </div>

            {/* Sort Dropdown */}
            <select
              value={`${sortBy}_${sortOrder}`}
              onChange={(e) => {
                const [newSortBy, newSortOrder] = e.target.value.split('_') as [
                  ConceptSortBy,
                  ConceptSortOrder
                ];
                setSortBy(newSortBy);
                setSortOrder(newSortOrder);
              }}
              className="input text-sm py-1.5"
            >
              <option value="name_asc">Name A-Z</option>
              <option value="name_desc">Name Z-A</option>
              <option value="samples_desc">Most Samples</option>
              <option value="samples_asc">Fewest Samples</option>
              <option value="updated_desc">Recently Updated</option>
              <option value="updated_asc">Oldest First</option>
            </select>

            {/* Refresh Button */}
            <button
              type="button"
              onClick={() => loadConcepts(false)}
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
                onClick={() => loadConcepts()}
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
                <div className="h-8 w-24 bg-gray-200 rounded" />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Concept List */}
      {!isLoading && displayedConcepts.length > 0 && (
        <div className="space-y-4">
          {displayedConcepts.map((concept) => (
            <ConceptCard
              key={concept.name}
              concept={concept}
              onViewDetails={onViewDetails}
              onDelete={handleDelete}
              isDeleting={deletingConcept === concept.name}
              compact={compact}
            />
          ))}
        </div>
      )}

      {/* Empty State */}
      {!isLoading && displayedConcepts.length === 0 && (
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
              d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
            />
          </svg>
          <h3 className="mt-4 text-lg font-medium text-gray-900">
            {effectiveSearchQuery
              ? 'No concepts match your search'
              : filter === 'all'
              ? 'No concepts found'
              : filter === 'healthy'
              ? 'No healthy concepts'
              : 'No concepts with issues'}
          </h3>
          <p className="mt-2 text-sm text-gray-500 max-w-sm mx-auto">
            {effectiveSearchQuery
              ? 'Try adjusting your search terms or filters.'
              : filter === 'all'
              ? 'Extract samples from concepts using the CLI with "/extract <concept> <count>".'
              : filter === 'healthy'
              ? 'All concepts currently have dimension mismatches.'
              : 'All concepts have consistent dimensions.'}
          </p>
          {filter !== 'all' && (
            <button
              type="button"
              onClick={() => setFilter('all')}
              className="mt-4 btn-secondary"
            >
              Show All Concepts
            </button>
          )}
        </div>
      )}

      {/* Show more indicator */}
      {!isLoading && limit > 0 && sortedConcepts.length > limit && (
        <div className="text-center pt-2">
          <span className="text-sm text-gray-500">
            Showing {limit} of {sortedConcepts.length} concepts
          </span>
        </div>
      )}
    </div>
  );
};

export default ConceptList;
