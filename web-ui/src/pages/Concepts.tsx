/**
 * Concepts page - concept management with Kakeya geometry analysis.
 *
 * Displays stored concepts with their samples and hidden state statistics.
 * Used for validating geometric signatures in the Kakeya framework.
 */
import { useState, useCallback } from 'react';
import type { ConceptStats } from '@/types/concept';
import { ConceptList } from '@/components/concepts';

/**
 * Concepts page component.
 */
export const Concepts: React.FC = () => {
  // State for search
  const [searchQuery, setSearchQuery] = useState('');

  // State for concept stats
  const [concepts, setConcepts] = useState<ConceptStats[]>([]);

  // Calculate stats from concepts
  const totalConcepts = concepts.length;
  const totalSamples = concepts.reduce((sum, c) => sum + c.sampleCount, 0);
  const healthyCount = concepts.filter(
    (c) => !c.mismatchedIds || c.mismatchedIds.length === 0
  ).length;

  // Get unique dimensions
  const uniqueDimensions = new Set(concepts.map((c) => c.dimension).filter((d) => d > 0));
  const dimensionDisplay =
    uniqueDimensions.size === 0
      ? 'N/A'
      : uniqueDimensions.size === 1
      ? `${Array.from(uniqueDimensions)[0]}`
      : `${uniqueDimensions.size} different`;

  // Handle concepts change
  const handleConceptsChange = useCallback((newConcepts: ConceptStats[]) => {
    setConcepts(newConcepts);
  }, []);

  // Handle concept deleted
  const handleConceptDeleted = useCallback((name: string) => {
    setConcepts((prev) => prev.filter((c) => c.name !== name));
  }, []);

  // Handle view details
  const handleViewDetails = useCallback((name: string) => {
    // TODO: Navigate to concept detail page when implemented
    // For now, we could show a modal or expand the card
  }, []);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Concepts</h1>
        <p className="mt-2 text-gray-600">
          Manage stored concepts and their hidden state samples for Kakeya analysis
        </p>
      </div>

      {/* Status Summary */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Total Concepts</h3>
          <p className="text-2xl font-bold text-weaver-600">{totalConcepts}</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Total Samples</h3>
          <p className="text-2xl font-bold text-weaver-600">{totalSamples}</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Healthy</h3>
          <p className="text-2xl font-bold text-green-600">
            {healthyCount}
            <span className="text-sm font-normal text-gray-500 ml-1">
              / {totalConcepts}
            </span>
          </p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Dimensions</h3>
          <p className="text-2xl font-bold text-weaver-600">{dimensionDisplay}</p>
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
              placeholder="Search concepts by name or model..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input pl-10 w-full"
            />
          </div>
        </div>
      </div>

      {/* Concepts List */}
      <ConceptList
        onConceptsChange={handleConceptsChange}
        onConceptDeleted={handleConceptDeleted}
        onViewDetails={handleViewDetails}
        searchQuery={searchQuery}
        showHeader={true}
        title="Stored Concepts"
      />

      {/* Info Panel */}
      <div className="card bg-blue-50 border-blue-200">
        <div className="flex items-start space-x-3">
          <svg
            className="w-5 h-5 text-blue-500 mt-0.5"
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
          <div>
            <h3 className="text-sm font-medium text-blue-800">
              About Concepts
            </h3>
            <p className="mt-1 text-sm text-blue-700">
              Concepts store hidden state samples extracted from language models.
              Each concept contains multiple samples that are used for Kakeya
              geometry analysis to validate geometric signatures. Use the CLI
              command <code className="bg-blue-100 px-1 rounded">/extract &lt;concept&gt; &lt;count&gt;</code> to
              add samples to a concept.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Concepts;
