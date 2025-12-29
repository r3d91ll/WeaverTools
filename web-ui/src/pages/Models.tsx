/**
 * Models page - model browser with load/unload capabilities.
 *
 * Displays available models, their status, and controls for loading/unloading.
 */
export const Models: React.FC = () => {
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
          <p className="text-2xl font-bold text-weaver-600">0</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Loaded</h3>
          <p className="text-2xl font-bold text-green-600">0</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">
            Memory Usage
          </h3>
          <p className="text-2xl font-bold text-weaver-600">0 GB</p>
        </div>
      </div>

      {/* Filters */}
      <div className="card">
        <div className="flex items-center space-x-4">
          <input
            type="text"
            placeholder="Search models..."
            className="input flex-1"
          />
          <select className="input w-48" defaultValue="all">
            <option value="all">All Models</option>
            <option value="loaded">Loaded</option>
            <option value="available">Available</option>
          </select>
          <button className="btn-secondary">Refresh</button>
        </div>
      </div>

      {/* Models List */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Available Models
        </h2>
        <div className="divide-y divide-gray-200">
          {/* Empty State */}
          <div className="py-12 text-center">
            <svg
              className="w-12 h-12 mx-auto text-gray-400 mb-4"
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
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              No Models Available
            </h3>
            <p className="text-gray-500 max-w-sm mx-auto">
              Connect to a backend to see available models. Ensure TheLoom
              is running and configured properly.
            </p>
            <button className="btn-primary mt-4">Check Connection</button>
          </div>
        </div>
      </div>

      {/* Backend Status */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Backend Status
        </h2>
        <div className="space-y-3">
          <div className="flex items-center justify-between py-2 border-b border-gray-100">
            <div className="flex items-center space-x-3">
              <span className="w-3 h-3 rounded-full bg-gray-300"></span>
              <span className="font-medium">TheLoom</span>
            </div>
            <span className="text-sm text-gray-500">Not connected</span>
          </div>
          <div className="flex items-center justify-between py-2 border-b border-gray-100">
            <div className="flex items-center space-x-3">
              <span className="w-3 h-3 rounded-full bg-gray-300"></span>
              <span className="font-medium">Claude Code</span>
            </div>
            <span className="text-sm text-gray-500">Not connected</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Models;
