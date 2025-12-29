/**
 * Models page component.
 * Displays model browser with load/unload capabilities.
 */

interface ModelsProps {
  className?: string;
}

export const Models: React.FC<ModelsProps> = ({ className }) => {
  return (
    <div className={className}>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Models</h1>
        <p className="mt-2 text-gray-600">
          Manage your AI models and view their status
        </p>
      </div>

      <div className="card mb-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-800">
            Available Models
          </h2>
          <button className="btn-secondary">Refresh</button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Placeholder Model Card */}
        <div className="card">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h3 className="font-semibold text-gray-800">No Models</h3>
              <p className="text-sm text-gray-500">Connect a backend to view models</p>
            </div>
            <span className="px-2 py-1 text-xs rounded-full bg-gray-100 text-gray-600">
              Offline
            </span>
          </div>
          <div className="space-y-2 text-sm text-gray-600">
            <div className="flex justify-between">
              <span>Memory:</span>
              <span>--</span>
            </div>
            <div className="flex justify-between">
              <span>Backend:</span>
              <span>--</span>
            </div>
          </div>
          <div className="mt-4 flex gap-2">
            <button className="btn-primary flex-1" disabled>
              Load
            </button>
            <button className="btn-secondary flex-1" disabled>
              Unload
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Models;
