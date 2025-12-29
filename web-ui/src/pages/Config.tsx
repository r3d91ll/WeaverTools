/**
 * Config page - configuration management and YAML editor.
 *
 * Provides form-based and raw YAML editing for Weaver configuration.
 */
export const Config: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Configuration</h1>
        <p className="mt-2 text-gray-600">
          Manage Weaver configuration, agents, and backend settings
        </p>
      </div>

      {/* Config Editor Tabs */}
      <div className="card">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8">
            <button className="border-b-2 border-weaver-500 py-4 px-1 text-sm font-medium text-weaver-600">
              Form Editor
            </button>
            <button className="border-b-2 border-transparent py-4 px-1 text-sm font-medium text-gray-500 hover:text-gray-700 hover:border-gray-300">
              YAML View
            </button>
          </nav>
        </div>
      </div>

      {/* Configuration Sections */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Agents Section */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Agents</h2>
          <p className="text-gray-500 text-sm mb-4">
            Configure AI agents with their roles, backends, and parameters.
          </p>
          <div className="border border-dashed border-gray-300 rounded-lg p-4 text-center">
            <p className="text-gray-400">Agent configuration form coming soon</p>
          </div>
        </div>

        {/* Backends Section */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Backends</h2>
          <p className="text-gray-500 text-sm mb-4">
            Configure backend connections (Claude Code, TheLoom, etc.).
          </p>
          <div className="border border-dashed border-gray-300 rounded-lg p-4 text-center">
            <p className="text-gray-400">Backend configuration form coming soon</p>
          </div>
        </div>

        {/* Session Settings */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Session Settings
          </h2>
          <p className="text-gray-500 text-sm mb-4">
            Configure session defaults and measurement modes.
          </p>
          <div className="border border-dashed border-gray-300 rounded-lg p-4 text-center">
            <p className="text-gray-400">Session settings form coming soon</p>
          </div>
        </div>

        {/* Save Actions */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Actions</h2>
          <div className="space-y-3">
            <button className="btn-primary w-full">Save Configuration</button>
            <button className="btn-secondary w-full">Validate</button>
            <button className="btn-secondary w-full">Reset to Defaults</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Config;
