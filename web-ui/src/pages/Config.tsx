/**
 * Config page component.
 * Provides configuration management UI for editing YAML configs.
 */

interface ConfigProps {
  className?: string;
}

export const Config: React.FC<ConfigProps> = ({ className }) => {
  return (
    <div className={className}>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Configuration</h1>
        <p className="mt-2 text-gray-600">
          Manage your WeaverTools configuration settings
        </p>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-lg font-semibold text-gray-800">
            Configuration Editor
          </h2>
          <div className="flex gap-2">
            <button className="btn-secondary">Reset</button>
            <button className="btn-primary">Save</button>
          </div>
        </div>

        <div className="border rounded-lg bg-gray-50 p-4 min-h-[400px]">
          <p className="text-gray-500 text-center mt-32">
            Configuration editor will be loaded here.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Config;
