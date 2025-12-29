import { Link } from 'react-router';

/**
 * Dashboard page - main landing page showing session list and stats.
 *
 * Displays quick stats, navigation links, and recent sessions list.
 */
export const Dashboard: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-2 text-gray-600">
          Welcome to WeaverTools - Multi-agent AI Research Platform
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Active Sessions</h3>
          <p className="text-2xl font-bold text-weaver-600">0</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Total Agents</h3>
          <p className="text-2xl font-bold text-weaver-600">0</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Models Loaded</h3>
          <p className="text-2xl font-bold text-weaver-600">0</p>
        </div>
      </div>

      {/* Quick Links */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Link
          to="/chat"
          className="card hover:shadow-md transition-shadow"
        >
          <h3 className="font-semibold text-gray-900">Chat</h3>
          <p className="text-sm text-gray-500">
            Interactive chat with AI agents
          </p>
        </Link>
        <Link
          to="/config"
          className="card hover:shadow-md transition-shadow"
        >
          <h3 className="font-semibold text-gray-900">Configuration</h3>
          <p className="text-sm text-gray-500">
            Manage agents and settings
          </p>
        </Link>
        <Link
          to="/metrics"
          className="card hover:shadow-md transition-shadow"
        >
          <h3 className="font-semibold text-gray-900">Metrics</h3>
          <p className="text-sm text-gray-500">
            View conveyance metrics
          </p>
        </Link>
        <Link
          to="/models"
          className="card hover:shadow-md transition-shadow"
        >
          <h3 className="font-semibold text-gray-900">Models</h3>
          <p className="text-sm text-gray-500">
            Manage loaded models
          </p>
        </Link>
      </div>

      {/* Sessions List Placeholder */}
      <div>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-gray-900">
            Recent Sessions
          </h2>
          <button className="btn-primary">New Session</button>
        </div>
        <div className="card">
          <p className="text-gray-500 text-center py-8">
            No sessions yet. Start a new session to begin.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
