/**
 * Dashboard page component.
 * Displays an overview of sessions, experiments, and quick actions.
 */

interface DashboardProps {
  className?: string;
}

export const Dashboard: React.FC<DashboardProps> = ({ className }) => {
  return (
    <div className={className}>
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
        <p className="mt-2 text-gray-600">
          Overview of your WeaverTools sessions and experiments
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {/* Sessions Overview Card */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Sessions</h2>
          <p className="text-gray-600 text-sm">
            Manage your conversation sessions and experiments.
          </p>
          <div className="mt-4">
            <span className="text-2xl font-bold text-weaver-600">0</span>
            <span className="text-gray-500 ml-2">active sessions</span>
          </div>
        </div>

        {/* Agents Overview Card */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Agents</h2>
          <p className="text-gray-600 text-sm">
            View and configure your AI agents.
          </p>
          <div className="mt-4">
            <span className="text-2xl font-bold text-weaver-600">0</span>
            <span className="text-gray-500 ml-2">configured agents</span>
          </div>
        </div>

        {/* Quick Actions Card */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-800 mb-4">Quick Actions</h2>
          <div className="space-y-2">
            <button className="btn-primary w-full">New Session</button>
            <button className="btn-secondary w-full">View Metrics</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
