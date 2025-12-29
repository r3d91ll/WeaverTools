/**
 * Metrics page - real-time visualization of conveyance metrics.
 *
 * Displays D_eff, Beta, Alignment, and C_pair metrics with charts.
 */
export const Metrics: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Metrics</h1>
        <p className="mt-2 text-gray-600">
          Real-time conveyance metrics visualization (D_eff, Beta, Alignment, C_pair)
        </p>
      </div>

      {/* Metrics Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">D_eff (Effective Dimension)</h3>
          <p className="text-2xl font-bold text-weaver-600">--</p>
          <p className="text-xs text-gray-400 mt-1">No data</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Beta (Convergence)</h3>
          <p className="text-2xl font-bold text-weaver-600">--</p>
          <p className="text-xs text-gray-400 mt-1">No data</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">Alignment</h3>
          <p className="text-2xl font-bold text-weaver-600">--</p>
          <p className="text-xs text-gray-400 mt-1">No data</p>
        </div>
        <div className="card">
          <h3 className="text-sm font-medium text-gray-500">C_pair (Cross-pair)</h3>
          <p className="text-2xl font-bold text-weaver-600">--</p>
          <p className="text-xs text-gray-400 mt-1">No data</p>
        </div>
      </div>

      {/* Chart Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* D_eff Chart */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Effective Dimension (D_eff)
          </h2>
          <div className="h-64 border border-dashed border-gray-300 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">D_eff chart will render here</p>
          </div>
        </div>

        {/* Beta Chart */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Convergence (Beta)
          </h2>
          <div className="h-64 border border-dashed border-gray-300 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">Beta chart will render here</p>
          </div>
        </div>

        {/* Alignment Chart */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Alignment
          </h2>
          <div className="h-64 border border-dashed border-gray-300 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">Alignment chart will render here</p>
          </div>
        </div>

        {/* C_pair Chart */}
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Cross-pair Correlation (C_pair)
          </h2>
          <div className="h-64 border border-dashed border-gray-300 rounded-lg flex items-center justify-center">
            <p className="text-gray-400">C_pair chart will render here</p>
          </div>
        </div>
      </div>

      {/* Session Selector */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Session Data
        </h2>
        <div className="flex items-center space-x-4">
          <select className="input flex-1" defaultValue="">
            <option value="" disabled>
              Select a session to view metrics
            </option>
          </select>
          <button className="btn-primary">Load Metrics</button>
        </div>
        <p className="text-sm text-gray-500 mt-2">
          Metrics update in real-time via WebSocket when a session is active.
        </p>
      </div>
    </div>
  );
};

export default Metrics;
