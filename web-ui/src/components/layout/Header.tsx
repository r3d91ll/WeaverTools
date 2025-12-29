import { Link } from 'react-router';

/**
 * Connection status indicator type.
 */
export type ConnectionStatus = 'connected' | 'connecting' | 'disconnected';

/**
 * Header component props.
 */
export interface HeaderProps {
  /** Whether sidebar is collapsed */
  sidebarCollapsed?: boolean;
  /** Callback to toggle sidebar */
  onToggleSidebar?: () => void;
  /** Current connection status */
  connectionStatus?: ConnectionStatus;
}

/**
 * Get status color classes based on connection status.
 */
function getStatusColor(status: ConnectionStatus): string {
  switch (status) {
    case 'connected':
      return 'bg-green-500';
    case 'connecting':
      return 'bg-yellow-500 animate-pulse';
    case 'disconnected':
      return 'bg-red-500';
  }
}

/**
 * Get status text based on connection status.
 */
function getStatusText(status: ConnectionStatus): string {
  switch (status) {
    case 'connected':
      return 'Connected';
    case 'connecting':
      return 'Connecting...';
    case 'disconnected':
      return 'Disconnected';
  }
}

/**
 * Application header component with logo, connection status, and sidebar toggle.
 */
export const Header: React.FC<HeaderProps> = ({
  sidebarCollapsed = false,
  onToggleSidebar,
  connectionStatus = 'disconnected',
}) => {
  return (
    <header className="bg-white border-b border-gray-200 sticky top-0 z-20">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Left section: Menu toggle and Logo */}
          <div className="flex items-center gap-4">
            {/* Sidebar toggle button */}
            {onToggleSidebar && (
              <button
                type="button"
                onClick={onToggleSidebar}
                className="p-2 rounded-md text-gray-500 hover:text-gray-900 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-weaver-500"
                aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
              >
                <svg
                  className="w-6 h-6"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                  aria-hidden="true"
                >
                  {sidebarCollapsed ? (
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 6h16M4 12h16M4 18h16"
                    />
                  ) : (
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M4 6h16M4 12h10M4 18h16"
                    />
                  )}
                </svg>
              </button>
            )}

            {/* Logo */}
            <Link to="/" className="flex items-center gap-2">
              <span className="text-2xl font-bold text-weaver-700">
                WeaverTools
              </span>
            </Link>
          </div>

          {/* Right section: Status and actions */}
          <div className="flex items-center gap-4">
            {/* Connection status indicator */}
            <div className="flex items-center gap-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${getStatusColor(connectionStatus)}`}
                aria-hidden="true"
              />
              <span className="text-sm text-gray-600">
                {getStatusText(connectionStatus)}
              </span>
            </div>

            {/* Future: User menu, settings, etc. */}
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;
