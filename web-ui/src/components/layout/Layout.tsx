import { useState, useCallback } from 'react';
import { Header, ConnectionStatus } from './Header';
import { Sidebar, NavItem } from './Sidebar';

/**
 * Layout component props.
 */
export interface LayoutProps {
  /** Child content to render in main area */
  children: React.ReactNode;
  /** Current connection status */
  connectionStatus?: ConnectionStatus;
  /** Custom navigation items (optional) */
  navItems?: NavItem[];
  /** Initial sidebar collapsed state */
  defaultSidebarCollapsed?: boolean;
}

/**
 * Main layout component that provides the app shell with header, sidebar, and content area.
 *
 * Features:
 * - Collapsible sidebar with navigation
 * - Sticky header with connection status
 * - Responsive layout
 * - Footer with platform info
 */
export const Layout: React.FC<LayoutProps> = ({
  children,
  connectionStatus = 'disconnected',
  navItems,
  defaultSidebarCollapsed = false,
}) => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(defaultSidebarCollapsed);

  const handleToggleSidebar = useCallback(() => {
    setSidebarCollapsed((prev) => !prev);
  }, []);

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <Header
        sidebarCollapsed={sidebarCollapsed}
        onToggleSidebar={handleToggleSidebar}
        connectionStatus={connectionStatus}
      />

      {/* Main content area with sidebar */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <Sidebar collapsed={sidebarCollapsed} navItems={navItems} />

        {/* Content area */}
        <main className="flex-1 overflow-y-auto">
          <div className="p-6">
            {children}
          </div>
        </main>
      </div>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 py-3">
        <div className="px-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            WeaverTools - Multi-agent AI Research Platform
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Layout;
