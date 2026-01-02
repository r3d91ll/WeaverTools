import { BrowserRouter, Routes, Route } from 'react-router';
import { Layout } from './components/layout';
import { Dashboard } from './pages/Dashboard';
import { SessionDetail } from './pages/SessionDetail';
import { Config } from './pages/Config';
import { Chat } from './pages/Chat';
import { Metrics } from './pages/Metrics';
import { Models } from './pages/Models';
import { Concepts } from './pages/Concepts';
import { Export } from './pages/Export';
import { Resources } from './pages/Resources';
import { AtlasDashboard } from './pages/AtlasDashboard';
import {
  ConfigProvider,
  ConnectionProvider,
  SessionProvider,
  useConnectionStatus,
} from './contexts';

/**
 * AppContent renders the main application content within context providers.
 * This is separated to allow access to the connection context for Layout.
 */
function AppContent(): React.ReactElement {
  const { status } = useConnectionStatus();

  return (
    <Layout connectionStatus={status}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/sessions/:id" element={<SessionDetail />} />
        <Route path="/config" element={<Config />} />
        <Route path="/chat" element={<Chat />} />
        <Route path="/metrics" element={<Metrics />} />
        <Route path="/models" element={<Models />} />
        <Route path="/concepts" element={<Concepts />} />
        <Route path="/export" element={<Export />} />
        <Route path="/resources" element={<Resources />} />
        <Route path="/atlas-dashboard" element={<AtlasDashboard />} />
      </Routes>
    </Layout>
  );
}

/**
 * Main application component with routing and context providers.
 *
 * Uses a layered context structure:
 * - ConnectionProvider: WebSocket connection state (outermost, required by other contexts)
 * - ConfigProvider: Configuration state management
 * - SessionProvider: Session state management (uses WebSocket for events)
 *
 * The Layout component provides a consistent shell with:
 * - Header with logo and connection status
 * - Collapsible sidebar navigation
 * - Content area for page components
 * - Footer with platform info
 */
function App(): React.ReactElement {
  return (
    <BrowserRouter>
      <ConnectionProvider autoConnect initialChannels={['measurements', 'messages', 'status', 'resources']}>
        <ConfigProvider autoLoad>
          <SessionProvider autoLoad>
            <AppContent />
          </SessionProvider>
        </ConfigProvider>
      </ConnectionProvider>
    </BrowserRouter>
  );
}

export default App;
