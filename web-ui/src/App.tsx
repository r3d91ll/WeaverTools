import { BrowserRouter, Routes, Route } from 'react-router';
import { Layout } from './components/layout';
import { Dashboard } from './pages/Dashboard';
import { SessionDetail } from './pages/SessionDetail';
import { Config } from './pages/Config';
import { Chat } from './pages/Chat';
import { Metrics } from './pages/Metrics';
import { Models } from './pages/Models';

/**
 * Main application component with routing.
 *
 * Uses the Layout component to provide a consistent shell with:
 * - Header with logo and connection status
 * - Collapsible sidebar navigation
 * - Content area for page components
 * - Footer with platform info
 */
function App() {
  // TODO: Replace with actual connection status from WebSocket hook
  const connectionStatus = 'disconnected' as const;

  return (
    <BrowserRouter>
      <Layout connectionStatus={connectionStatus}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/sessions/:id" element={<SessionDetail />} />
          <Route path="/config" element={<Config />} />
          <Route path="/chat" element={<Chat />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/models" element={<Models />} />
        </Routes>
      </Layout>
    </BrowserRouter>
  );
}

export default App;
