import { BrowserRouter, Routes, Route, Link, NavLink } from 'react-router';
import { Dashboard } from './pages/Dashboard';
import { Config } from './pages/Config';
import { Chat } from './pages/Chat';
import { Metrics } from './pages/Metrics';
import { Models } from './pages/Models';

/**
 * Navigation item configuration.
 */
interface NavItem {
  path: string;
  label: string;
}

const navItems: NavItem[] = [
  { path: '/', label: 'Dashboard' },
  { path: '/config', label: 'Config' },
  { path: '/chat', label: 'Chat' },
  { path: '/metrics', label: 'Metrics' },
  { path: '/models', label: 'Models' },
];

/**
 * Main application component with routing.
 */
function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 sticky top-0 z-10">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
              {/* Logo */}
              <Link to="/" className="flex items-center gap-2">
                <span className="text-2xl font-bold text-weaver-700">
                  WeaverTools
                </span>
              </Link>

              {/* Navigation */}
              <nav className="flex items-center gap-1">
                {navItems.map((item) => (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={({ isActive }) =>
                      `px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                        isActive
                          ? 'bg-weaver-100 text-weaver-700'
                          : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                ))}
              </nav>
            </div>
          </div>
        </header>

        {/* Main Content */}
        <main className="flex-1">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/config" element={<Config />} />
              <Route path="/chat" element={<Chat />} />
              <Route path="/metrics" element={<Metrics />} />
              <Route path="/models" element={<Models />} />
            </Routes>
          </div>
        </main>

        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 py-4">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <p className="text-center text-sm text-gray-500">
              WeaverTools - Multi-agent AI Research Platform
            </p>
          </div>
        </footer>
      </div>
    </BrowserRouter>
  );
}

export default App;
