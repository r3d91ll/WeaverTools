/**
 * Context exports for WeaverTools web-ui.
 *
 * These contexts provide global state management across the application:
 * - ConfigContext: Configuration state (load, save, validate)
 * - ConnectionContext: WebSocket connection state
 * - SessionContext: Session management state
 */

// Config context
export {
  ConfigProvider,
  useConfig,
  useAgentConfig,
  type ConfigContextValue,
  type ConfigProviderProps,
} from './ConfigContext';

// Connection context
export {
  ConnectionProvider,
  useConnection,
  useConnectionStatus,
  type ConnectionContextValue,
  type ConnectionProviderProps,
} from './ConnectionContext';

// Session context
export {
  SessionProvider,
  useSession,
  useCurrentSession,
  type SessionContextValue,
  type SessionProviderProps,
} from './SessionContext';
