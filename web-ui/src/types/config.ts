/**
 * Configuration types matching Weaver/pkg/config/config.go
 * Root configuration structure for Weaver.
 */

import type { AgentConfig } from './agent';

/** Valid measurement modes. */
export type MeasurementMode = 'active' | 'passive' | 'disabled';

/** ClaudeCodeConfig holds Claude Code backend settings. */
export interface ClaudeCodeConfig {
  enabled: boolean;
}

/** LoomConfig holds The Loom backend settings. */
export interface LoomConfig {
  enabled: boolean;
  url: string;
  /** Path to TheLoom directory (for auto-start) */
  path: string;
  /** Start TheLoom if not running */
  auto_start: boolean;
  /** Port for TheLoom server */
  port: number;
}

/** BackendsConfig holds backend settings. */
export interface BackendsConfig {
  claudecode: ClaudeCodeConfig;
  loom: LoomConfig;
}

/** SessionConfig holds session settings (from config file). */
export interface ConfigSessionSettings {
  measurement_mode: MeasurementMode;
  auto_export: boolean;
  export_path: string;
}

/** Config is the root configuration structure. */
export interface Config {
  backends: BackendsConfig;
  agents: Record<string, AgentConfig>;
  session: ConfigSessionSettings;
}

/** Default configuration values. */
export const DEFAULT_CONFIG: Config = {
  backends: {
    claudecode: {
      enabled: true,
    },
    loom: {
      enabled: true,
      url: 'http://localhost:8080',
      path: '../TheLoom/the-loom',
      auto_start: true,
      port: 8080,
    },
  },
  agents: {},
  session: {
    measurement_mode: 'active',
    auto_export: true,
    export_path: './experiments',
  },
};
