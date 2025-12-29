/**
 * SessionStats component - displays aggregate statistics for sessions.
 *
 * Shows summary cards for active sessions, total agents, models loaded,
 * and other key metrics for the dashboard overview.
 */
import { useState, useEffect } from 'react';
import { listSessions, type SessionSummary } from '@/services/sessionApi';

/**
 * SessionStats component props.
 */
export interface SessionStatsProps {
  /** Pre-loaded sessions (optional) */
  sessions?: SessionSummary[];
  /** Whether to show loading state */
  loading?: boolean;
  /** Additional agent count from API */
  agentCount?: number;
  /** Additional model count from API */
  modelCount?: number;
}

/**
 * Calculated stats from sessions.
 */
interface Stats {
  activeSessions: number;
  totalSessions: number;
  totalMessages: number;
  totalMeasurements: number;
  avgDEff: number | null;
  avgBeta: number | null;
}

/**
 * Calculate stats from sessions array.
 */
function calculateStats(sessions: SessionSummary[]): Stats {
  const activeSessions = sessions.filter((s) => s.isActive).length;
  let totalMessages = 0;
  let totalMeasurements = 0;
  let dEffSum = 0;
  let betaSum = 0;
  let measurementSessionCount = 0;

  for (const session of sessions) {
    if (session.stats) {
      totalMessages += session.stats.messageCount ?? 0;
      totalMeasurements += session.stats.measurementCount ?? 0;

      if ((session.stats.measurementCount ?? 0) > 0) {
        dEffSum += session.stats.avgDEff ?? 0;
        betaSum += session.stats.avgBeta ?? 0;
        measurementSessionCount++;
      }
    }
  }

  return {
    activeSessions,
    totalSessions: sessions.length,
    totalMessages,
    totalMeasurements,
    avgDEff: measurementSessionCount > 0 ? dEffSum / measurementSessionCount : null,
    avgBeta: measurementSessionCount > 0 ? betaSum / measurementSessionCount : null,
  };
}

/**
 * Single stat card component.
 */
interface StatCardProps {
  label: string;
  value: string | number;
  subvalue?: string;
  icon?: React.ReactNode;
  color?: 'default' | 'green' | 'blue' | 'purple';
}

const StatCard: React.FC<StatCardProps> = ({
  label,
  value,
  subvalue,
  icon,
  color = 'default',
}) => {
  const colorClasses = {
    default: 'text-weaver-600',
    green: 'text-green-600',
    blue: 'text-blue-600',
    purple: 'text-purple-600',
  };

  return (
    <div className="card">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="text-sm font-medium text-gray-500">{label}</h3>
          <p className={`text-2xl font-bold ${colorClasses[color]}`}>{value}</p>
          {subvalue && (
            <p className="text-xs text-gray-400 mt-1">{subvalue}</p>
          )}
        </div>
        {icon && (
          <div className={`p-2 rounded-lg bg-gray-50 ${colorClasses[color]}`}>
            {icon}
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Session stats component for dashboard.
 */
export const SessionStats: React.FC<SessionStatsProps> = ({
  sessions: propSessions,
  loading: propLoading,
  agentCount = 0,
  modelCount = 0,
}) => {
  const [sessions, setSessions] = useState<SessionSummary[]>(propSessions ?? []);
  const [loading, setLoading] = useState(propLoading ?? !propSessions);
  const [error, setError] = useState<string | null>(null);

  // Fetch sessions if not provided
  useEffect(() => {
    if (propSessions) {
      setSessions(propSessions);
      setLoading(false);
      return;
    }

    const fetchSessions = async () => {
      setLoading(true);
      setError(null);
      try {
        const result = await listSessions({ limit: 100 });
        setSessions(result.sessions);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load sessions');
      } finally {
        setLoading(false);
      }
    };

    fetchSessions();
  }, [propSessions]);

  // Update loading state from props
  useEffect(() => {
    if (propLoading !== undefined) {
      setLoading(propLoading);
    }
  }, [propLoading]);

  const stats = calculateStats(sessions);

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[1, 2, 3].map((i) => (
          <div key={i} className="card animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-24 mb-2" />
            <div className="h-8 bg-gray-200 rounded w-16" />
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <div className="card bg-red-50 border-red-200">
        <div className="flex items-center gap-2 text-red-600">
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <span className="text-sm">{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <StatCard
        label="Active Sessions"
        value={stats.activeSessions}
        subvalue={`${stats.totalSessions} total`}
        color="green"
        icon={
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
        }
      />
      <StatCard
        label="Total Agents"
        value={agentCount}
        subvalue={`${stats.totalMessages} messages`}
        color="blue"
        icon={
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
            />
          </svg>
        }
      />
      <StatCard
        label="Measurements"
        value={stats.totalMeasurements}
        subvalue={
          stats.avgBeta !== null
            ? `Avg beta: ${stats.avgBeta.toFixed(3)}`
            : undefined
        }
        color="purple"
        icon={
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
        }
      />
    </div>
  );
};

export default SessionStats;
