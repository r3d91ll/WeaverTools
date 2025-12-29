/**
 * SessionDetail page - displays detailed view of a single session.
 *
 * Shows session metadata, conversations, messages, and measurements.
 * Supports actions like ending session and exporting data.
 */
import { useState, useEffect, useCallback } from 'react';
import { useParams, useNavigate, Link } from 'react-router';
import { ConversationList, MessageList } from '@/components/dashboard';
import {
  getSession,
  getSessionMessages,
  endSession,
  deleteSession,
  isSessionActive,
  getSessionDuration,
  formatDuration,
} from '@/services/sessionApi';
import type { Session, Message, Conversation } from '@/types';

/**
 * Format a date for display.
 */
function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString(undefined, {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * SessionDetail component - main page for viewing session details.
 */
export const SessionDetail: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  // State
  const [session, setSession] = useState<Session | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [selectedConversation, setSelectedConversation] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);

  /**
   * Fetch session data.
   */
  const fetchSession = useCallback(async () => {
    if (!id) {
      setError('No session ID provided');
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      // Fetch session and messages in parallel
      const [sessionData, messagesData] = await Promise.all([
        getSession(id),
        getSessionMessages(id),
      ]);

      setSession(sessionData);
      setMessages(messagesData.messages);

      // Group messages into conversations (by conversationId or simulate)
      const conversationMap = new Map<string, Message[]>();
      for (const msg of messagesData.messages) {
        const convId = (msg.metadata?.conversationId as string) || 'default';
        const existing = conversationMap.get(convId) || [];
        existing.push(msg);
        conversationMap.set(convId, existing);
      }

      // Convert to Conversation objects
      const convs: Conversation[] = Array.from(conversationMap.entries()).map(
        ([convId, msgs], index) => ({
          id: convId,
          name: convId === 'default' ? 'Main Conversation' : `Conversation ${index + 1}`,
          messages: msgs,
          participants: extractParticipants(msgs),
          createdAt: msgs[0]?.timestamp || new Date().toISOString(),
          updatedAt: msgs[msgs.length - 1]?.timestamp || new Date().toISOString(),
        })
      );

      setConversations(convs);

      // Select first conversation by default
      if (convs.length > 0 && !selectedConversation) {
        setSelectedConversation(convs[0].id);
      }
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : 'Failed to load session'
      );
    } finally {
      setLoading(false);
    }
  }, [id, selectedConversation]);

  /**
   * Extract participants from messages.
   */
  function extractParticipants(msgs: Message[]): Record<string, Conversation['participants'][string]> {
    const participants: Record<string, Conversation['participants'][string]> = {};

    for (const msg of msgs) {
      const agentId = msg.agentId || msg.role;
      if (!participants[agentId]) {
        participants[agentId] = {
          agentId,
          agentName: msg.agentName || msg.role,
          role: msg.role,
          joinedAt: msg.timestamp,
          messageCount: 0,
        };
      }
      participants[agentId].messageCount++;
    }

    return participants;
  }

  /**
   * Handle ending the session.
   */
  const handleEndSession = async () => {
    if (!id || !session || !isSessionActive(session)) return;

    try {
      setActionLoading(true);
      const updated = await endSession(id);
      setSession(updated);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to end session');
    } finally {
      setActionLoading(false);
    }
  };

  /**
   * Handle deleting the session.
   */
  const handleDeleteSession = async () => {
    if (!id) return;

    const confirmed = window.confirm(
      'Are you sure you want to delete this session? This action cannot be undone.'
    );
    if (!confirmed) return;

    try {
      setActionLoading(true);
      await deleteSession(id);
      navigate('/');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete session');
      setActionLoading(false);
    }
  };

  // Fetch data on mount or when ID changes
  useEffect(() => {
    fetchSession();
  }, [fetchSession]);

  // Get current conversation's messages
  const currentMessages = selectedConversation
    ? conversations.find((c) => c.id === selectedConversation)?.messages || []
    : messages;

  // Calculate session status
  const isActive = session ? isSessionActive(session) : false;
  const duration = session ? getSessionDuration(session) : null;

  // Loading state
  if (loading) {
    return (
      <div className="space-y-6 animate-pulse">
        <div className="h-8 bg-gray-200 rounded w-1/4" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-1">
            <div className="card h-64 bg-gray-100" />
          </div>
          <div className="lg:col-span-2">
            <div className="card h-96 bg-gray-100" />
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Link to="/" className="text-weaver-600 hover:text-weaver-700">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">Session Details</h1>
        </div>
        <div className="card bg-red-50 border-red-200">
          <div className="flex items-center gap-3">
            <svg className="w-6 h-6 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
              />
            </svg>
            <div>
              <p className="font-medium text-red-800">{error}</p>
              <button
                onClick={fetchSession}
                className="text-sm text-red-600 hover:text-red-700 mt-1"
              >
                Try again
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Not found state
  if (!session) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-2">
          <Link to="/" className="text-weaver-600 hover:text-weaver-700">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </Link>
          <h1 className="text-2xl font-bold text-gray-900">Session Not Found</h1>
        </div>
        <div className="card">
          <p className="text-gray-600">
            The session you're looking for doesn't exist or has been deleted.
          </p>
          <Link to="/" className="btn-primary mt-4 inline-block">
            Back to Dashboard
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <Link
            to="/"
            className="text-weaver-600 hover:text-weaver-700 p-1 rounded hover:bg-weaver-50"
            title="Back to Dashboard"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
            </svg>
          </Link>
          <div>
            <h1 className="text-2xl font-bold text-gray-900">{session.name}</h1>
            {session.description && (
              <p className="text-gray-600 mt-1">{session.description}</p>
            )}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
              isActive
                ? 'bg-green-100 text-green-800'
                : 'bg-gray-100 text-gray-600'
            }`}
          >
            {isActive ? (
              <>
                <span className="w-2 h-2 bg-green-500 rounded-full mr-2 animate-pulse" />
                Active
              </>
            ) : (
              'Completed'
            )}
          </span>
        </div>
      </div>

      {/* Session Info and Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Started</p>
          <p className="text-sm font-medium text-gray-900 mt-1">
            {formatDate(session.startedAt)}
          </p>
        </div>
        <div className="card">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Duration</p>
          <p className="text-sm font-medium text-gray-900 mt-1">
            {duration !== null ? formatDuration(duration) : 'In progress'}
          </p>
        </div>
        <div className="card">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Messages</p>
          <p className="text-lg font-semibold text-weaver-600 mt-1">
            {session.stats?.messageCount ?? messages.length}
          </p>
        </div>
        <div className="card">
          <p className="text-xs text-gray-500 uppercase tracking-wide">Measurements</p>
          <p className="text-lg font-semibold text-weaver-600 mt-1">
            {session.stats?.measurementCount ?? 0}
          </p>
        </div>
      </div>

      {/* Metrics Summary (if available) */}
      {session.stats && (session.stats.measurementCount ?? 0) > 0 && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Metrics Summary</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-gray-500">Avg D_eff</p>
              <p className="text-xl font-semibold text-weaver-600">
                {session.stats.avgDEff?.toFixed(3) ?? 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Avg Beta</p>
              <p className="text-xl font-semibold text-weaver-600">
                {session.stats.avgBeta?.toFixed(3) ?? 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Avg Alignment</p>
              <p className="text-xl font-semibold text-weaver-600">
                {session.stats.avgAlignment?.toFixed(3) ?? 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500">Bilateral</p>
              <p className="text-xl font-semibold text-weaver-600">
                {session.stats.bilateralCount ?? 0}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Main Content: Conversations and Messages */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Conversations List */}
        <div className="lg:col-span-1">
          <ConversationList
            conversations={conversations}
            selectedId={selectedConversation}
            onSelect={setSelectedConversation}
            loading={loading}
          />
        </div>

        {/* Messages List */}
        <div className="lg:col-span-2">
          <MessageList
            messages={currentMessages}
            loading={loading}
            conversationName={
              conversations.find((c) => c.id === selectedConversation)?.name || 'Messages'
            }
          />
        </div>
      </div>

      {/* Actions */}
      <div className="card bg-gray-50">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Actions</h2>
        <div className="flex flex-wrap gap-3">
          {isActive && (
            <button
              onClick={handleEndSession}
              disabled={actionLoading}
              className="btn-secondary flex items-center gap-2 disabled:opacity-50"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M9 10a1 1 0 011-1h4a1 1 0 011 1v4a1 1 0 01-1 1h-4a1 1 0 01-1-1v-4z"
                />
              </svg>
              End Session
            </button>
          )}
          <Link
            to={`/metrics?session=${id}`}
            className="btn-primary flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
              />
            </svg>
            View Metrics
          </Link>
          <button
            onClick={() => navigate('/export', { state: { sessionId: id } })}
            className="btn-secondary flex items-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
              />
            </svg>
            Export
          </button>
          <button
            onClick={handleDeleteSession}
            disabled={actionLoading}
            className="btn-secondary text-red-600 hover:bg-red-50 border-red-200 flex items-center gap-2 disabled:opacity-50"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
              />
            </svg>
            Delete
          </button>
        </div>
      </div>
    </div>
  );
};

export default SessionDetail;
