/**
 * ConversationList component - displays list of conversations in a session.
 *
 * Shows conversation summaries with participant count and message count.
 * Supports selection for viewing conversation details.
 */
import type { Conversation } from '@/types';

/**
 * ConversationList component props.
 */
export interface ConversationListProps {
  /** List of conversations to display */
  conversations: Conversation[];
  /** Currently selected conversation ID */
  selectedId: string | null;
  /** Callback when a conversation is selected */
  onSelect: (id: string) => void;
  /** Whether the list is loading */
  loading?: boolean;
}

/**
 * Format a relative time (e.g., "2 hours ago").
 */
function formatRelativeTime(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSeconds = Math.floor(diffMs / 1000);
  const diffMinutes = Math.floor(diffSeconds / 60);
  const diffHours = Math.floor(diffMinutes / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffDays > 0) {
    return diffDays === 1 ? '1 day ago' : `${diffDays} days ago`;
  }
  if (diffHours > 0) {
    return diffHours === 1 ? '1 hour ago' : `${diffHours} hours ago`;
  }
  if (diffMinutes > 0) {
    return diffMinutes === 1 ? '1 minute ago' : `${diffMinutes} minutes ago`;
  }
  return 'Just now';
}

/**
 * Get preview text from conversation.
 */
function getPreviewText(conversation: Conversation): string {
  if (conversation.messages.length === 0) {
    return 'No messages yet';
  }
  const lastMessage = conversation.messages[conversation.messages.length - 1];
  const content = lastMessage.content || '';
  return content.length > 50 ? `${content.substring(0, 50)}...` : content;
}

/**
 * Loading skeleton for conversation item.
 */
const ConversationSkeleton: React.FC = () => (
  <div className="p-3 animate-pulse">
    <div className="flex items-start justify-between mb-2">
      <div className="h-4 bg-gray-200 rounded w-2/3" />
      <div className="h-3 bg-gray-200 rounded w-16" />
    </div>
    <div className="h-3 bg-gray-200 rounded w-full mb-1" />
    <div className="h-3 bg-gray-200 rounded w-3/4" />
  </div>
);

/**
 * Empty state when no conversations exist.
 */
const EmptyState: React.FC = () => (
  <div className="p-6 text-center">
    <svg
      className="w-12 h-12 text-gray-300 mx-auto mb-3"
      fill="none"
      stroke="currentColor"
      viewBox="0 0 24 24"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth={1.5}
        d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
      />
    </svg>
    <p className="text-gray-500 text-sm">No conversations yet</p>
    <p className="text-gray-400 text-xs mt-1">
      Start a conversation to see it here
    </p>
  </div>
);

/**
 * Single conversation item component.
 */
interface ConversationItemProps {
  conversation: Conversation;
  selected: boolean;
  onSelect: () => void;
}

const ConversationItem: React.FC<ConversationItemProps> = ({
  conversation,
  selected,
  onSelect,
}) => {
  const participantCount = Object.keys(conversation.participants).length;
  const messageCount = conversation.messages.length;

  return (
    <button
      type="button"
      onClick={onSelect}
      className={`w-full text-left p-3 transition-colors ${
        selected
          ? 'bg-weaver-50 border-l-2 border-weaver-500'
          : 'hover:bg-gray-50 border-l-2 border-transparent'
      }`}
    >
      <div className="flex items-start justify-between mb-1">
        <h3
          className={`text-sm font-medium truncate ${
            selected ? 'text-weaver-700' : 'text-gray-900'
          }`}
        >
          {conversation.name}
        </h3>
        <span className="text-xs text-gray-400 flex-shrink-0 ml-2">
          {formatRelativeTime(conversation.updatedAt)}
        </span>
      </div>

      <p className="text-xs text-gray-500 truncate mb-2">
        {getPreviewText(conversation)}
      </p>

      <div className="flex items-center gap-3 text-xs text-gray-400">
        <span className="flex items-center gap-1">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
            />
          </svg>
          {participantCount}
        </span>
        <span className="flex items-center gap-1">
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
            />
          </svg>
          {messageCount}
        </span>
      </div>
    </button>
  );
};

/**
 * ConversationList component.
 */
export const ConversationList: React.FC<ConversationListProps> = ({
  conversations,
  selectedId,
  onSelect,
  loading = false,
}) => {
  return (
    <div className="card p-0 overflow-hidden">
      {/* Header */}
      <div className="p-4 border-b border-gray-100">
        <div className="flex items-center justify-between">
          <h2 className="font-semibold text-gray-900">Conversations</h2>
          <span className="text-xs text-gray-500">
            {conversations.length} total
          </span>
        </div>
      </div>

      {/* List */}
      <div className="divide-y divide-gray-100 max-h-[calc(100vh-400px)] overflow-y-auto scrollbar-thin">
        {loading ? (
          // Loading skeletons
          <>
            <ConversationSkeleton />
            <ConversationSkeleton />
            <ConversationSkeleton />
          </>
        ) : conversations.length === 0 ? (
          // Empty state
          <EmptyState />
        ) : (
          // Conversation items
          conversations.map((conversation) => (
            <ConversationItem
              key={conversation.id}
              conversation={conversation}
              selected={selectedId === conversation.id}
              onSelect={() => onSelect(conversation.id)}
            />
          ))
        )}
      </div>
    </div>
  );
};

export default ConversationList;
