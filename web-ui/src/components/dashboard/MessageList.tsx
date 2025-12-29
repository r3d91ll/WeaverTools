/**
 * MessageList component - displays messages from a session or conversation.
 *
 * Supports pagination, virtualization for large lists, and real-time updates.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import type { Message, MessageRole } from '@/types';

/**
 * MessageList component props.
 */
export interface MessageListProps {
  /** Messages to display */
  messages: Message[];
  /** Whether more messages are available */
  hasMore?: boolean;
  /** Callback to load more messages */
  onLoadMore?: () => void;
  /** Whether loading is in progress */
  loading?: boolean;
  /** Whether to auto-scroll to bottom on new messages */
  autoScroll?: boolean;
  /** Maximum height in pixels (enables scrolling) */
  maxHeight?: number;
  /** Whether to show timestamps */
  showTimestamps?: boolean;
  /** Whether to show agent badges */
  showAgentBadges?: boolean;
  /** Callback when a message is clicked */
  onMessageClick?: (message: Message) => void;
  /** Currently selected message ID */
  selectedMessageId?: string;
  /** Conversation/list name for header display */
  conversationName?: string;
  /** Whether to show the header */
  showHeader?: boolean;
}

/**
 * Format a timestamp for display.
 */
function formatTimestamp(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * Format a full date and time.
 */
function formatFullDateTime(timestamp: string): string {
  const date = new Date(timestamp);
  return date.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

/**
 * Get the role display color class.
 */
function getRoleColor(role: MessageRole): string {
  switch (role) {
    case 'user':
      return 'bg-blue-100 text-blue-800';
    case 'assistant':
      return 'bg-green-100 text-green-800';
    case 'system':
      return 'bg-gray-100 text-gray-800';
    case 'tool':
      return 'bg-purple-100 text-purple-800';
    default:
      return 'bg-gray-100 text-gray-600';
  }
}

/**
 * Get the message alignment class based on role.
 */
function getMessageAlignment(role: MessageRole): string {
  switch (role) {
    case 'user':
      return 'ml-auto';
    case 'assistant':
      return 'mr-auto';
    case 'system':
    case 'tool':
      return 'mx-auto';
    default:
      return '';
  }
}

/**
 * Get the message background class based on role.
 */
function getMessageBackground(role: MessageRole, isSelected: boolean): string {
  if (isSelected) {
    return 'bg-weaver-100 border-weaver-300';
  }
  switch (role) {
    case 'user':
      return 'bg-blue-50 border-blue-200';
    case 'assistant':
      return 'bg-green-50 border-green-200';
    case 'system':
      return 'bg-gray-50 border-gray-200';
    case 'tool':
      return 'bg-purple-50 border-purple-200';
    default:
      return 'bg-white border-gray-200';
  }
}

/**
 * Single message item component.
 */
interface MessageItemProps {
  message: Message;
  showTimestamp: boolean;
  showAgentBadge: boolean;
  isSelected: boolean;
  onClick?: (message: Message) => void;
}

const MessageItem: React.FC<MessageItemProps> = ({
  message,
  showTimestamp,
  showAgentBadge,
  isSelected,
  onClick,
}) => {
  const handleClick = () => {
    onClick?.(message);
  };

  const alignment = getMessageAlignment(message.role);
  const background = getMessageBackground(message.role, isSelected);
  const roleColor = getRoleColor(message.role);

  const isSystemOrTool = message.role === 'system' || message.role === 'tool';

  return (
    <div
      className={`flex flex-col ${alignment} max-w-[85%] ${onClick ? 'cursor-pointer' : ''}`}
      onClick={handleClick}
    >
      {/* Header with role/agent info */}
      <div className={`flex items-center gap-2 mb-1 ${isSystemOrTool ? 'justify-center' : ''}`}>
        <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${roleColor}`}>
          {message.role}
        </span>
        {showAgentBadge && message.agentName && (
          <span className="text-xs text-gray-500">
            {message.agentName}
          </span>
        )}
        {showTimestamp && (
          <span
            className="text-xs text-gray-400"
            title={formatFullDateTime(message.timestamp)}
          >
            {formatTimestamp(message.timestamp)}
          </span>
        )}
      </div>

      {/* Message content */}
      <div
        className={`rounded-lg border px-4 py-3 ${background} ${
          isSelected ? 'ring-2 ring-weaver-500' : ''
        }`}
      >
        <div className="text-sm text-gray-900 whitespace-pre-wrap break-words">
          {message.content}
        </div>

        {/* Tool info if present */}
        {message.toolName && (
          <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500">
            Tool: {message.toolName}
            {message.toolCallId && (
              <span className="ml-2">ID: {message.toolCallId}</span>
            )}
          </div>
        )}

        {/* Hidden state indicator if present */}
        {message.hiddenState && (
          <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-weaver-600 flex items-center gap-1">
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
              <path fillRule="evenodd" d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z" clipRule="evenodd" />
            </svg>
            Hidden state captured (layer {message.hiddenState.layer})
          </div>
        )}
      </div>
    </div>
  );
};

/**
 * Loading skeleton for messages.
 */
const MessageSkeleton: React.FC = () => (
  <div className="space-y-4">
    {[...Array(3)].map((_, i) => (
      <div
        key={i}
        className={`flex flex-col ${i % 2 === 0 ? 'ml-auto' : 'mr-auto'} max-w-[85%]`}
      >
        <div className="flex items-center gap-2 mb-1">
          <div className="w-16 h-5 bg-gray-200 rounded animate-pulse" />
          <div className="w-12 h-4 bg-gray-100 rounded animate-pulse" />
        </div>
        <div className="rounded-lg border border-gray-200 px-4 py-3 bg-gray-50">
          <div className="space-y-2">
            <div className="h-4 bg-gray-200 rounded animate-pulse w-full" />
            <div className="h-4 bg-gray-200 rounded animate-pulse w-3/4" />
          </div>
        </div>
      </div>
    ))}
  </div>
);

/**
 * Empty state component.
 */
const EmptyState: React.FC = () => (
  <div className="flex flex-col items-center justify-center py-12 text-gray-500">
    <svg
      className="w-12 h-12 text-gray-300 mb-4"
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
    <p className="text-sm">No messages yet</p>
    <p className="text-xs text-gray-400 mt-1">Messages will appear here</p>
  </div>
);

/**
 * MessageList component displaying messages from a session or conversation.
 */
export const MessageList: React.FC<MessageListProps> = ({
  messages,
  hasMore = false,
  onLoadMore,
  loading = false,
  autoScroll = true,
  maxHeight,
  showTimestamps = true,
  showAgentBadges = true,
  onMessageClick,
  selectedMessageId,
  conversationName = 'Messages',
  showHeader = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const [shouldAutoScroll, setShouldAutoScroll] = useState(autoScroll);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (shouldAutoScroll && bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages.length, shouldAutoScroll]);

  // Detect scroll position to toggle auto-scroll
  const handleScroll = useCallback(() => {
    if (!containerRef.current) return;

    const { scrollTop, scrollHeight, clientHeight } = containerRef.current;
    const isAtBottom = scrollHeight - scrollTop - clientHeight < 100;

    if (autoScroll) {
      setShouldAutoScroll(isAtBottom);
    }
  }, [autoScroll]);

  // Infinite scroll: load more when reaching top
  const handleScrollTop = useCallback(() => {
    if (!containerRef.current || !hasMore || loading || !onLoadMore) return;

    const { scrollTop } = containerRef.current;
    if (scrollTop < 100) {
      onLoadMore();
    }
  }, [hasMore, loading, onLoadMore]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    container.addEventListener('scroll', handleScroll);
    container.addEventListener('scroll', handleScrollTop);

    return () => {
      container.removeEventListener('scroll', handleScroll);
      container.removeEventListener('scroll', handleScrollTop);
    };
  }, [handleScroll, handleScrollTop]);

  // Wrapper component with optional header
  const renderContent = (content: React.ReactNode) => (
    <div className="card p-0 overflow-hidden flex flex-col" style={maxHeight ? { height: maxHeight } : { height: 'calc(100vh - 400px)' }}>
      {showHeader && (
        <div className="p-4 border-b border-gray-100 flex-shrink-0">
          <div className="flex items-center justify-between">
            <h2 className="font-semibold text-gray-900">{conversationName}</h2>
            <span className="text-xs text-gray-500">
              {messages.length} message{messages.length !== 1 ? 's' : ''}
            </span>
          </div>
        </div>
      )}
      <div className="flex-1 overflow-y-auto scrollbar-thin">
        {content}
      </div>
    </div>
  );

  if (loading && messages.length === 0) {
    return renderContent(
      <div className="p-4">
        <MessageSkeleton />
      </div>
    );
  }

  if (messages.length === 0) {
    return renderContent(<EmptyState />);
  }

  return renderContent(
    <div
      ref={containerRef}
      className="p-4 space-y-4"
      onScroll={handleScroll}
    >
      {/* Load more indicator at top */}
      {hasMore && (
        <div className="text-center">
          {loading ? (
            <div className="flex items-center justify-center gap-2 text-sm text-gray-500">
              <svg
                className="animate-spin h-4 w-4 text-weaver-600"
                viewBox="0 0 24 24"
              >
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                  fill="none"
                />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Loading older messages...
            </div>
          ) : (
            <button
              onClick={onLoadMore}
              className="text-sm text-weaver-600 hover:text-weaver-700"
            >
              Load more messages
            </button>
          )}
        </div>
      )}

      {/* Messages */}
      {messages.map((message) => (
        <MessageItem
          key={message.id}
          message={message}
          showTimestamp={showTimestamps}
          showAgentBadge={showAgentBadges}
          isSelected={selectedMessageId === message.id}
          onClick={onMessageClick}
        />
      ))}

      {/* Scroll anchor */}
      <div ref={bottomRef} />

      {/* Auto-scroll indicator */}
      {!shouldAutoScroll && autoScroll && messages.length > 0 && (
        <button
          onClick={() => {
            setShouldAutoScroll(true);
            bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
          }}
          className="fixed bottom-24 right-8 bg-weaver-600 text-white px-3 py-2 rounded-full shadow-lg hover:bg-weaver-700 flex items-center gap-2 text-sm"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 14l-7 7m0 0l-7-7m7 7V3"
            />
          </svg>
          New messages
        </button>
      )}
    </div>
  );
};

export default MessageList;
