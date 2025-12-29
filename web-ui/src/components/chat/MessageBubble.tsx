/**
 * MessageBubble component - displays a single chat message.
 *
 * Renders messages with role-based styling and supports streaming state.
 */
import type { Message, MessageRole } from '@/types';

/**
 * MessageBubble component props.
 */
export interface MessageBubbleProps {
  /** Message to display */
  message: Message;
  /** Whether the message is currently streaming */
  isStreaming?: boolean;
  /** Whether to show the timestamp */
  showTimestamp?: boolean;
  /** Whether to show the agent name */
  showAgent?: boolean;
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
 * Format full date and time for tooltip.
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
 * Get bubble styling based on message role.
 */
function getBubbleStyles(role: MessageRole): {
  container: string;
  bubble: string;
  roleColor: string;
} {
  switch (role) {
    case 'user':
      return {
        container: 'ml-auto',
        bubble: 'bg-weaver-600 text-white',
        roleColor: 'text-weaver-200',
      };
    case 'assistant':
      return {
        container: 'mr-auto',
        bubble: 'bg-white border border-gray-200 text-gray-900',
        roleColor: 'text-gray-500',
      };
    case 'system':
      return {
        container: 'mx-auto',
        bubble: 'bg-gray-100 text-gray-700 text-sm italic',
        roleColor: 'text-gray-400',
      };
    case 'tool':
      return {
        container: 'mr-auto',
        bubble: 'bg-purple-50 border border-purple-200 text-purple-900',
        roleColor: 'text-purple-500',
      };
    default:
      return {
        container: 'mr-auto',
        bubble: 'bg-gray-50 border border-gray-200 text-gray-900',
        roleColor: 'text-gray-500',
      };
  }
}

/**
 * Streaming cursor component.
 */
const StreamingCursor: React.FC = () => (
  <span className="inline-flex ml-1">
    <span className="w-2 h-4 bg-current opacity-75 animate-pulse" />
  </span>
);

/**
 * MessageBubble component for displaying individual chat messages.
 */
export const MessageBubble: React.FC<MessageBubbleProps> = ({
  message,
  isStreaming = false,
  showTimestamp = true,
  showAgent = true,
}) => {
  const styles = getBubbleStyles(message.role);
  const isSystemOrTool = message.role === 'system' || message.role === 'tool';

  return (
    <div className={`flex flex-col ${styles.container} max-w-[80%]`}>
      {/* Header with agent/role info */}
      <div
        className={`flex items-center gap-2 mb-1 text-xs ${
          isSystemOrTool ? 'justify-center' : message.role === 'user' ? 'justify-end' : 'justify-start'
        }`}
      >
        {showAgent && message.agentName && message.role === 'assistant' && (
          <span className="font-medium text-gray-600">
            {message.agentName}
          </span>
        )}
        {message.role === 'user' && (
          <span className="font-medium text-gray-600">You</span>
        )}
        {showTimestamp && (
          <span
            className="text-gray-400"
            title={formatFullDateTime(message.timestamp)}
          >
            {formatTimestamp(message.timestamp)}
          </span>
        )}
      </div>

      {/* Message bubble */}
      <div
        className={`rounded-2xl px-4 py-3 ${styles.bubble} ${
          message.role === 'user' ? 'rounded-tr-sm' : 'rounded-tl-sm'
        }`}
      >
        <div className="whitespace-pre-wrap break-words">
          {message.content}
          {isStreaming && <StreamingCursor />}
        </div>

        {/* Tool info if present */}
        {message.toolName && (
          <div className="mt-2 pt-2 border-t border-current/10 text-xs opacity-75">
            <span className="font-mono">Tool: {message.toolName}</span>
            {message.toolCallId && (
              <span className="ml-2 font-mono opacity-50">
                ID: {message.toolCallId.slice(0, 8)}...
              </span>
            )}
          </div>
        )}

        {/* Hidden state indicator */}
        {message.hiddenState && (
          <div className="mt-2 pt-2 border-t border-current/10 text-xs flex items-center gap-1 opacity-75">
            <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
              <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
              <path
                fillRule="evenodd"
                d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                clipRule="evenodd"
              />
            </svg>
            Hidden state (layer {message.hiddenState.layer})
          </div>
        )}
      </div>
    </div>
  );
};

export default MessageBubble;
