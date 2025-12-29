/**
 * ChatContainer component - main chat interface container.
 *
 * Manages chat state, message history, and coordinates between
 * AgentSelector, MessageBubble, and ChatInput components.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import type { Message, AgentInfo } from '@/types';
import { listAgents, chatStream } from '@/services/agentApi';
import { MessageBubble } from './MessageBubble';
import { ChatInput } from './ChatInput';
import { AgentSelector } from './AgentSelector';

/**
 * ChatContainer component props.
 */
export interface ChatContainerProps {
  /** Initial messages to display */
  initialMessages?: Message[];
  /** Initial selected agent */
  initialAgent?: string | null;
  /** Callback when a new message is sent */
  onMessageSent?: (message: Message) => void;
  /** Callback when a response is received */
  onResponseReceived?: (message: Message) => void;
  /** Callback when session should be created */
  onNewSession?: () => void;
  /** Whether to show the header */
  showHeader?: boolean;
  /** Session ID for the chat */
  sessionId?: string;
}

/**
 * Generate a unique message ID.
 */
function generateMessageId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Empty state component shown when there are no messages.
 */
const EmptyState: React.FC = () => (
  <div className="flex flex-col items-center justify-center py-16 text-center">
    <div className="text-gray-300 mb-4">
      <svg
        className="w-16 h-16 mx-auto"
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
    </div>
    <h3 className="text-lg font-medium text-gray-900 mb-2">
      Start a Conversation
    </h3>
    <p className="text-gray-500 max-w-sm">
      Send a message to begin. Use <code className="px-1 py-0.5 bg-gray-100 rounded text-sm">@agent</code> to
      target a specific agent, or select one from the dropdown above.
    </p>
  </div>
);

/**
 * ChatContainer component for the main chat interface.
 */
export const ChatContainer: React.FC<ChatContainerProps> = ({
  initialMessages = [],
  initialAgent = null,
  onMessageSent,
  onResponseReceived,
  onNewSession,
  showHeader = true,
  sessionId,
}) => {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(initialAgent);
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const abortControllerRef = useRef<AbortController | null>(null);

  // Fetch available agents
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const agentList = await listAgents();
        setAgents(agentList);
      } catch {
        // Silently fail - agents will be empty
      }
    };
    fetchAgents();
  }, []);

  // Scroll to bottom when messages change
  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  // Handle sending a message
  const handleSendMessage = useCallback(
    async (content: string, targetAgent: string | null) => {
      if (!content.trim()) return;

      // Require an agent to be selected
      if (!targetAgent) {
        setError('Please select an agent or use @agent syntax');
        return;
      }

      setError(null);
      const userMessageId = generateMessageId();
      const userMessage: Message = {
        id: userMessageId,
        role: 'user',
        content: content.trim(),
        timestamp: new Date().toISOString(),
        agentName: targetAgent,
      };

      // Add user message
      setMessages((prev) => [...prev, userMessage]);
      setInputValue('');
      onMessageSent?.(userMessage);

      // Create placeholder for assistant response
      const assistantMessageId = generateMessageId();
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        agentName: targetAgent,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setIsStreaming(true);
      setStreamingMessageId(assistantMessageId);

      // Create abort controller
      abortControllerRef.current = new AbortController();

      try {
        let fullContent = '';

        await chatStream(
          targetAgent,
          { message: content.trim() },
          // onChunk
          (chunk) => {
            fullContent += chunk;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, content: fullContent }
                  : msg
              )
            );
          },
          // onComplete
          (response) => {
            const finalMessage: Message = {
              id: assistantMessageId,
              role: 'assistant',
              content: response.content,
              timestamp: new Date().toISOString(),
              agentName: targetAgent,
              metadata: {
                model: response.model,
                latencyMs: response.latencyMs,
                finishReason: response.finishReason,
              },
            };

            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId ? finalMessage : msg
              )
            );
            onResponseReceived?.(finalMessage);
          },
          // onError
          (err) => {
            setError(err.message);
            // Remove the empty assistant message
            setMessages((prev) =>
              prev.filter((msg) => msg.id !== assistantMessageId)
            );
          },
          abortControllerRef.current.signal
        );
      } catch (err) {
        if (abortControllerRef.current?.signal.aborted) {
          // User stopped the stream
          return;
        }
        setError(err instanceof Error ? err.message : 'Failed to send message');
        // Remove the empty assistant message
        setMessages((prev) =>
          prev.filter((msg) => msg.id !== assistantMessageId)
        );
      } finally {
        setIsStreaming(false);
        setStreamingMessageId(null);
        abortControllerRef.current = null;
      }
    },
    [onMessageSent, onResponseReceived]
  );

  // Stop streaming
  const handleStopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
      setStreamingMessageId(null);
    }
  }, []);

  // Handle new session
  const handleNewSession = useCallback(() => {
    setMessages([]);
    setError(null);
    setInputValue('');
    onNewSession?.();
  }, [onNewSession]);

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      {showHeader && (
        <div className="flex items-center justify-between pb-4 border-b border-gray-200 flex-shrink-0">
          <div>
            <h1 className="text-xl font-bold text-gray-900">Chat</h1>
            <p className="text-sm text-gray-500">
              Use @agent syntax to target specific agents
            </p>
          </div>
          <div className="flex items-center gap-4">
            <AgentSelector
              selectedAgent={selectedAgent}
              onSelectAgent={setSelectedAgent}
            />
            <button
              type="button"
              onClick={handleNewSession}
              className="btn-secondary text-sm"
            >
              New Session
            </button>
          </div>
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto py-6 scrollbar-thin">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="space-y-4 px-2">
            {messages.map((message) => (
              <MessageBubble
                key={message.id}
                message={message}
                isStreaming={message.id === streamingMessageId}
                showTimestamp
                showAgent
              />
            ))}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="mx-2 mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700 flex items-center gap-2">
          <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
          <span>{error}</span>
          <button
            type="button"
            onClick={() => setError(null)}
            className="ml-auto text-red-600 hover:text-red-800"
          >
            <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
              <path
                fillRule="evenodd"
                d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                clipRule="evenodd"
              />
            </svg>
          </button>
        </div>
      )}

      {/* Input area */}
      <div className="pt-4 border-t border-gray-200 flex-shrink-0">
        <ChatInput
          value={inputValue}
          onChange={setInputValue}
          onSubmit={handleSendMessage}
          selectedAgent={selectedAgent}
          agents={agents}
          disabled={isStreaming}
          isStreaming={isStreaming}
          onStopStreaming={handleStopStreaming}
        />
      </div>
    </div>
  );
};

export default ChatContainer;
