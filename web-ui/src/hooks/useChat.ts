/**
 * useChat hook for managing chat state with streaming response support.
 *
 * Provides a reusable hook for chat functionality including:
 * - Message state management
 * - Streaming response handling via SSE
 * - Abort/cancel support
 * - Error handling
 */

import { useState, useCallback, useRef } from 'react';
import type { Message } from '@/types';
import { chatStream, type ChatRequestPayload, type ChatResponseNormalized } from '@/services/agentApi';

/**
 * Options for the useChat hook.
 */
export interface UseChatOptions {
  /** Initial messages to populate the chat */
  initialMessages?: Message[];
  /** Callback when a user message is sent */
  onMessageSent?: (message: Message) => void;
  /** Callback when an assistant response is received */
  onResponseReceived?: (message: Message) => void;
  /** Callback when an error occurs */
  onError?: (error: Error) => void;
  /** Callback when streaming starts */
  onStreamStart?: () => void;
  /** Callback when streaming ends */
  onStreamEnd?: () => void;
}

/**
 * Return type for the useChat hook.
 */
export interface UseChatReturn {
  /** Current messages in the chat */
  messages: Message[];
  /** Whether a streaming response is in progress */
  isStreaming: boolean;
  /** ID of the message currently being streamed */
  streamingMessageId: string | null;
  /** Current error, if any */
  error: string | null;
  /** Send a message to an agent */
  sendMessage: (content: string, agentName: string, options?: ChatRequestPayload['options']) => Promise<void>;
  /** Stop the current streaming response */
  stopStreaming: () => void;
  /** Clear all messages */
  clearMessages: () => void;
  /** Clear the current error */
  clearError: () => void;
  /** Add a message manually (e.g., for system messages) */
  addMessage: (message: Message) => void;
  /** Update a specific message by ID */
  updateMessage: (id: string, updates: Partial<Message>) => void;
  /** Remove a message by ID */
  removeMessage: (id: string) => void;
  /** Set messages directly (e.g., for loading from history) */
  setMessages: React.Dispatch<React.SetStateAction<Message[]>>;
}

/**
 * Generate a unique message ID.
 */
function generateMessageId(): string {
  return `msg-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

/**
 * Hook for managing chat state with streaming response support.
 *
 * @param options - Configuration options for the hook
 * @returns Chat state and control functions
 *
 * @example
 * ```tsx
 * function ChatComponent() {
 *   const {
 *     messages,
 *     isStreaming,
 *     error,
 *     sendMessage,
 *     stopStreaming,
 *     clearMessages,
 *   } = useChat({
 *     onMessageSent: (msg) => console.log('Sent:', msg),
 *     onResponseReceived: (msg) => console.log('Received:', msg),
 *   });
 *
 *   const handleSend = async (content: string) => {
 *     await sendMessage(content, 'assistant');
 *   };
 *
 *   return (
 *     <div>
 *       {messages.map((msg) => (
 *         <div key={msg.id}>{msg.content}</div>
 *       ))}
 *       {isStreaming && <button onClick={stopStreaming}>Stop</button>}
 *     </div>
 *   );
 * }
 * ```
 */
export function useChat(options: UseChatOptions = {}): UseChatReturn {
  const {
    initialMessages = [],
    onMessageSent,
    onResponseReceived,
    onError,
    onStreamStart,
    onStreamEnd,
  } = options;

  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  /**
   * Send a message to an agent and handle the streaming response.
   */
  const sendMessage = useCallback(
    async (
      content: string,
      agentName: string,
      chatOptions?: ChatRequestPayload['options']
    ): Promise<void> => {
      if (!content.trim()) return;

      setError(null);

      // Create user message
      const userMessageId = generateMessageId();
      const userMessage: Message = {
        id: userMessageId,
        role: 'user',
        content: content.trim(),
        timestamp: new Date().toISOString(),
        agentName,
      };

      // Add user message
      setMessages((prev) => [...prev, userMessage]);
      onMessageSent?.(userMessage);

      // Create placeholder for assistant response
      const assistantMessageId = generateMessageId();
      const assistantMessage: Message = {
        id: assistantMessageId,
        role: 'assistant',
        content: '',
        timestamp: new Date().toISOString(),
        agentName,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      setIsStreaming(true);
      setStreamingMessageId(assistantMessageId);
      onStreamStart?.();

      // Create abort controller for this request
      abortControllerRef.current = new AbortController();

      try {
        let fullContent = '';

        await chatStream(
          agentName,
          { message: content.trim(), options: chatOptions },
          // onChunk - update message content as chunks arrive
          (chunk: string) => {
            fullContent += chunk;
            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId
                  ? { ...msg, content: fullContent }
                  : msg
              )
            );
          },
          // onComplete - finalize the message with metadata
          (response: ChatResponseNormalized) => {
            const finalMessage: Message = {
              id: assistantMessageId,
              role: 'assistant',
              content: response.content,
              timestamp: new Date().toISOString(),
              agentName,
              metadata: {
                model: response.model,
                latencyMs: response.latencyMs,
                finishReason: response.finishReason,
                usage: response.usage,
                agent: response.agent,
              },
            };

            setMessages((prev) =>
              prev.map((msg) =>
                msg.id === assistantMessageId ? finalMessage : msg
              )
            );
            onResponseReceived?.(finalMessage);
          },
          // onError - handle stream errors
          (err: Error) => {
            setError(err.message);
            onError?.(err);
            // Remove the empty assistant message on error
            setMessages((prev) =>
              prev.filter((msg) => msg.id !== assistantMessageId)
            );
          },
          abortControllerRef.current.signal
        );
      } catch (err) {
        // Handle aborted requests silently
        if (abortControllerRef.current?.signal.aborted) {
          return;
        }

        const errorMessage = err instanceof Error ? err.message : 'Failed to send message';
        setError(errorMessage);
        onError?.(err instanceof Error ? err : new Error(errorMessage));

        // Remove the empty assistant message on error
        setMessages((prev) =>
          prev.filter((msg) => msg.id !== assistantMessageId)
        );
      } finally {
        setIsStreaming(false);
        setStreamingMessageId(null);
        abortControllerRef.current = null;
        onStreamEnd?.();
      }
    },
    [onMessageSent, onResponseReceived, onError, onStreamStart, onStreamEnd]
  );

  /**
   * Stop the current streaming response.
   */
  const stopStreaming = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      setIsStreaming(false);
      setStreamingMessageId(null);
    }
  }, []);

  /**
   * Clear all messages.
   */
  const clearMessages = useCallback(() => {
    setMessages([]);
    setError(null);
  }, []);

  /**
   * Clear the current error.
   */
  const clearError = useCallback(() => {
    setError(null);
  }, []);

  /**
   * Add a message manually.
   */
  const addMessage = useCallback((message: Message) => {
    setMessages((prev) => [...prev, message]);
  }, []);

  /**
   * Update a specific message by ID.
   */
  const updateMessage = useCallback((id: string, updates: Partial<Message>) => {
    setMessages((prev) =>
      prev.map((msg) => (msg.id === id ? { ...msg, ...updates } : msg))
    );
  }, []);

  /**
   * Remove a message by ID.
   */
  const removeMessage = useCallback((id: string) => {
    setMessages((prev) => prev.filter((msg) => msg.id !== id));
  }, []);

  return {
    messages,
    isStreaming,
    streamingMessageId,
    error,
    sendMessage,
    stopStreaming,
    clearMessages,
    clearError,
    addMessage,
    updateMessage,
    removeMessage,
    setMessages,
  };
}

/**
 * Hook for managing chat state with agent selection.
 * Extends useChat with agent management functionality.
 *
 * @param agents - List of available agents
 * @param options - Chat options
 * @returns Extended chat state with agent selection
 */
export interface UseChatWithAgentReturn extends UseChatReturn {
  /** Currently selected agent */
  selectedAgent: string | null;
  /** Set the selected agent */
  setSelectedAgent: (agent: string | null) => void;
  /** Send a message using the selected agent */
  sendToSelectedAgent: (content: string, options?: ChatRequestPayload['options']) => Promise<void>;
}

export interface UseChatWithAgentOptions extends UseChatOptions {
  /** Initial selected agent */
  initialAgent?: string | null;
  /** Callback when agent selection changes */
  onAgentChange?: (agent: string | null) => void;
}

/**
 * Hook for managing chat with agent selection support.
 *
 * @param options - Configuration options
 * @returns Chat state with agent management
 */
export function useChatWithAgent(
  options: UseChatWithAgentOptions = {}
): UseChatWithAgentReturn {
  const { initialAgent = null, onAgentChange, ...chatOptions } = options;
  const [selectedAgent, setSelectedAgentState] = useState<string | null>(initialAgent);

  const chat = useChat(chatOptions);

  const setSelectedAgent = useCallback(
    (agent: string | null) => {
      setSelectedAgentState(agent);
      onAgentChange?.(agent);
    },
    [onAgentChange]
  );

  const sendToSelectedAgent = useCallback(
    async (content: string, requestOptions?: ChatRequestPayload['options']): Promise<void> => {
      if (!selectedAgent) {
        throw new Error('No agent selected');
      }
      return chat.sendMessage(content, selectedAgent, requestOptions);
    },
    [selectedAgent, chat]
  );

  return {
    ...chat,
    selectedAgent,
    setSelectedAgent,
    sendToSelectedAgent,
  };
}

export default useChat;
