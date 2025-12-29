/**
 * Agent API service for WeaverTools web-ui.
 * Handles agent-related operations including chat.
 */

import { get, post, ApiError } from './api';
import type { AgentInfo, AgentListResponse, ChatAgentResponse, ChatHistoryMessage } from '@/types';

// Re-export AgentInfo for consumers
export type { AgentInfo } from '@/types';

/** API endpoints for agent operations */
const ENDPOINTS = {
  agents: '/api/agents',
  agent: (name: string) => `/api/agents/${encodeURIComponent(name)}`,
  chat: (name: string) => `/api/agents/${encodeURIComponent(name)}/chat`,
} as const;

/** Single agent response */
interface AgentResponse {
  agent: AgentInfo;
}

/** Chat request payload matching backend ChatAgentRequest */
export interface ChatRequestPayload {
  message: string;
  history?: ChatHistoryMessage[];
  options?: {
    max_tokens?: number;
    temperature?: number;
    stream?: boolean;
    return_hidden_states?: boolean;
  };
}

/** Streaming chat event types */
export interface ChatStreamEvent {
  type: 'content' | 'done' | 'error';
  content?: string;
  finish_reason?: string;
  error?: string;
}

/**
 * Normalized chat response for frontend use.
 * Combines fields from ChatAgentResponse with additional frontend-needed fields.
 */
export interface ChatResponseNormalized {
  content: string;
  agent: string;
  model: string;
  latencyMs: number;
  finishReason: string;
  metadata?: Record<string, unknown>;
  usage?: { promptTokens: number; completionTokens: number; totalTokens: number };
}

/**
 * List all available agents.
 * @returns Promise resolving to array of AgentInfo
 * @throws ApiError on failure
 */
export async function listAgents(): Promise<AgentInfo[]> {
  const response = await get<AgentListResponse>(ENDPOINTS.agents);
  return response.data.agents;
}

/**
 * Get a specific agent by name.
 * @param name - Agent name
 * @returns Promise resolving to AgentInfo
 * @throws ApiError on failure (404 if not found)
 */
export async function getAgent(name: string): Promise<AgentInfo> {
  const response = await get<AgentResponse>(ENDPOINTS.agent(name));
  return response.data.agent;
}

/**
 * Check if an agent is available (ready).
 * @param name - Agent name
 * @returns Promise resolving to availability status
 */
export async function isAgentAvailable(name: string): Promise<boolean> {
  try {
    const agent = await getAgent(name);
    return agent.ready;
  } catch (error) {
    if (error instanceof ApiError && error.isNotFound()) {
      return false;
    }
    throw error;
  }
}

/**
 * Send a chat message to an agent (non-streaming).
 * @param name - Agent name
 * @param payload - Chat request payload
 * @returns Promise resolving to ChatAgentResponse
 * @throws ApiError on failure
 */
export async function chat(
  name: string,
  payload: ChatRequestPayload
): Promise<ChatAgentResponse> {
  // Send only the fields the backend expects (message and history)
  const requestBody = {
    message: payload.message,
    history: payload.history,
  };
  const response = await post<ChatAgentResponse>(ENDPOINTS.chat(name), requestBody);
  return response.data;
}

/**
 * Send a simple message to an agent.
 * Convenience wrapper around chat() for basic messages.
 * @param name - Agent name
 * @param message - Message content
 * @returns Promise resolving to response content
 */
export async function sendMessage(
  name: string,
  message: string
): Promise<string> {
  const response = await chat(name, { message });
  return response.content;
}

/**
 * Send a chat message and stream the response.
 * Supports both SSE streaming and regular JSON responses (fallback).
 * @param name - Agent name
 * @param payload - Chat request payload
 * @param onChunk - Callback for each content chunk
 * @param onComplete - Callback when stream completes
 * @param onError - Callback on error
 * @param signal - Optional AbortSignal for cancellation
 */
export async function chatStream(
  name: string,
  payload: ChatRequestPayload,
  onChunk: (content: string) => void,
  onComplete?: (response: ChatResponseNormalized) => void,
  onError?: (error: Error) => void,
  signal?: AbortSignal
): Promise<void> {
  // Send only the fields the backend expects (message and history)
  const requestBody = {
    message: payload.message,
    history: payload.history,
  };

  try {
    const response = await fetch(ENDPOINTS.chat(name), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json, text/event-stream',
      },
      body: JSON.stringify(requestBody),
      signal,
    });

    if (!response.ok) {
      // Try to parse error response
      let errorMessage = `Chat failed: ${response.status} ${response.statusText}`;
      try {
        const errorBody = await response.json();
        // Handle wrapped error response
        if (errorBody && typeof errorBody === 'object') {
          if (errorBody.error?.message) {
            errorMessage = errorBody.error.message;
          } else if (errorBody.message) {
            errorMessage = errorBody.message;
          }
        }
      } catch {
        // Ignore parse errors
      }
      throw new ApiError(
        errorMessage,
        response.status,
        response.statusText
      );
    }

    const contentType = response.headers.get('content-type') || '';

    // Handle SSE streaming response
    if (contentType.includes('text/event-stream')) {
      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('Response body is not readable');
      }

      const decoder = new TextDecoder();
      let buffer = '';
      let fullContent = '';

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });

        // Process complete SSE messages
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6);

            if (data === '[DONE]') {
              onComplete?.({
                content: fullContent,
                agent: name,
                model: '',
                latencyMs: 0,
                finishReason: 'stop',
              });
              return;
            }

            try {
              const event = JSON.parse(data) as ChatStreamEvent;

              if (event.type === 'content' && event.content) {
                fullContent += event.content;
                onChunk(event.content);
              } else if (event.type === 'done') {
                onComplete?.({
                  content: fullContent,
                  agent: name,
                  model: '',
                  latencyMs: 0,
                  finishReason: event.finish_reason ?? 'stop',
                });
                return;
              } else if (event.type === 'error') {
                throw new Error(event.error ?? 'Unknown stream error');
              }
            } catch (parseError) {
              // Ignore parse errors for non-JSON lines
            }
          }
        }
      }

      // Stream ended without explicit done event
      onComplete?.({
        content: fullContent,
        agent: name,
        model: '',
        latencyMs: 0,
        finishReason: 'stop',
      });
    } else {
      // Handle regular JSON response (non-streaming fallback)
      const rawJson = await response.json();

      // Handle wrapped API response: {success: boolean, data: ChatAgentResponse}
      let chatResponse: ChatAgentResponse;
      if (rawJson && typeof rawJson === 'object' && 'success' in rawJson) {
        if (!rawJson.success) {
          throw new Error(rawJson.error?.message || 'Chat request failed');
        }
        chatResponse = rawJson.data as ChatAgentResponse;
      } else {
        chatResponse = rawJson as ChatAgentResponse;
      }

      // Deliver full content as a single chunk
      if (chatResponse.content) {
        onChunk(chatResponse.content);
      }

      // Complete with the response
      onComplete?.({
        content: chatResponse.content,
        agent: chatResponse.agent || name,
        model: chatResponse.model || '',
        latencyMs: chatResponse.latencyMs || 0,
        finishReason: 'stop',
        metadata: chatResponse.metadata,
      });
    }
  } catch (error) {
    if (signal?.aborted) {
      return;
    }
    onError?.(error instanceof Error ? error : new Error(String(error)));
  }
}

/**
 * Parse @agent mention from message content.
 * Extracts agent name from patterns like "@agent_name message"
 * @param message - Raw message content
 * @returns Tuple of [agentName, cleanMessage] or [null, message] if no mention
 */
export function parseAgentMention(message: string): [string | null, string] {
  // Match @agent_name at start of message
  const match = message.match(/^@(\w+)\s+(.*)$/s);
  if (match) {
    return [match[1], match[2].trim()];
  }
  return [null, message];
}

/**
 * Format a message with agent mention.
 * @param agentName - Target agent name
 * @param message - Message content
 * @returns Formatted message with @mention
 */
export function formatAgentMention(agentName: string, message: string): string {
  return `@${agentName} ${message}`;
}

/** Agent API service object */
export const agentApi = {
  listAgents,
  getAgent,
  isAgentAvailable,
  chat,
  sendMessage,
  chatStream,
  parseAgentMention,
  formatAgentMention,
};

export default agentApi;
