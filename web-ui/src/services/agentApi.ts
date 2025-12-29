/**
 * Agent API service for WeaverTools web-ui.
 * Handles agent-related operations including chat.
 */

import { get, post, ApiError } from './api';
import type { AgentInfo, AgentListResponse, ChatMessage, ChatResponse } from '@/types';

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

/** Chat request payload */
export interface ChatRequestPayload {
  message: string;
  history?: ChatMessage[];
  options?: {
    max_tokens?: number;
    temperature?: number;
    stream?: boolean;
    return_hidden_states?: boolean;
  };
}

/** Chat response from API */
interface ChatApiResponse {
  response: ChatResponse;
}

/** Streaming chat event types */
export interface ChatStreamEvent {
  type: 'content' | 'done' | 'error';
  content?: string;
  finish_reason?: string;
  error?: string;
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
 * Send a chat message to an agent.
 * @param name - Agent name
 * @param payload - Chat request payload
 * @returns Promise resolving to ChatResponse
 * @throws ApiError on failure
 */
export async function chat(
  name: string,
  payload: ChatRequestPayload
): Promise<ChatResponse> {
  const response = await post<ChatApiResponse>(ENDPOINTS.chat(name), payload);
  return response.data.response;
}

/**
 * Send a simple message to an agent.
 * Convenience wrapper around chat() for basic messages.
 * @param name - Agent name
 * @param message - Message content
 * @param options - Optional chat parameters
 * @returns Promise resolving to response content
 */
export async function sendMessage(
  name: string,
  message: string,
  options?: ChatRequestPayload['options']
): Promise<string> {
  const response = await chat(name, { message, options });
  return response.content;
}

/**
 * Send a chat message and stream the response.
 * Uses Server-Sent Events for real-time updates.
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
  onComplete?: (response: ChatResponse) => void,
  onError?: (error: Error) => void,
  signal?: AbortSignal
): Promise<void> {
  // Ensure stream is requested
  const streamPayload = {
    ...payload,
    options: {
      ...payload.options,
      stream: true,
    },
  };

  try {
    const response = await fetch(ENDPOINTS.chat(name), {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify(streamPayload),
      signal,
    });

    if (!response.ok) {
      throw new ApiError(
        `Chat stream failed: ${response.status} ${response.statusText}`,
        response.status,
        response.statusText
      );
    }

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
              usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
              latencyMs: 0,
              model: '',
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
                usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
                latencyMs: 0,
                model: '',
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
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      latencyMs: 0,
      model: '',
      finishReason: 'stop',
    });
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
