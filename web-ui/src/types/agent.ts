/**
 * Agent types matching Weaver/pkg/api/handlers_agents.go
 *
 * Note: Field names use camelCase to match the API JSON responses.
 */

import type { AgentRole, BackendType } from './config';

/**
 * AgentInfo is the JSON representation of an agent's information.
 * Maps to AgentInfo in handlers_agents.go.
 */
export interface AgentInfo {
  name: string;
  role: AgentRole | string;
  backend: BackendType | string;
  model?: string;
  ready: boolean;
  hiddenStates: boolean;
}

/**
 * AgentListResponse is the JSON response for GET /api/agents.
 */
export interface AgentListResponse {
  agents: AgentInfo[];
}

/**
 * MessageRole represents the sender type.
 * Maps to MessageRole in Yarn/message.go.
 */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

/** Check if a role is valid. */
export function isValidMessageRole(role: string): role is MessageRole {
  return ['system', 'user', 'assistant', 'tool'].includes(role);
}

/**
 * ChatHistoryMessage represents a single message in chat history.
 * Maps to ChatHistoryMessage in handlers_agents.go.
 */
export interface ChatHistoryMessage {
  role: MessageRole;
  content: string;
  name?: string;
}

/**
 * ChatAgentRequest is the expected JSON body for POST /api/agents/:name/chat.
 * Maps to ChatAgentRequest in handlers_agents.go.
 */
export interface ChatAgentRequest {
  message: string;
  history?: ChatHistoryMessage[];
}

/**
 * ChatAgentResponse is the JSON response for POST /api/agents/:name/chat.
 * Maps to ChatAgentResponse in handlers_agents.go.
 */
export interface ChatAgentResponse {
  content: string;
  agent: string;
  model?: string;
  latencyMs?: number;
  metadata?: Record<string, unknown>;
}

/**
 * Message represents a chat message in the UI.
 * Extended from Yarn Message type for frontend use.
 */
export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: string;
  agentId?: string;
  agentName?: string;
  hiddenState?: import('./measurement').HiddenState | null;
  metadata?: Record<string, unknown>;
  // Tool-related fields
  toolCallId?: string;
  toolName?: string;
}

/**
 * Participant tracks an agent's involvement in the conversation.
 */
export interface Participant {
  agentId: string;
  agentName: string;
  role: string;
  joinedAt: string;
  messageCount: number;
}

/**
 * Conversation represents a conversation thread.
 */
export interface Conversation {
  id: string;
  name: string;
  messages: Message[];
  participants: Record<string, Participant>;
  createdAt: string;
  updatedAt: string;
  metadata?: Record<string, unknown>;
}
