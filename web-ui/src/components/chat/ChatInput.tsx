/**
 * ChatInput component - message input field with @agent targeting support.
 *
 * Parses @agent syntax from messages and provides autocomplete suggestions.
 */
import { useState, useRef, useEffect, useCallback } from 'react';
import type { AgentInfo } from '@/types';

/**
 * ChatInput component props.
 */
export interface ChatInputProps {
  /** Current input value */
  value: string;
  /** Callback when input value changes */
  onChange: (value: string) => void;
  /** Callback when message is submitted */
  onSubmit: (message: string, targetAgent: string | null) => void;
  /** Currently selected agent (from selector) */
  selectedAgent: string | null;
  /** Available agents for autocomplete */
  agents?: AgentInfo[];
  /** Placeholder text */
  placeholder?: string;
  /** Whether the input is disabled */
  disabled?: boolean;
  /** Whether a response is currently streaming */
  isStreaming?: boolean;
  /** Callback to stop streaming */
  onStopStreaming?: () => void;
}

/**
 * Parse @agent mention from start of message.
 */
function parseAgentMention(message: string): { agent: string | null; content: string } {
  const match = message.match(/^@(\w+)\s+(.*)$/s);
  if (match) {
    return { agent: match[1], content: match[2].trim() };
  }
  return { agent: null, content: message };
}

/**
 * Check if cursor is in @mention position.
 */
function getAtMentionPrefix(value: string, cursorPosition: number): string | null {
  // Look backwards from cursor for @ symbol
  const textBeforeCursor = value.slice(0, cursorPosition);
  const match = textBeforeCursor.match(/@(\w*)$/);
  if (match) {
    return match[1];
  }
  return null;
}

/**
 * ChatInput component for entering chat messages.
 */
export const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChange,
  onSubmit,
  selectedAgent,
  agents = [],
  placeholder = 'Type your message... (use @agent to target)',
  disabled = false,
  isStreaming = false,
  onStopStreaming,
}) => {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [suggestionFilter, setSuggestionFilter] = useState('');
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Filter agents based on current mention prefix
  const filteredAgents = agents.filter((agent) =>
    agent.name.toLowerCase().startsWith(suggestionFilter.toLowerCase())
  );

  // Handle input change
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const newValue = e.target.value;
      onChange(newValue);

      // Check for @mention
      const cursorPosition = e.target.selectionStart ?? 0;
      const mentionPrefix = getAtMentionPrefix(newValue, cursorPosition);

      if (mentionPrefix !== null) {
        setSuggestionFilter(mentionPrefix);
        setShowSuggestions(true);
        setSelectedSuggestionIndex(0);
      } else {
        setShowSuggestions(false);
      }
    },
    [onChange]
  );

  // Handle suggestion selection
  const selectSuggestion = useCallback(
    (agentName: string) => {
      if (!inputRef.current) return;

      const cursorPosition = inputRef.current.selectionStart ?? 0;
      const textBeforeCursor = value.slice(0, cursorPosition);
      const textAfterCursor = value.slice(cursorPosition);

      // Find the @ symbol position
      const atIndex = textBeforeCursor.lastIndexOf('@');
      if (atIndex >= 0) {
        const newValue =
          textBeforeCursor.slice(0, atIndex) + `@${agentName} ` + textAfterCursor;
        onChange(newValue);

        // Move cursor after the inserted mention
        setTimeout(() => {
          if (inputRef.current) {
            const newPosition = atIndex + agentName.length + 2;
            inputRef.current.setSelectionRange(newPosition, newPosition);
            inputRef.current.focus();
          }
        }, 0);
      }

      setShowSuggestions(false);
    },
    [value, onChange]
  );

  // Handle keyboard navigation in suggestions
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (showSuggestions && filteredAgents.length > 0) {
        switch (e.key) {
          case 'ArrowDown':
            e.preventDefault();
            setSelectedSuggestionIndex((prev) =>
              prev < filteredAgents.length - 1 ? prev + 1 : prev
            );
            return;
          case 'ArrowUp':
            e.preventDefault();
            setSelectedSuggestionIndex((prev) => (prev > 0 ? prev - 1 : prev));
            return;
          case 'Tab':
          case 'Enter':
            if (showSuggestions) {
              e.preventDefault();
              selectSuggestion(filteredAgents[selectedSuggestionIndex].name);
              return;
            }
            break;
          case 'Escape':
            e.preventDefault();
            setShowSuggestions(false);
            return;
        }
      }

      // Submit on Enter (without Shift)
      if (e.key === 'Enter' && !e.shiftKey && !showSuggestions) {
        e.preventDefault();
        handleSubmit();
      }
    },
    [showSuggestions, filteredAgents, selectedSuggestionIndex, selectSuggestion]
  );

  // Handle form submission
  const handleSubmit = useCallback(() => {
    const trimmedValue = value.trim();
    if (!trimmedValue || disabled || isStreaming) return;

    // Parse @agent mention from message
    const { agent: mentionedAgent, content } = parseAgentMention(trimmedValue);

    // Determine target agent: mentioned > selected > null
    const targetAgent = mentionedAgent || selectedAgent;

    // Submit the clean message content
    onSubmit(mentionedAgent ? content : trimmedValue, targetAgent);
  }, [value, disabled, isStreaming, selectedAgent, onSubmit]);

  // Auto-resize textarea
  useEffect(() => {
    if (inputRef.current) {
      inputRef.current.style.height = 'auto';
      inputRef.current.style.height = `${Math.min(inputRef.current.scrollHeight, 150)}px`;
    }
  }, [value]);

  // Close suggestions on click outside
  useEffect(() => {
    const handleClickOutside = () => setShowSuggestions(false);
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  return (
    <div className="relative">
      {/* @agent autocomplete suggestions */}
      {showSuggestions && filteredAgents.length > 0 && (
        <div className="absolute bottom-full mb-2 left-0 w-64 bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden z-50">
          <div className="px-3 py-2 text-xs text-gray-500 bg-gray-50 border-b border-gray-100">
            Agent suggestions
          </div>
          <div className="max-h-40 overflow-y-auto">
            {filteredAgents.map((agent, index) => (
              <button
                key={agent.name}
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  selectSuggestion(agent.name);
                }}
                className={`w-full text-left px-3 py-2 flex items-center gap-2 text-sm ${
                  index === selectedSuggestionIndex
                    ? 'bg-weaver-50 text-weaver-700'
                    : 'hover:bg-gray-50'
                }`}
              >
                <span
                  className={`w-2 h-2 rounded-full ${
                    agent.ready ? 'bg-green-500' : 'bg-gray-300'
                  }`}
                />
                <span className="font-medium">@{agent.name}</span>
                <span className="text-gray-400 text-xs">{agent.role}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input area */}
      <div className="flex items-end gap-3">
        <div className="flex-1 relative">
          <textarea
            ref={inputRef}
            value={value}
            onChange={handleChange}
            onKeyDown={handleKeyDown}
            placeholder={
              selectedAgent
                ? `Message @${selectedAgent}...`
                : placeholder
            }
            disabled={disabled}
            rows={1}
            className={`
              w-full px-4 py-3 pr-12 resize-none
              bg-white border border-gray-300 rounded-xl
              placeholder:text-gray-400 text-gray-900
              focus:outline-none focus:ring-2 focus:ring-weaver-500 focus:border-weaver-500
              disabled:opacity-50 disabled:cursor-not-allowed
              scrollbar-thin
            `}
            style={{ minHeight: '48px' }}
          />

          {/* Character indicator for @mention */}
          {selectedAgent && !value.startsWith('@') && (
            <div className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-gray-400">
              @{selectedAgent}
            </div>
          )}
        </div>

        {/* Send/Stop button */}
        {isStreaming ? (
          <button
            type="button"
            onClick={onStopStreaming}
            className="flex-shrink-0 p-3 bg-red-500 text-white rounded-xl hover:bg-red-600 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 transition-colors"
            title="Stop generating"
          >
            <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
              <rect x="6" y="6" width="8" height="8" rx="1" />
            </svg>
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSubmit}
            disabled={disabled || !value.trim()}
            className="flex-shrink-0 p-3 bg-weaver-600 text-white rounded-xl hover:bg-weaver-700 focus:outline-none focus:ring-2 focus:ring-weaver-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            title="Send message"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
              />
            </svg>
          </button>
        )}
      </div>

      {/* Hint text */}
      <div className="mt-2 flex items-center justify-between text-xs text-gray-400">
        <span>
          Press <kbd className="px-1 py-0.5 bg-gray-100 rounded text-gray-500">Enter</kbd> to send,{' '}
          <kbd className="px-1 py-0.5 bg-gray-100 rounded text-gray-500">Shift+Enter</kbd> for new line
        </span>
        <span>
          Type <kbd className="px-1 py-0.5 bg-gray-100 rounded text-gray-500">@</kbd> to mention an agent
        </span>
      </div>
    </div>
  );
};

export default ChatInput;
