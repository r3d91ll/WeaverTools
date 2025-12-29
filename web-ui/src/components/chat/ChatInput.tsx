/**
 * ChatInput component - message input field with @agent targeting support.
 *
 * Parses @agent syntax from messages and provides autocomplete suggestions.
 * Uses messageParser utility for consistent parsing following Weaver/pkg/shell/shell.go patterns.
 */
import { useState, useRef, useEffect, useCallback } from 'react';
import type { AgentInfo } from '@/types';
import {
  parseMessage,
  getAutocompleteContext,
  suggestAgents,
  isValidAgentName,
  type ParsedMessage,
  type AutocompleteContext,
} from '@/utils/messageParser';

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
  /** Callback when message is submitted with full parsed info */
  onSubmitParsed?: (parsed: ParsedMessage) => void;
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
 * ChatInput component for entering chat messages.
 */
export const ChatInput: React.FC<ChatInputProps> = ({
  value,
  onChange,
  onSubmit,
  onSubmitParsed,
  selectedAgent,
  agents = [],
  placeholder = 'Type your message... (use @agent to target)',
  disabled = false,
  isStreaming = false,
  onStopStreaming,
}) => {
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [autocompleteCtx, setAutocompleteCtx] = useState<AutocompleteContext | null>(null);
  const [selectedSuggestionIndex, setSelectedSuggestionIndex] = useState(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Get available agent names for autocomplete
  const agentNames = agents.filter((a) => a.ready).map((a) => a.name);

  // Get filtered agent suggestions using messageParser utility
  const filteredAgentNames = autocompleteCtx
    ? suggestAgents(autocompleteCtx.partial, agentNames, 8)
    : [];

  // Map back to agent info objects for display
  const filteredAgents = filteredAgentNames
    .map((name) => agents.find((a) => a.name === name))
    .filter((a): a is AgentInfo => a !== undefined);

  // Handle input change - uses messageParser for autocomplete context
  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const newValue = e.target.value;
      onChange(newValue);

      // Check for @mention using messageParser utility
      const cursorPosition = e.target.selectionStart ?? 0;
      const ctx = getAutocompleteContext(newValue, cursorPosition);

      setAutocompleteCtx(ctx);
      if (ctx !== null) {
        setShowSuggestions(true);
        setSelectedSuggestionIndex(0);
      } else {
        setShowSuggestions(false);
      }
    },
    [onChange]
  );

  // Handle suggestion selection - uses autocomplete context for proper insertion
  const selectSuggestion = useCallback(
    (agentName: string) => {
      if (!inputRef.current || !autocompleteCtx) return;

      // Validate agent name before inserting
      if (!isValidAgentName(agentName)) return;

      const { startPosition } = autocompleteCtx;
      const cursorPosition = inputRef.current.selectionStart ?? 0;
      const textBeforeMention = value.slice(0, startPosition);
      const textAfterCursor = value.slice(cursorPosition);

      // Build new value with the complete agent mention
      const newValue = `${textBeforeMention}@${agentName} ${textAfterCursor}`;
      onChange(newValue);

      // Move cursor after the inserted mention
      setTimeout(() => {
        if (inputRef.current) {
          const newPosition = startPosition + agentName.length + 2; // @ + name + space
          inputRef.current.setSelectionRange(newPosition, newPosition);
          inputRef.current.focus();
        }
      }, 0);

      setShowSuggestions(false);
      setAutocompleteCtx(null);
    },
    [value, onChange, autocompleteCtx]
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

  // Handle form submission - uses parseMessage for consistent parsing
  const handleSubmit = useCallback(() => {
    const trimmedValue = value.trim();
    if (!trimmedValue || disabled || isStreaming) return;

    // Parse message using messageParser utility (follows shell.go patterns)
    const parsed = parseMessage(trimmedValue);

    // Ignore command messages (starting with /)
    if (parsed.isCommand) {
      // Commands should be handled separately in the future
      return;
    }

    // Determine target agent: mentioned > selected > null
    const targetAgent = parsed.targetAgent || selectedAgent;

    // If full parsed info is needed, call the new callback
    if (onSubmitParsed) {
      // Enrich parsed message with fallback agent
      const enrichedParsed: ParsedMessage = {
        ...parsed,
        targetAgent: targetAgent,
      };
      onSubmitParsed(enrichedParsed);
    }

    // Submit the clean message content (backwards compatible)
    onSubmit(parsed.content, targetAgent);
  }, [value, disabled, isStreaming, selectedAgent, onSubmit, onSubmitParsed]);

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

          {/* Target indicator - shows current @agent target from parsing */}
          {(() => {
            const parsed = parseMessage(value);
            const effectiveTarget = parsed.targetAgent || selectedAgent;
            if (!effectiveTarget || !value.trim()) return null;
            return (
              <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1 text-xs">
                <svg className="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
                </svg>
                <span className={parsed.isTargeted ? 'text-weaver-600 font-medium' : 'text-gray-400'}>
                  @{effectiveTarget}
                </span>
              </div>
            );
          })()}
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
