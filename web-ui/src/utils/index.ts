/**
 * Utility exports for Weaver web UI.
 */

// Message parsing utilities
export {
  parseMessage,
  findAgentMentions,
  isValidAgentName,
  extractTargetAgent,
  stripAgentPrefix,
  formatAgentMessage,
  hasAgentMention,
  highlightAgentMentions,
  suggestAgents,
  getAutocompleteContext,
  type ParsedMessage,
  type AgentMention,
  type TextSegment,
  type AutocompleteContext,
} from './messageParser';
