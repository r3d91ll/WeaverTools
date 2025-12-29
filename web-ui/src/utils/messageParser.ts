/**
 * Message parsing utilities for Weaver chat interface.
 *
 * Implements @agent targeting syntax and command parsing following
 * the patterns from Weaver/pkg/shell/shell.go.
 *
 * Supported syntax:
 * - "@agentName message" - Send message to specific agent
 * - "/command args" - Execute a command (reserved for future use)
 * - "message" - Send message to default agent
 */

/**
 * Represents a parsed message with agent targeting information.
 */
export interface ParsedMessage {
  /** The message content to send (without the @agent prefix) */
  content: string;
  /** The target agent name, or null for default agent */
  targetAgent: string | null;
  /** Whether the message uses explicit @agent targeting */
  isTargeted: boolean;
  /** Whether the input is a command (starts with /) */
  isCommand: boolean;
  /** The command name if isCommand is true */
  commandName: string | null;
  /** Command arguments if isCommand is true */
  commandArgs: string[];
  /** The original raw input */
  rawInput: string;
}

/**
 * Represents an @agent mention found in the message.
 */
export interface AgentMention {
  /** The agent name (without @ prefix) */
  agentName: string;
  /** Starting index of the mention in the original string */
  startIndex: number;
  /** Ending index of the mention in the original string */
  endIndex: number;
}

/**
 * Parse a chat message for @agent targeting syntax.
 *
 * Follows the pattern from Weaver shell.go handleMessage:
 * - If line starts with "@", extract agent name from first word
 * - The rest of the message (after space) is the content
 * - If no message follows @agent, content will be empty
 *
 * @param input - The raw user input string
 * @returns Parsed message with targeting information
 *
 * @example
 * parseMessage("@researcher What is AI?")
 * // => { content: "What is AI?", targetAgent: "researcher", isTargeted: true, ... }
 *
 * @example
 * parseMessage("Hello world")
 * // => { content: "Hello world", targetAgent: null, isTargeted: false, ... }
 */
export function parseMessage(input: string): ParsedMessage {
  const trimmed = input.trim();

  // Base result structure
  const result: ParsedMessage = {
    content: trimmed,
    targetAgent: null,
    isTargeted: false,
    isCommand: false,
    commandName: null,
    commandArgs: [],
    rawInput: input,
  };

  // Empty input
  if (!trimmed) {
    result.content = '';
    return result;
  }

  // Check for command (starts with /)
  if (trimmed.startsWith('/')) {
    result.isCommand = true;
    const parts = trimmed.split(/\s+/);
    result.commandName = parts[0].substring(1); // Remove leading /
    result.commandArgs = parts.slice(1);
    result.content = trimmed;
    return result;
  }

  // Check for @agent targeting (starts with @)
  if (trimmed.startsWith('@')) {
    result.isTargeted = true;

    // Split into agent name and message content
    // Pattern: @agentName message content here
    const spaceIndex = trimmed.indexOf(' ');

    if (spaceIndex === -1) {
      // Only @agent, no message
      result.targetAgent = trimmed.substring(1); // Remove @
      result.content = '';
    } else {
      // @agent followed by message
      result.targetAgent = trimmed.substring(1, spaceIndex); // Remove @ and get name
      result.content = trimmed.substring(spaceIndex + 1).trim();
    }

    return result;
  }

  // Regular message without targeting
  return result;
}

/**
 * Find all @agent mentions within a message.
 *
 * This finds mentions anywhere in the text, not just at the start.
 * Useful for highlighting or extracting multiple agent references.
 *
 * @param input - The message text to search
 * @returns Array of agent mentions found
 *
 * @example
 * findAgentMentions("Ask @researcher and @analyst about this")
 * // => [{ agentName: "researcher", startIndex: 4, endIndex: 15 }, ...]
 */
export function findAgentMentions(input: string): AgentMention[] {
  const mentions: AgentMention[] = [];

  // Match @word patterns (alphanumeric and underscore/hyphen)
  const regex = /@([a-zA-Z][a-zA-Z0-9_-]*)/g;
  let match: RegExpExecArray | null;

  while ((match = regex.exec(input)) !== null) {
    mentions.push({
      agentName: match[1],
      startIndex: match.index,
      endIndex: match.index + match[0].length,
    });
  }

  return mentions;
}

/**
 * Validate an agent name against common naming rules.
 *
 * Agent names should:
 * - Start with a letter
 * - Contain only alphanumeric characters, underscores, or hyphens
 * - Be between 1 and 50 characters
 *
 * @param name - The agent name to validate
 * @returns True if the name is valid
 */
export function isValidAgentName(name: string): boolean {
  if (!name || name.length === 0 || name.length > 50) {
    return false;
  }

  // Must start with letter, then alphanumeric/underscore/hyphen
  const validPattern = /^[a-zA-Z][a-zA-Z0-9_-]*$/;
  return validPattern.test(name);
}

/**
 * Extract the primary agent target from a message.
 *
 * Returns the first @agent found, whether at the start or inline.
 *
 * @param input - The message text
 * @returns The target agent name or null
 */
export function extractTargetAgent(input: string): string | null {
  const parsed = parseMessage(input);
  if (parsed.targetAgent) {
    return parsed.targetAgent;
  }

  // Fall back to first inline mention
  const mentions = findAgentMentions(input);
  return mentions.length > 0 ? mentions[0].agentName : null;
}

/**
 * Remove @agent prefix from the beginning of a message.
 *
 * @param input - The message with potential @agent prefix
 * @returns The message content without the @agent prefix
 */
export function stripAgentPrefix(input: string): string {
  const parsed = parseMessage(input);
  return parsed.content;
}

/**
 * Format a message with @agent prefix for display or sending.
 *
 * @param agentName - The target agent name
 * @param message - The message content
 * @returns Formatted message string
 */
export function formatAgentMessage(agentName: string, message: string): string {
  if (!agentName) {
    return message;
  }
  return `@${agentName} ${message}`;
}

/**
 * Check if a message contains any agent mentions.
 *
 * @param input - The message text to check
 * @returns True if the message contains @agent mentions
 */
export function hasAgentMention(input: string): boolean {
  return /@[a-zA-Z][a-zA-Z0-9_-]*/.test(input);
}

/**
 * Highlight agent mentions in a message for rich text display.
 *
 * Returns an array of segments that can be rendered with different styling.
 *
 * @param input - The message text
 * @returns Array of text segments with mention metadata
 */
export interface TextSegment {
  text: string;
  isMention: boolean;
  agentName?: string;
}

export function highlightAgentMentions(input: string): TextSegment[] {
  const segments: TextSegment[] = [];
  const mentions = findAgentMentions(input);

  if (mentions.length === 0) {
    return [{ text: input, isMention: false }];
  }

  let lastIndex = 0;

  for (const mention of mentions) {
    // Add text before mention
    if (mention.startIndex > lastIndex) {
      segments.push({
        text: input.substring(lastIndex, mention.startIndex),
        isMention: false,
      });
    }

    // Add mention
    segments.push({
      text: input.substring(mention.startIndex, mention.endIndex),
      isMention: true,
      agentName: mention.agentName,
    });

    lastIndex = mention.endIndex;
  }

  // Add remaining text
  if (lastIndex < input.length) {
    segments.push({
      text: input.substring(lastIndex),
      isMention: false,
    });
  }

  return segments;
}

/**
 * Suggest agent names based on partial input.
 *
 * Filters an agent list by a partial name match for autocomplete.
 *
 * @param partial - The partial agent name (without @)
 * @param availableAgents - List of available agent names
 * @param limit - Maximum suggestions to return
 * @returns Matching agent names sorted by relevance
 */
export function suggestAgents(
  partial: string,
  availableAgents: string[],
  limit = 5
): string[] {
  if (!partial) {
    return availableAgents.slice(0, limit);
  }

  const lowerPartial = partial.toLowerCase();

  // Score matches: exact start gets highest priority
  const scored = availableAgents
    .map((agent) => {
      const lowerAgent = agent.toLowerCase();
      let score = 0;

      if (lowerAgent === lowerPartial) {
        score = 100; // Exact match
      } else if (lowerAgent.startsWith(lowerPartial)) {
        score = 75; // Starts with
      } else if (lowerAgent.includes(lowerPartial)) {
        score = 50; // Contains
      }

      return { agent, score };
    })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || a.agent.localeCompare(b.agent));

  return scored.slice(0, limit).map(({ agent }) => agent);
}

/**
 * Parse the current cursor position for autocomplete context.
 *
 * Determines if the user is currently typing an @agent mention and
 * returns the partial text for autocomplete suggestions.
 *
 * @param input - The full input string
 * @param cursorPosition - The current cursor position
 * @returns Autocomplete context or null if not in an @mention
 */
export interface AutocompleteContext {
  /** The partial agent name being typed */
  partial: string;
  /** Start position of the @ symbol */
  startPosition: number;
  /** Whether we're at the start of input (primary targeting) */
  isAtStart: boolean;
}

export function getAutocompleteContext(
  input: string,
  cursorPosition: number
): AutocompleteContext | null {
  // Look backwards from cursor for @ symbol
  let atPosition = -1;

  for (let i = cursorPosition - 1; i >= 0; i--) {
    const char = input[i];

    // Stop at whitespace
    if (/\s/.test(char)) {
      break;
    }

    // Found @
    if (char === '@') {
      atPosition = i;
      break;
    }
  }

  if (atPosition === -1) {
    return null;
  }

  // Extract the partial agent name (between @ and cursor)
  const partial = input.substring(atPosition + 1, cursorPosition);

  // Validate partial is a valid prefix
  if (partial && !/^[a-zA-Z][a-zA-Z0-9_-]*$/.test(partial)) {
    // Not typing a valid agent name
    return null;
  }

  return {
    partial,
    startPosition: atPosition,
    isAtStart: atPosition === 0,
  };
}
