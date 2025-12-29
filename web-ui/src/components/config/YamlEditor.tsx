/**
 * YamlEditor component - raw YAML configuration editor.
 *
 * Provides syntax-highlighted YAML editing with validation,
 * line numbers, and error display for configuration management.
 */
import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import * as yaml from 'js-yaml';
import type { Config } from '@/types';

/**
 * YamlEditor component props.
 */
export interface YamlEditorProps {
  /** Current configuration object */
  config: Config;
  /** Callback when config changes from YAML edits */
  onChange: (config: Config) => void;
  /** Whether editor is disabled */
  disabled?: boolean;
  /** Callback when YAML parsing error occurs */
  onParseError?: (error: string | null) => void;
}

/**
 * Convert Config object to YAML string.
 * Uses snake_case keys to match the actual config.yaml format.
 */
function configToYaml(config: Config): string {
  // Transform to snake_case for YAML output
  const yamlConfig = {
    backends: {
      claude_code: {
        enabled: config.backends.claudeCode.enabled,
      },
      loom: {
        enabled: config.backends.loom.enabled,
        url: config.backends.loom.url,
        path: config.backends.loom.path,
        auto_start: config.backends.loom.autoStart,
        port: config.backends.loom.port,
      },
    },
    agents: Object.fromEntries(
      Object.entries(config.agents).map(([name, agent]) => [
        name,
        {
          role: agent.role,
          backend: agent.backend,
          model: agent.model,
          system_prompt: agent.systemPrompt,
          tools: agent.tools,
          tools_enabled: agent.toolsEnabled,
          active: agent.active,
          max_tokens: agent.maxTokens,
          temperature: agent.temperature,
          context_length: agent.contextLength,
          top_p: agent.topP,
          top_k: agent.topK,
          gpu: agent.gpu,
        },
      ])
    ),
    session: {
      measurement_mode: config.session.measurementMode,
      auto_export: config.session.autoExport,
      export_path: config.session.exportPath,
    },
  };

  return yaml.dump(yamlConfig, {
    indent: 2,
    lineWidth: 120,
    noRefs: true,
    sortKeys: false,
  });
}

/**
 * Parse YAML string to Config object.
 * Converts snake_case keys to camelCase.
 */
function yamlToConfig(yamlStr: string): Config {
  const parsed = yaml.load(yamlStr) as Record<string, unknown>;

  if (!parsed || typeof parsed !== 'object') {
    throw new Error('Invalid YAML: expected an object');
  }

  const backends = parsed.backends as Record<string, unknown> | undefined;
  const agents = parsed.agents as Record<string, Record<string, unknown>> | undefined;
  const session = parsed.session as Record<string, unknown> | undefined;

  if (!backends || !agents || !session) {
    throw new Error('Invalid config: missing required sections (backends, agents, session)');
  }

  const claudeCode = (backends.claude_code ?? backends.claudeCode) as Record<string, unknown> | undefined;
  const loom = backends.loom as Record<string, unknown> | undefined;

  if (!claudeCode || !loom) {
    throw new Error('Invalid config: backends must have claude_code and loom sections');
  }

  return {
    backends: {
      claudeCode: {
        enabled: Boolean(claudeCode.enabled),
      },
      loom: {
        enabled: Boolean(loom.enabled),
        url: String(loom.url ?? 'http://localhost:8080'),
        path: String(loom.path ?? '../TheLoom/the-loom'),
        autoStart: Boolean(loom.auto_start ?? loom.autoStart),
        port: Number(loom.port ?? 8080),
      },
    },
    agents: Object.fromEntries(
      Object.entries(agents).map(([name, agent]) => [
        name,
        {
          role: String(agent.role ?? 'junior'),
          backend: String(agent.backend ?? 'loom'),
          model: agent.model ? String(agent.model) : undefined,
          systemPrompt: String(agent.system_prompt ?? agent.systemPrompt ?? ''),
          tools: Array.isArray(agent.tools) ? agent.tools.map(String) : undefined,
          toolsEnabled: Boolean(agent.tools_enabled ?? agent.toolsEnabled),
          active: Boolean(agent.active),
          maxTokens: agent.max_tokens ?? agent.maxTokens ? Number(agent.max_tokens ?? agent.maxTokens) : undefined,
          temperature: agent.temperature != null ? Number(agent.temperature) : undefined,
          contextLength: agent.context_length ?? agent.contextLength ? Number(agent.context_length ?? agent.contextLength) : undefined,
          topP: agent.top_p ?? agent.topP != null ? Number(agent.top_p ?? agent.topP) : undefined,
          topK: agent.top_k ?? agent.topK ? Number(agent.top_k ?? agent.topK) : undefined,
          gpu: agent.gpu ? String(agent.gpu) : undefined,
        },
      ])
    ),
    session: {
      measurementMode: String(session.measurement_mode ?? session.measurementMode ?? 'active'),
      autoExport: Boolean(session.auto_export ?? session.autoExport),
      exportPath: String(session.export_path ?? session.exportPath ?? './experiments'),
    },
  };
}

/**
 * Basic YAML syntax highlighting.
 * Returns HTML with spans for different token types.
 */
function highlightYaml(yamlStr: string): string {
  const lines = yamlStr.split('\n');
  const highlighted = lines.map((line) => {
    // Escape HTML special characters
    let escaped = line
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');

    // Comment lines
    if (/^\s*#/.test(escaped)) {
      return `<span class="yaml-comment">${escaped}</span>`;
    }

    // Process keys and values
    escaped = escaped
      // Keys (word followed by colon)
      .replace(/^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)(:)/gm, '$1<span class="yaml-key">$2</span>$3')
      // Strings in quotes
      .replace(/"([^"]*)"/g, '<span class="yaml-string">"$1"</span>')
      .replace(/'([^']*)'/g, "<span class=\"yaml-string\">'$1'</span>")
      // Booleans
      .replace(/\b(true|false)\b/g, '<span class="yaml-boolean">$1</span>')
      // Numbers
      .replace(/\b(\d+(?:\.\d+)?)\b/g, '<span class="yaml-number">$1</span>')
      // Null
      .replace(/\b(null|~)\b/g, '<span class="yaml-null">$1</span>')
      // Array indicators
      .replace(/^(\s*)(-)(\s)/gm, '$1<span class="yaml-array">$2</span>$3');

    return escaped;
  });

  return highlighted.join('\n');
}

/**
 * Raw YAML editor component with syntax highlighting.
 */
export const YamlEditor: React.FC<YamlEditorProps> = ({
  config,
  onChange,
  disabled = false,
  onParseError,
}) => {
  // Convert config to YAML once, memoized
  const initialYaml = useMemo(() => configToYaml(config), [config]);

  // Local YAML state for editing
  const [yamlContent, setYamlContent] = useState(initialYaml);
  const [parseError, setParseError] = useState<string | null>(null);
  const [cursorPosition, setCursorPosition] = useState({ line: 1, column: 1 });

  // Refs for synchronized scrolling
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const highlightRef = useRef<HTMLPreElement>(null);
  const lineNumbersRef = useRef<HTMLDivElement>(null);

  // Sync YAML content when config prop changes externally
  useEffect(() => {
    const newYaml = configToYaml(config);
    // Only update if the YAML would be different (avoid cursor jump on own edits)
    if (newYaml !== yamlContent && parseError === null) {
      setYamlContent(newYaml);
    }
  }, [config]); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle YAML text changes
  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLTextAreaElement>) => {
      const newYaml = event.target.value;
      setYamlContent(newYaml);

      try {
        const newConfig = yamlToConfig(newYaml);
        setParseError(null);
        onParseError?.(null);
        onChange(newConfig);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Invalid YAML';
        setParseError(errorMessage);
        onParseError?.(errorMessage);
      }
    },
    [onChange, onParseError]
  );

  // Update cursor position on selection change
  const handleSelect = useCallback(() => {
    if (textareaRef.current) {
      const start = textareaRef.current.selectionStart;
      const textBeforeCursor = yamlContent.substring(0, start);
      const lines = textBeforeCursor.split('\n');
      setCursorPosition({
        line: lines.length,
        column: (lines[lines.length - 1]?.length ?? 0) + 1,
      });
    }
  }, [yamlContent]);

  // Sync scroll between textarea and highlight overlay
  const handleScroll = useCallback(() => {
    if (textareaRef.current && highlightRef.current && lineNumbersRef.current) {
      const scrollTop = textareaRef.current.scrollTop;
      const scrollLeft = textareaRef.current.scrollLeft;
      highlightRef.current.scrollTop = scrollTop;
      highlightRef.current.scrollLeft = scrollLeft;
      lineNumbersRef.current.scrollTop = scrollTop;
    }
  }, []);

  // Generate line numbers
  const lineCount = yamlContent.split('\n').length;
  const lineNumbers = Array.from({ length: lineCount }, (_, i) => i + 1);

  // Get highlighted HTML
  const highlightedHtml = useMemo(
    () => highlightYaml(yamlContent),
    [yamlContent]
  );

  // Format YAML (re-parse and re-dump)
  const handleFormat = useCallback(() => {
    try {
      const parsed = yamlToConfig(yamlContent);
      const formatted = configToYaml(parsed);
      setYamlContent(formatted);
      setParseError(null);
      onParseError?.(null);
      onChange(parsed);
    } catch {
      // Keep error state, don't format
    }
  }, [yamlContent, onChange, onParseError]);

  // Copy to clipboard
  const handleCopy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(yamlContent);
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement('textarea');
      textarea.value = yamlContent;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    }
  }, [yamlContent]);

  return (
    <div className="space-y-4">
      {/* Toolbar */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={handleFormat}
            disabled={disabled || parseError !== null}
            className="btn-secondary text-sm flex items-center gap-1"
            title="Format YAML"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16m-7 6h7" />
            </svg>
            Format
          </button>
          <button
            type="button"
            onClick={handleCopy}
            disabled={disabled}
            className="btn-secondary text-sm flex items-center gap-1"
            title="Copy to clipboard"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
            </svg>
            Copy
          </button>
        </div>
        <div className="text-sm text-gray-500">
          Line {cursorPosition.line}, Column {cursorPosition.column}
        </div>
      </div>

      {/* Parse Error Display */}
      {parseError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-3">
          <div className="flex items-start gap-2">
            <svg className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
            </svg>
            <div>
              <h4 className="text-sm font-medium text-red-800">YAML Parse Error</h4>
              <p className="text-sm text-red-700 mt-1 font-mono">{parseError}</p>
            </div>
          </div>
        </div>
      )}

      {/* Editor Container */}
      <div className="relative border border-gray-300 rounded-lg overflow-hidden bg-gray-900">
        <div className="flex">
          {/* Line Numbers */}
          <div
            ref={lineNumbersRef}
            className="flex-shrink-0 bg-gray-800 text-gray-500 text-sm font-mono py-3 px-2 select-none overflow-hidden border-r border-gray-700"
            style={{ minWidth: '3rem' }}
          >
            {lineNumbers.map((num) => (
              <div
                key={num}
                className="text-right pr-2 leading-6"
              >
                {num}
              </div>
            ))}
          </div>

          {/* Editor Area */}
          <div className="relative flex-1 overflow-hidden">
            {/* Syntax Highlighted Overlay */}
            <pre
              ref={highlightRef}
              className="absolute inset-0 text-sm font-mono py-3 px-4 whitespace-pre overflow-auto pointer-events-none leading-6 m-0"
              style={{
                color: 'transparent',
                caretColor: 'transparent',
              }}
              dangerouslySetInnerHTML={{ __html: highlightedHtml + '\n' }}
              aria-hidden="true"
            />

            {/* Actual Textarea */}
            <textarea
              ref={textareaRef}
              value={yamlContent}
              onChange={handleChange}
              onSelect={handleSelect}
              onScroll={handleScroll}
              onKeyUp={handleSelect}
              onClick={handleSelect}
              disabled={disabled}
              spellCheck={false}
              className="w-full h-[500px] text-sm font-mono py-3 px-4 bg-transparent text-gray-100 resize-none outline-none leading-6 caret-white"
              style={{
                WebkitTextFillColor: 'transparent',
              }}
            />
          </div>
        </div>
      </div>

      {/* Editor Info */}
      <div className="flex items-center justify-between text-sm text-gray-500">
        <div className="flex items-center gap-4">
          <span>{lineCount} lines</span>
          <span>{yamlContent.length} characters</span>
        </div>
        <div className="flex items-center gap-2">
          {parseError ? (
            <span className="text-red-500 flex items-center gap-1">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              Invalid YAML
            </span>
          ) : (
            <span className="text-green-500 flex items-center gap-1">
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              Valid YAML
            </span>
          )}
        </div>
      </div>

      {/* Syntax Highlighting Styles */}
      <style>{`
        .yaml-key {
          color: #7dd3fc;
        }
        .yaml-string {
          color: #86efac;
        }
        .yaml-boolean {
          color: #fca5a5;
        }
        .yaml-number {
          color: #fde68a;
        }
        .yaml-null {
          color: #a78bfa;
        }
        .yaml-comment {
          color: #6b7280;
          font-style: italic;
        }
        .yaml-array {
          color: #f9a8d4;
        }
      `}</style>
    </div>
  );
};

export default YamlEditor;
