/**
 * AgentSelector component - dropdown for selecting chat agents.
 *
 * Fetches available agents and allows selection for targeting messages.
 */
import { useState, useEffect, useRef } from 'react';
import type { AgentInfo } from '@/types';
import { listAgents } from '@/services/agentApi';

/**
 * AgentSelector component props.
 */
export interface AgentSelectorProps {
  /** Currently selected agent name */
  selectedAgent: string | null;
  /** Callback when agent selection changes */
  onSelectAgent: (agentName: string | null) => void;
  /** Whether the selector is disabled */
  disabled?: boolean;
  /** Placeholder text when no agent is selected */
  placeholder?: string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * Get role badge color.
 */
function getRoleBadgeColor(role: string): string {
  switch (role) {
    case 'senior':
      return 'bg-blue-100 text-blue-700';
    case 'junior':
      return 'bg-green-100 text-green-700';
    case 'critic':
      return 'bg-orange-100 text-orange-700';
    case 'reviewer':
      return 'bg-purple-100 text-purple-700';
    default:
      return 'bg-gray-100 text-gray-700';
  }
}

/**
 * AgentOption component for individual agent items.
 */
interface AgentOptionProps {
  agent: AgentInfo;
  isSelected: boolean;
  onClick: () => void;
}

const AgentOption: React.FC<AgentOptionProps> = ({ agent, isSelected, onClick }) => {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`w-full text-left px-3 py-2 flex items-center justify-between hover:bg-gray-50 ${
        isSelected ? 'bg-weaver-50' : ''
      }`}
    >
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full ${agent.ready ? 'bg-green-500' : 'bg-gray-300'}`} />
        <span className="font-medium text-gray-900">{agent.name}</span>
        <span className={`text-xs px-1.5 py-0.5 rounded ${getRoleBadgeColor(agent.role)}`}>
          {agent.role}
        </span>
      </div>
      {agent.model && (
        <span className="text-xs text-gray-500 truncate max-w-[150px]">
          {agent.model}
        </span>
      )}
    </button>
  );
};

/**
 * AgentSelector component for selecting chat agents.
 */
export const AgentSelector: React.FC<AgentSelectorProps> = ({
  selectedAgent,
  onSelectAgent,
  disabled = false,
  placeholder = 'Select Agent',
  className = '',
}) => {
  const [agents, setAgents] = useState<AgentInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Fetch agents on mount
  useEffect(() => {
    const fetchAgents = async () => {
      try {
        setLoading(true);
        setError(null);
        const agentList = await listAgents();
        setAgents(agentList);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load agents');
      } finally {
        setLoading(false);
      }
    };

    fetchAgents();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Get currently selected agent info
  const selectedAgentInfo = agents.find((a) => a.name === selectedAgent);

  const handleSelect = (agentName: string | null) => {
    onSelectAgent(agentName);
    setIsOpen(false);
  };

  const handleToggle = () => {
    if (!disabled) {
      setIsOpen(!isOpen);
    }
  };

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      {/* Trigger button */}
      <button
        type="button"
        onClick={handleToggle}
        disabled={disabled || loading}
        className={`
          flex items-center justify-between gap-2 px-3 py-2 w-full min-w-[180px]
          bg-white border border-gray-300 rounded-lg text-sm
          ${disabled ? 'opacity-50 cursor-not-allowed' : 'hover:border-gray-400 cursor-pointer'}
          focus:outline-none focus:ring-2 focus:ring-weaver-500 focus:border-weaver-500
        `}
      >
        <div className="flex items-center gap-2 truncate">
          {loading ? (
            <span className="text-gray-400">Loading agents...</span>
          ) : error ? (
            <span className="text-red-500 text-xs">{error}</span>
          ) : selectedAgentInfo ? (
            <>
              <span
                className={`w-2 h-2 rounded-full ${
                  selectedAgentInfo.ready ? 'bg-green-500' : 'bg-gray-300'
                }`}
              />
              <span className="font-medium text-gray-900">{selectedAgentInfo.name}</span>
              <span className={`text-xs px-1.5 py-0.5 rounded ${getRoleBadgeColor(selectedAgentInfo.role)}`}>
                {selectedAgentInfo.role}
              </span>
            </>
          ) : (
            <span className="text-gray-400">{placeholder}</span>
          )}
        </div>
        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown menu */}
      {isOpen && !loading && !error && (
        <div className="absolute z-50 mt-1 w-full min-w-[250px] bg-white border border-gray-200 rounded-lg shadow-lg overflow-hidden">
          {/* Clear selection option */}
          {selectedAgent && (
            <>
              <button
                type="button"
                onClick={() => handleSelect(null)}
                className="w-full text-left px-3 py-2 text-sm text-gray-500 hover:bg-gray-50 flex items-center gap-2"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
                Clear selection
              </button>
              <div className="border-t border-gray-100" />
            </>
          )}

          {/* Agent list */}
          {agents.length === 0 ? (
            <div className="px-3 py-4 text-center text-sm text-gray-500">
              No agents available
            </div>
          ) : (
            <div className="max-h-60 overflow-y-auto scrollbar-thin">
              {agents.map((agent) => (
                <AgentOption
                  key={agent.name}
                  agent={agent}
                  isSelected={agent.name === selectedAgent}
                  onClick={() => handleSelect(agent.name)}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default AgentSelector;
