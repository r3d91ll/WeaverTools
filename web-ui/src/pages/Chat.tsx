import { useState } from 'react';

/**
 * Chat page - interactive chat interface with agent targeting.
 *
 * Supports @agent syntax for targeting specific agents and streaming responses.
 */
export const Chat: React.FC = () => {
  const [message, setMessage] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement chat message sending
    setMessage('');
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Chat Header */}
      <div className="flex items-center justify-between pb-4 border-b border-gray-200">
        <div>
          <h1 className="text-xl font-bold text-gray-900">Chat</h1>
          <p className="text-sm text-gray-500">
            Use @agent syntax to target specific agents
          </p>
        </div>
        <div className="flex items-center space-x-4">
          <select
            className="input py-2 px-3 text-sm"
            defaultValue=""
          >
            <option value="" disabled>
              Select Agent
            </option>
            <option value="researcher">Researcher</option>
            <option value="analyst">Analyst</option>
          </select>
          <button className="btn-secondary text-sm">New Session</button>
        </div>
      </div>

      {/* Chat Messages Area */}
      <div className="flex-1 overflow-y-auto py-6 scrollbar-thin">
        <div className="space-y-4">
          {/* Empty State */}
          <div className="text-center py-16">
            <div className="text-gray-400 mb-4">
              <svg
                className="w-16 h-16 mx-auto"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">
              Start a Conversation
            </h3>
            <p className="text-gray-500 max-w-sm mx-auto">
              Send a message to begin. Use @agent to target a specific agent,
              or select one from the dropdown above.
            </p>
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="pt-4 border-t border-gray-200">
        <form onSubmit={handleSubmit}>
          <div className="flex space-x-4">
            <input
              type="text"
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              placeholder="Type your message... (use @agent to target)"
              className="input flex-1 py-3 px-4"
            />
            <button
              type="submit"
              disabled={!message.trim()}
              className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default Chat;
