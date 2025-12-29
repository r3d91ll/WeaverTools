/**
 * Chat page component.
 * Provides interactive chat interface with AI agents.
 */

interface ChatProps {
  className?: string;
}

export const Chat: React.FC<ChatProps> = ({ className }) => {
  return (
    <div className={`flex flex-col h-full ${className ?? ''}`}>
      <div className="mb-4">
        <h1 className="text-3xl font-bold text-gray-900">Chat</h1>
        <p className="mt-2 text-gray-600">
          Interact with AI agents using @agent syntax
        </p>
      </div>

      <div className="flex-1 flex flex-col card min-h-[500px]">
        {/* Agent Selector */}
        <div className="flex items-center gap-4 pb-4 border-b">
          <label className="text-sm font-medium text-gray-700">
            Target Agent:
          </label>
          <select className="input max-w-xs">
            <option value="">Select an agent...</option>
          </select>
        </div>

        {/* Message Area */}
        <div className="flex-1 overflow-y-auto py-4">
          <div className="flex items-center justify-center h-full text-gray-400">
            <p>No messages yet. Start a conversation with an agent.</p>
          </div>
        </div>

        {/* Input Area */}
        <div className="pt-4 border-t">
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Type your message... (use @agent to target specific agent)"
              className="input flex-1"
            />
            <button className="btn-primary">Send</button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;
