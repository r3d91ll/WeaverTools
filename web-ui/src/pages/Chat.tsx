/**
 * Chat page - interactive chat interface with agent targeting.
 *
 * Supports @agent syntax for targeting specific agents and streaming responses.
 */
import { ChatContainer } from '@/components/chat';

/**
 * Chat page component.
 */
export const Chat: React.FC = () => {
  const handleNewSession = () => {
    // TODO: Create new session via API
  };

  return (
    <div className="h-[calc(100vh-8rem)]">
      <ChatContainer
        onNewSession={handleNewSession}
        showHeader
      />
    </div>
  );
};

export default Chat;
