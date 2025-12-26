package yarn

import (
	"sync"
	"testing"
	"time"
)

// TestMessagesByRole_Basic tests basic filtering by role.
func TestMessagesByRole_Basic(t *testing.T) {
	conv := NewConversation("test")

	// Add messages with different roles
	conv.Add(NewMessage(RoleSystem, "system message"))
	conv.Add(NewMessage(RoleUser, "user message 1"))
	conv.Add(NewMessage(RoleAssistant, "assistant message"))
	conv.Add(NewMessage(RoleUser, "user message 2"))
	conv.Add(NewMessage(RoleTool, "tool message"))

	// Test filtering by each role
	tests := []struct {
		role     MessageRole
		expected int
	}{
		{RoleSystem, 1},
		{RoleUser, 2},
		{RoleAssistant, 1},
		{RoleTool, 1},
	}

	for _, tc := range tests {
		result := conv.MessagesByRole(tc.role)
		if len(result) != tc.expected {
			t.Errorf("MessagesByRole(%s): expected %d, got %d", tc.role, tc.expected, len(result))
		}

		// Verify all returned messages have the correct role
		for _, msg := range result {
			if msg.Role != tc.role {
				t.Errorf("MessagesByRole(%s): returned message with role %s", tc.role, msg.Role)
			}
		}
	}
}

// TestMessagesByRole_EmptyConversation tests filtering on empty conversation.
func TestMessagesByRole_EmptyConversation(t *testing.T) {
	conv := NewConversation("empty")

	result := conv.MessagesByRole(RoleUser)
	if result == nil {
		t.Error("MessagesByRole on empty conversation should return nil, not panic")
	}
	if len(result) != 0 {
		t.Errorf("MessagesByRole on empty conversation: expected 0, got %d", len(result))
	}
}

// TestMessagesByRole_NoMatches tests when no messages match the role.
func TestMessagesByRole_NoMatches(t *testing.T) {
	conv := NewConversation("test")
	conv.Add(NewMessage(RoleUser, "user message"))
	conv.Add(NewMessage(RoleAssistant, "assistant message"))

	result := conv.MessagesByRole(RoleTool)
	if len(result) != 0 {
		t.Errorf("MessagesByRole with no matches: expected 0, got %d", len(result))
	}
}

// TestMessagesByRole_AllMatch tests when all messages match.
func TestMessagesByRole_AllMatch(t *testing.T) {
	conv := NewConversation("test")
	conv.Add(NewMessage(RoleUser, "message 1"))
	conv.Add(NewMessage(RoleUser, "message 2"))
	conv.Add(NewMessage(RoleUser, "message 3"))

	result := conv.MessagesByRole(RoleUser)
	if len(result) != 3 {
		t.Errorf("MessagesByRole all match: expected 3, got %d", len(result))
	}
}

// TestMessagesByAgent_Basic tests basic filtering by agent ID.
func TestMessagesByAgent_Basic(t *testing.T) {
	conv := NewConversation("test")

	conv.Add(NewAgentMessage(RoleAssistant, "message 1", "agent-1", "Agent One"))
	conv.Add(NewAgentMessage(RoleAssistant, "message 2", "agent-2", "Agent Two"))
	conv.Add(NewAgentMessage(RoleAssistant, "message 3", "agent-1", "Agent One"))
	conv.Add(NewAgentMessage(RoleAssistant, "message 4", "agent-3", "Agent Three"))

	result := conv.MessagesByAgent("agent-1")
	if len(result) != 2 {
		t.Errorf("MessagesByAgent(agent-1): expected 2, got %d", len(result))
	}

	// Verify all returned messages have the correct agent ID
	for _, msg := range result {
		if msg.AgentID != "agent-1" {
			t.Errorf("MessagesByAgent(agent-1): returned message with agent ID %s", msg.AgentID)
		}
	}
}

// TestMessagesByAgent_EmptyConversation tests filtering on empty conversation.
func TestMessagesByAgent_EmptyConversation(t *testing.T) {
	conv := NewConversation("empty")

	result := conv.MessagesByAgent("any-agent")
	if result == nil {
		t.Error("MessagesByAgent on empty conversation should return nil, not panic")
	}
	if len(result) != 0 {
		t.Errorf("MessagesByAgent on empty conversation: expected 0, got %d", len(result))
	}
}

// TestMessagesByAgent_NoMatches tests when no messages match the agent ID.
func TestMessagesByAgent_NoMatches(t *testing.T) {
	conv := NewConversation("test")
	conv.Add(NewAgentMessage(RoleAssistant, "message", "agent-1", "Agent One"))
	conv.Add(NewAgentMessage(RoleAssistant, "message", "agent-2", "Agent Two"))

	result := conv.MessagesByAgent("agent-99")
	if len(result) != 0 {
		t.Errorf("MessagesByAgent with no matches: expected 0, got %d", len(result))
	}
}

// TestMessagesByAgent_EmptyAgentID tests filtering for empty agent ID.
func TestMessagesByAgent_EmptyAgentID(t *testing.T) {
	conv := NewConversation("test")

	// Add messages with and without agent IDs
	conv.Add(NewMessage(RoleUser, "user message")) // No agent ID
	conv.Add(NewAgentMessage(RoleAssistant, "agent message", "agent-1", "Agent One"))
	conv.Add(NewMessage(RoleUser, "another user message")) // No agent ID

	result := conv.MessagesByAgent("")
	if len(result) != 2 {
		t.Errorf("MessagesByAgent(''): expected 2 messages with empty AgentID, got %d", len(result))
	}

	// Verify all returned messages have empty agent ID
	for _, msg := range result {
		if msg.AgentID != "" {
			t.Errorf("MessagesByAgent(''): returned message with non-empty agent ID %s", msg.AgentID)
		}
	}
}

// TestMessagesSince_Basic tests basic filtering by time.
func TestMessagesSince_Basic(t *testing.T) {
	conv := NewConversation("test")

	// Create messages with controlled timestamps
	baseTime := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)

	msg1 := NewMessage(RoleUser, "old message 1")
	msg1.Timestamp = baseTime.Add(-2 * time.Hour)
	conv.Add(msg1)

	msg2 := NewMessage(RoleUser, "old message 2")
	msg2.Timestamp = baseTime.Add(-1 * time.Hour)
	conv.Add(msg2)

	msg3 := NewMessage(RoleUser, "new message 1")
	msg3.Timestamp = baseTime.Add(1 * time.Hour)
	conv.Add(msg3)

	msg4 := NewMessage(RoleUser, "new message 2")
	msg4.Timestamp = baseTime.Add(2 * time.Hour)
	conv.Add(msg4)

	result := conv.MessagesSince(baseTime)
	if len(result) != 2 {
		t.Errorf("MessagesSince: expected 2 messages after baseTime, got %d", len(result))
	}

	// Verify messages are in chronological order
	for i := 1; i < len(result); i++ {
		if result[i].Timestamp.Before(result[i-1].Timestamp) {
			t.Error("MessagesSince: messages not in chronological order")
		}
	}
}

// TestMessagesSince_EmptyConversation tests filtering on empty conversation.
func TestMessagesSince_EmptyConversation(t *testing.T) {
	conv := NewConversation("empty")

	result := conv.MessagesSince(time.Now())
	if result == nil {
		t.Error("MessagesSince on empty conversation should return nil, not panic")
	}
	if len(result) != 0 {
		t.Errorf("MessagesSince on empty conversation: expected 0, got %d", len(result))
	}
}

// TestMessagesSince_NoMatches tests when no messages are after the given time.
func TestMessagesSince_NoMatches(t *testing.T) {
	conv := NewConversation("test")

	// Add old messages
	oldTime := time.Date(2020, 1, 1, 0, 0, 0, 0, time.UTC)
	msg := NewMessage(RoleUser, "old message")
	msg.Timestamp = oldTime
	conv.Add(msg)

	// Filter for messages after current time
	result := conv.MessagesSince(time.Now())
	if len(result) != 0 {
		t.Errorf("MessagesSince with no matches: expected 0, got %d", len(result))
	}
}

// TestMessagesSince_AllMatch tests when all messages are after the given time.
func TestMessagesSince_AllMatch(t *testing.T) {
	conv := NewConversation("test")

	// Add messages with recent timestamps
	conv.Add(NewMessage(RoleUser, "message 1"))
	conv.Add(NewMessage(RoleUser, "message 2"))
	conv.Add(NewMessage(RoleUser, "message 3"))

	// Filter for messages after a very old time
	oldTime := time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC)
	result := conv.MessagesSince(oldTime)
	if len(result) != 3 {
		t.Errorf("MessagesSince all match: expected 3, got %d", len(result))
	}
}

// TestMessagesSince_ExactTimestamp tests that messages at exact timestamp are excluded.
func TestMessagesSince_ExactTimestamp(t *testing.T) {
	conv := NewConversation("test")

	exactTime := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)

	msg := NewMessage(RoleUser, "exact time message")
	msg.Timestamp = exactTime
	conv.Add(msg)

	// MessagesSince should be strictly after, so exact match should NOT be included
	result := conv.MessagesSince(exactTime)
	if len(result) != 0 {
		t.Errorf("MessagesSince at exact timestamp: expected 0 (strictly after), got %d", len(result))
	}
}

// TestMessagesWithMetadata_Basic tests basic filtering by metadata key.
func TestMessagesWithMetadata_Basic(t *testing.T) {
	conv := NewConversation("test")

	msg1 := NewMessage(RoleUser, "message 1")
	msg1.WithMetadata("important", true)
	conv.Add(msg1)

	msg2 := NewMessage(RoleUser, "message 2")
	msg2.WithMetadata("other_key", "value")
	conv.Add(msg2)

	msg3 := NewMessage(RoleUser, "message 3")
	msg3.WithMetadata("important", false)
	conv.Add(msg3)

	result := conv.MessagesWithMetadata("important")
	if len(result) != 2 {
		t.Errorf("MessagesWithMetadata('important'): expected 2, got %d", len(result))
	}

	// Verify all returned messages have the key
	for _, msg := range result {
		if _, exists := msg.Metadata["important"]; !exists {
			t.Error("MessagesWithMetadata: returned message without the specified key")
		}
	}
}

// TestMessagesWithMetadata_EmptyConversation tests filtering on empty conversation.
func TestMessagesWithMetadata_EmptyConversation(t *testing.T) {
	conv := NewConversation("empty")

	result := conv.MessagesWithMetadata("any_key")
	if result == nil {
		t.Error("MessagesWithMetadata on empty conversation should return nil, not panic")
	}
	if len(result) != 0 {
		t.Errorf("MessagesWithMetadata on empty conversation: expected 0, got %d", len(result))
	}
}

// TestMessagesWithMetadata_NoMatches tests when no messages have the key.
func TestMessagesWithMetadata_NoMatches(t *testing.T) {
	conv := NewConversation("test")

	msg := NewMessage(RoleUser, "message")
	msg.WithMetadata("other_key", "value")
	conv.Add(msg)

	result := conv.MessagesWithMetadata("nonexistent_key")
	if len(result) != 0 {
		t.Errorf("MessagesWithMetadata with no matches: expected 0, got %d", len(result))
	}
}

// TestMessagesWithMetadata_NilMetadata tests handling of nil Metadata maps.
func TestMessagesWithMetadata_NilMetadata(t *testing.T) {
	conv := NewConversation("test")

	// Create a message and explicitly set Metadata to nil
	msg1 := &Message{
		ID:        "msg-1",
		Role:      RoleUser,
		Content:   "message with nil metadata",
		Timestamp: time.Now(),
		Metadata:  nil, // Explicitly nil
	}
	conv.Add(msg1)

	msg2 := NewMessage(RoleUser, "message with metadata")
	msg2.WithMetadata("key", "value")
	conv.Add(msg2)

	// Should not panic and should only return the message with metadata
	result := conv.MessagesWithMetadata("key")
	if len(result) != 1 {
		t.Errorf("MessagesWithMetadata with nil Metadata: expected 1, got %d", len(result))
	}
}

// TestMessagesWithMetadata_EmptyKey tests filtering with empty key string.
func TestMessagesWithMetadata_EmptyKey(t *testing.T) {
	conv := NewConversation("test")

	msg1 := NewMessage(RoleUser, "message 1")
	msg1.WithMetadata("", "empty key value")
	conv.Add(msg1)

	msg2 := NewMessage(RoleUser, "message 2")
	msg2.WithMetadata("normal_key", "value")
	conv.Add(msg2)

	result := conv.MessagesWithMetadata("")
	if len(result) != 1 {
		t.Errorf("MessagesWithMetadata(''): expected 1, got %d", len(result))
	}
}

// TestMessagesWithMetadata_NilValue tests that nil values in metadata are matched.
func TestMessagesWithMetadata_NilValue(t *testing.T) {
	conv := NewConversation("test")

	msg := NewMessage(RoleUser, "message")
	msg.WithMetadata("key_with_nil", nil)
	conv.Add(msg)

	result := conv.MessagesWithMetadata("key_with_nil")
	if len(result) != 1 {
		t.Errorf("MessagesWithMetadata with nil value: expected 1, got %d", len(result))
	}
}

// TestConcurrentAccess tests thread safety of filter methods.
func TestConcurrentAccess(t *testing.T) {
	conv := NewConversation("concurrent-test")

	// Add some initial messages
	for i := 0; i < 10; i++ {
		msg := NewAgentMessage(RoleAssistant, "message", "agent-1", "Agent")
		msg.WithMetadata("key", i)
		conv.Add(msg)
	}

	var wg sync.WaitGroup
	errChan := make(chan error, 100)

	// Concurrent readers
	for i := 0; i < 10; i++ {
		wg.Add(4)

		go func() {
			defer wg.Done()
			_ = conv.MessagesByRole(RoleAssistant)
		}()

		go func() {
			defer wg.Done()
			_ = conv.MessagesByAgent("agent-1")
		}()

		go func() {
			defer wg.Done()
			_ = conv.MessagesSince(time.Now().Add(-1 * time.Hour))
		}()

		go func() {
			defer wg.Done()
			_ = conv.MessagesWithMetadata("key")
		}()
	}

	// Concurrent writer
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			msg := NewMessage(RoleUser, "new message")
			msg.WithMetadata("new_key", i)
			conv.Add(msg)
		}
	}()

	wg.Wait()
	close(errChan)

	// Check for any errors
	for err := range errChan {
		t.Error(err)
	}
}

// TestFilterMethodsReturnCopy verifies that filter methods return copies, not references.
func TestFilterMethodsReturnCopy(t *testing.T) {
	conv := NewConversation("test")
	conv.Add(NewMessage(RoleUser, "message 1"))
	conv.Add(NewMessage(RoleUser, "message 2"))

	result1 := conv.MessagesByRole(RoleUser)
	result2 := conv.MessagesByRole(RoleUser)

	// Modifying one result shouldn't affect the other
	if len(result1) < 2 {
		t.Fatal("Expected at least 2 messages")
	}

	// Change the first element of result1
	result1[0] = NewMessage(RoleAssistant, "modified")

	// result2 should still have the original message
	if result2[0].Role != RoleUser {
		t.Error("Filter methods should return slice copies, not shared references")
	}
}

// TestMessagesByRole_PreservesOrder tests that messages are returned in order.
func TestMessagesByRole_PreservesOrder(t *testing.T) {
	conv := NewConversation("test")

	msg1 := NewMessage(RoleUser, "first")
	msg2 := NewMessage(RoleAssistant, "second")
	msg3 := NewMessage(RoleUser, "third")
	msg4 := NewMessage(RoleUser, "fourth")

	conv.Add(msg1)
	conv.Add(msg2)
	conv.Add(msg3)
	conv.Add(msg4)

	result := conv.MessagesByRole(RoleUser)
	if len(result) != 3 {
		t.Fatalf("Expected 3 messages, got %d", len(result))
	}

	if result[0].Content != "first" || result[1].Content != "third" || result[2].Content != "fourth" {
		t.Error("MessagesByRole should preserve message order")
	}
}
