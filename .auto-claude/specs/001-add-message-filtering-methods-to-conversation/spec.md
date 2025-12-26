# Add Message Filtering Methods to Conversation

## Overview

Extend Conversation with additional filter methods beyond MessagesWithHiddenStates: MessagesByRole(role), MessagesByAgent(agentID), MessagesSince(time), MessagesWithMetadata(key). These enable richer conversation analysis.

## Rationale

Conversation.MessagesWithHiddenStates() already demonstrates the pattern of filtering messages and returning a slice copy. The same thread-safe pattern (RLock, iterate, append matches, return copy) can be replicated for different filter criteria.

---
*This spec was created from ideation and is pending detailed specification.*
