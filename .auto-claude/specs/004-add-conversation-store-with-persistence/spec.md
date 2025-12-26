# Add Conversation Store with Persistence

## Overview

Create a ConversationStore that mirrors the existing Concept Store pattern - thread-safe storage for conversations with Save/Load file persistence to JSON. This enables research session replay and conversation history analysis across runs.

## Rationale

The Concept Store in concepts/store.go establishes a mature pattern for thread-safe storage with JSON file persistence (Save/Load methods). Conversations currently exist only in-memory within Sessions. Applying this exact pattern to Conversations would be straightforward since the data structures are similar.

---
*This spec was created from ideation and is pending detailed specification.*
