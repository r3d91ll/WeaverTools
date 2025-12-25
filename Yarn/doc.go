// Package yarn manages conversations, measurements, and research session data.
//
// Yarn is the thread that connects everything - tracking WHAT HAPPENED in
// agent interactions. Just as yarn forms the continuous thread in a tapestry,
// this package weaves together the record of communication: the messages
// exchanged, the hidden states captured, and the conveyance metrics measured.
//
// The package provides the core data model for the Weaver research ecosystem,
// enabling systematic study of how meaning transfers between language models.
// It captures both the observable text of conversations and the internal
// representations (hidden states) that reveal the semantic structure beneath.
//
// # Type Hierarchy
//
// The core types form a containment hierarchy for organizing research data:
//
//	Session
//	├── Conversations []*Conversation    (one-to-many)
//	│   └── Conversation
//	│       ├── Participants map[string]Participant
//	│       └── Messages []*Message      (one-to-many)
//	│           └── Message
//	│               └── HiddenState *HiddenState  (optional)
//	└── Measurements []*Measurement      (parallel collection)
//
// # Session
//
// Session is the top-level container for a research activity. Use a Session
// when you need to:
//   - Group related conversations under a single research objective
//   - Configure measurement capture settings (passive, active, triggered)
//   - Export all data to files for later analysis
//   - Track aggregate statistics across multiple conversations
//
// Sessions own both Conversations and Measurements as parallel collections.
// This separation allows measurements to reference conversations by ID without
// creating circular dependencies, enabling flexible analysis workflows.
//
// Create a new session with NewSession:
//
//	session := yarn.NewSession("alignment-study", "Testing model alignment")
//
// # Conversation
//
// Conversation represents an ordered sequence of messages between participants.
// Use a Conversation when you need to:
//   - Track a dialogue between two or more agents
//   - Maintain message ordering and history
//   - Record which agents participated and their message counts
//   - Filter messages by properties (e.g., those with hidden states)
//
// Conversations are thread-safe and track participants automatically as
// messages are added. Each conversation maintains its own timeline independent
// of other conversations in the session.
//
// Add messages to a conversation:
//
//	conv := session.ActiveConversation()
//	conv.Add(yarn.NewAgentMessage(yarn.RoleAssistant, "Hello!", agentID, "Claude"))
//
// # Message
//
// Message is the atomic unit of communication. Use a Message when you need to:
//   - Record what an agent said (the observable text)
//   - Attach the internal representation (HiddenState) for analysis
//   - Track metadata about the generation (timing, tool calls, etc.)
//
// Messages can optionally carry a HiddenState - the boundary object that
// captures the model's semantic representation before projection to text.
// Not all messages require hidden states; attach them when you need to
// analyze the internal structure of what was communicated.
//
// Create a message with an optional hidden state:
//
//	msg := yarn.NewMessage(yarn.RoleAssistant, "The answer is 42.")
//	msg.WithHiddenState(&yarn.HiddenState{
//	    Vector: extractedVector,
//	    Layer:  24,
//	    DType:  "float32",
//	})
package yarn
