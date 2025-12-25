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
//
// # HiddenState (Boundary Object)
//
// HiddenState is the boundary object that bridges the gap between what a model
// "thinks" and what it says. It captures the semantic representation from an
// intermediate layer of the language model - the high-dimensional vector that
// exists just before projection to token probabilities.
//
// Use HiddenState when you need to:
//   - Analyze the internal structure of what was communicated
//   - Compare semantic representations between sender and receiver
//   - Compute conveyance metrics (DEff, Beta, Alignment, CPair)
//   - Study how meaning transforms as it passes between models
//
// A HiddenState contains:
//   - Vector: The hidden state values (typically 2048-8192 float32 values)
//   - Shape: Original tensor dimensions, e.g., [1, seq_len, hidden_dim]
//   - Layer: Which transformer layer this was extracted from
//   - DType: Data type, typically "float32"
//
// Memory note: Hidden states can be large (8-32KB each). Consider streaming
// or lazy loading when processing many messages with hidden states.
//
// # Measurement (Conveyance Metrics)
//
// Measurement captures conveyance metrics from a single point in a conversation.
// While Messages store what was said, Measurements quantify how effectively
// meaning transferred between agents by analyzing their hidden states.
//
// Measurements live as a parallel collection under Session (not nested under
// Conversation). They reference Conversation and Session by ID, enabling
// flexible analysis workflows where measurements can be computed lazily or
// by external analysis services.
//
// Use Measurement when you need to:
//   - Quantify how well meaning transferred in a conversation turn
//   - Track dimensional health of the representation space over time
//   - Detect semantic collapse or drift patterns
//   - Compare bilateral (both directions) vs unilateral conveyance
//
// Core conveyance metrics:
//
//	DEff (Effective Dimensionality): How many dimensions carry meaningful
//	information. Higher is better - indicates a rich representation space.
//
//	Beta (β): Collapse indicator from the Conveyance Framework. Lower values
//	indicate better dimensional preservation. Quality thresholds:
//	  - Optimal:    β ∈ [1.5, 2.0) - ideal range
//	  - Monitor:    β ∈ [2.0, 2.5) - acceptable, watch for drift
//	  - Concerning: β ∈ [2.5, 3.0) - dimensional compression detected
//	  - Critical:   β ≥ 3.0       - severe collapse, intervention needed
//
//	Alignment: Cosine similarity between sender and receiver hidden states.
//	Measures how similar the semantic representations are (range: -1 to 1).
//
//	CPair (Bilateral Conveyance): Combined conveyance score when both sender
//	and receiver hidden states are available. Requires IsBilateral() == true.
//
// Create a measurement for a conversation turn:
//
//	m := yarn.NewMeasurementForTurn(session.ID, conv.ID, turnNumber)
//	m.SetSender(agentID, "Claude", "assistant", senderHidden)
//	m.SetReceiver(agentID, "User", "user", receiverHidden)
//	m.DEff = 1847
//	m.Beta = 1.72
//	m.BetaStatus = yarn.ComputeBetaStatus(m.Beta)
//
// # Example: Complete Research Workflow
//
// This example demonstrates a complete workflow: creating a session, building
// a conversation with messages, attaching hidden states for analysis, and
// recording conveyance measurements.
//
//	// 1. Create a research session
//	session := yarn.NewSession("alignment-study", "Testing Claude alignment")
//	session.Config.MeasurementMode = yarn.MeasureActive
//
//	// 2. Get or create a conversation
//	conv := session.ActiveConversation()
//
//	// 3. Add messages to the conversation
//	userMsg := yarn.NewAgentMessage(
//	    yarn.RoleUser,
//	    "What is the meaning of life?",
//	    "user-001",
//	    "Human",
//	)
//	conv.Add(userMsg)
//
//	// 4. Create an assistant message with hidden state
//	assistantMsg := yarn.NewAgentMessage(
//	    yarn.RoleAssistant,
//	    "The meaning of life is a philosophical question...",
//	    "claude-001",
//	    "Claude",
//	)
//
//	// Attach the hidden state (boundary object from model internals)
//	assistantMsg.WithHiddenState(&yarn.HiddenState{
//	    Vector: extractedVector,  // []float32 from model layer
//	    Shape:  []int{1, 1, 4096},
//	    Layer:  24,
//	    DType:  "float32",
//	})
//	conv.Add(assistantMsg)
//
//	// 5. Record a conveyance measurement for this exchange
//	m := yarn.NewMeasurementForTurn(session.ID, conv.ID, 1)
//	m.SetSender("claude-001", "Claude", "assistant", assistantMsg.HiddenState)
//	m.SetReceiver("user-001", "Human", "user", nil) // No hidden state for human
//	m.IsUnilateral = true // Only sender has hidden state
//
//	// Set the computed metrics (from your analysis code)
//	m.DEff = 1847
//	m.Beta = 1.72
//	m.BetaStatus = yarn.ComputeBetaStatus(m.Beta) // Returns BetaOptimal
//	m.Alignment = 0.89
//	m.MessageContent = assistantMsg.Content
//
//	// Add measurement to session
//	session.AddMeasurement(m)
//
//	// 6. Check session statistics
//	stats := session.Stats()
//	fmt.Printf("Messages: %d, Measurements: %d, Avg β: %.2f\n",
//	    stats.MessageCount, stats.MeasurementCount, stats.AvgBeta)
//
//	// 7. End session and export data
//	session.End()
//	if err := session.Export(); err != nil {
//	    log.Fatal(err)
//	}
//
// # Example: Bilateral Measurement
//
// When both sender and receiver are language models with hidden states,
// you can perform bilateral conveyance analysis:
//
//	// Two AI agents conversing
//	claudeMsg := yarn.NewAgentMessage(yarn.RoleAssistant, "Hello!", "claude-001", "Claude")
//	claudeMsg.WithHiddenState(&yarn.HiddenState{Vector: claudeVector, Layer: 24, DType: "float32"})
//	conv.Add(claudeMsg)
//
//	gptMsg := yarn.NewAgentMessage(yarn.RoleAssistant, "Hi there!", "gpt-001", "GPT-4")
//	gptMsg.WithHiddenState(&yarn.HiddenState{Vector: gptVector, Layer: 32, DType: "float32"})
//	conv.Add(gptMsg)
//
//	// Bilateral measurement with both hidden states
//	m := yarn.NewMeasurementForTurn(session.ID, conv.ID, 1)
//	m.SetSender("claude-001", "Claude", "assistant", claudeMsg.HiddenState)
//	m.SetReceiver("gpt-001", "GPT-4", "assistant", gptMsg.HiddenState)
//
//	// Now IsBilateral() returns true, enabling CPair analysis
//	if m.IsBilateral() {
//	    m.CPair = computeBilateralConveyance(m.SenderHidden, m.ReceiverHidden)
//	}
//
// # Example: Filtering Messages with Hidden States
//
// When you need to analyze only messages that have hidden state data:
//
//	// Get all messages with hidden states for batch analysis
//	messagesWithStates := conv.MessagesWithHiddenStates()
//	for _, msg := range messagesWithStates {
//	    dim := msg.HiddenState.Dimension()
//	    fmt.Printf("%s: %d-dimensional hidden state\n", msg.AgentName, dim)
//	}
//
//	// Check individual messages
//	if msg.HasHiddenState() {
//	    analyzeHiddenState(msg.HiddenState)
//	}
package yarn
