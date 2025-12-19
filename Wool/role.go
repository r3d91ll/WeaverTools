// Package wool defines agent roles and capabilities.
// Wool is the raw material that becomes agents - defining WHAT an agent IS.
package wool

// Role defines an agent's function in the multi-agent system.
type Role string

const (
	// RoleSenior is for high-level reasoning, architecture decisions, and orchestration.
	// Typically uses Claude Code (opaque, no hidden state access).
	RoleSenior Role = "senior"

	// RoleJunior is for implementation tasks, file operations, and routine work.
	// Has tool access. Uses The Loom for hidden state extraction.
	RoleJunior Role = "junior"

	// RoleConversant participates in bilateral exchanges for conveyance measurement.
	// No tools. Hidden states are extracted during conversations.
	RoleConversant Role = "conversant"

	// RoleSubject is a single agent being studied in experiments.
	// No tools. Hidden states are measured without bilateral exchange.
	RoleSubject Role = "subject"

	// RoleObserver is for passive monitoring and logging.
	// Receives messages but doesn't generate responses.
	RoleObserver Role = "observer"
)

// String returns the string representation of the role.
func (r Role) String() string {
	return string(r)
}

// IsValid returns true if this is a valid role.
func (r Role) IsValid() bool {
	switch r {
	case RoleSenior, RoleJunior, RoleConversant, RoleSubject, RoleObserver:
		return true
	default:
		return false
	}
}

// RequiresHiddenStates returns true if this role needs hidden state extraction.
func (r Role) RequiresHiddenStates() bool {
	switch r {
	case RoleConversant, RoleSubject:
		return true
	case RoleJunior:
		return true // Optional but useful for measurement
	default:
		return false
	}
}

// SupportsTools returns true if this role can use tools.
func (r Role) SupportsTools() bool {
	switch r {
	case RoleSenior, RoleJunior:
		return true
	default:
		return false
	}
}

// CanGenerateResponses returns true if this role can produce messages.
func (r Role) CanGenerateResponses() bool {
	return r != RoleObserver
}

// Description returns a human-readable description of the role.
func (r Role) Description() string {
	switch r {
	case RoleSenior:
		return "High-level reasoning, architecture decisions, orchestration"
	case RoleJunior:
		return "Implementation tasks, file operations, tool execution"
	case RoleConversant:
		return "Bilateral exchange participant for conveyance measurement"
	case RoleSubject:
		return "Single agent target for conveyance measurement"
	case RoleObserver:
		return "Passive monitoring, logging, no response generation"
	default:
		return "Unknown role"
	}
}
