package wool

import (
	"testing"
)

// TestRoleIsValid tests the IsValid method for Role.
func TestRoleIsValid(t *testing.T) {
	tests := []struct {
		name  string
		role  Role
		valid bool
	}{
		{"senior role is valid", RoleSenior, true},
		{"junior role is valid", RoleJunior, true},
		{"conversant role is valid", RoleConversant, true},
		{"subject role is valid", RoleSubject, true},
		{"observer role is valid", RoleObserver, true},
		{"empty role is invalid", Role(""), false},
		{"unknown role is invalid", Role("unknown"), false},
		{"typo role is invalid", Role("seniior"), false},
		{"capitalized role is invalid", Role("Senior"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.role.IsValid(); got != tt.valid {
				t.Errorf("Role(%q).IsValid() = %v, want %v", tt.role, got, tt.valid)
			}
		})
	}
}

// TestRoleSupportsTools tests the SupportsTools method for Role.
func TestRoleSupportsTools(t *testing.T) {
	tests := []struct {
		name     string
		role     Role
		supports bool
	}{
		{"senior supports tools", RoleSenior, true},
		{"junior supports tools", RoleJunior, true},
		{"conversant does not support tools", RoleConversant, false},
		{"subject does not support tools", RoleSubject, false},
		{"observer does not support tools", RoleObserver, false},
		{"empty role does not support tools", Role(""), false},
		{"unknown role does not support tools", Role("unknown"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.role.SupportsTools(); got != tt.supports {
				t.Errorf("Role(%q).SupportsTools() = %v, want %v", tt.role, got, tt.supports)
			}
		})
	}
}

// TestRoleRequiresHiddenStates tests the RequiresHiddenStates method for Role.
func TestRoleRequiresHiddenStates(t *testing.T) {
	tests := []struct {
		name     string
		role     Role
		requires bool
	}{
		{"senior does not require hidden states", RoleSenior, false},
		{"junior requires hidden states", RoleJunior, true},
		{"conversant requires hidden states", RoleConversant, true},
		{"subject requires hidden states", RoleSubject, true},
		{"observer does not require hidden states", RoleObserver, false},
		{"empty role does not require hidden states", Role(""), false},
		{"unknown role does not require hidden states", Role("unknown"), false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.role.RequiresHiddenStates(); got != tt.requires {
				t.Errorf("Role(%q).RequiresHiddenStates() = %v, want %v", tt.role, got, tt.requires)
			}
		})
	}
}

// TestRoleCanGenerateResponses tests the CanGenerateResponses method for Role.
func TestRoleCanGenerateResponses(t *testing.T) {
	tests := []struct {
		name        string
		role        Role
		canGenerate bool
	}{
		{"senior can generate responses", RoleSenior, true},
		{"junior can generate responses", RoleJunior, true},
		{"conversant can generate responses", RoleConversant, true},
		{"subject can generate responses", RoleSubject, true},
		{"observer cannot generate responses", RoleObserver, false},
		{"empty role can generate responses", Role(""), true},
		{"unknown role can generate responses", Role("unknown"), true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.role.CanGenerateResponses(); got != tt.canGenerate {
				t.Errorf("Role(%q).CanGenerateResponses() = %v, want %v", tt.role, got, tt.canGenerate)
			}
		})
	}
}

// TestRoleDescription tests the Description method for Role.
func TestRoleDescription(t *testing.T) {
	tests := []struct {
		name     string
		role     Role
		wantDesc string
	}{
		{"senior description", RoleSenior, "High-level reasoning, architecture decisions, orchestration"},
		{"junior description", RoleJunior, "Implementation tasks, file operations, tool execution"},
		{"conversant description", RoleConversant, "Bilateral exchange participant for conveyance measurement"},
		{"subject description", RoleSubject, "Single agent target for conveyance measurement"},
		{"observer description", RoleObserver, "Passive monitoring, logging, no response generation"},
		{"unknown role description", Role("unknown"), "Unknown role"},
		{"empty role description", Role(""), "Unknown role"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.role.Description(); got != tt.wantDesc {
				t.Errorf("Role(%q).Description() = %q, want %q", tt.role, got, tt.wantDesc)
			}
		})
	}
}

// TestRoleString tests the String method for Role.
func TestRoleString(t *testing.T) {
	tests := []struct {
		name    string
		role    Role
		wantStr string
	}{
		{"senior string", RoleSenior, "senior"},
		{"junior string", RoleJunior, "junior"},
		{"conversant string", RoleConversant, "conversant"},
		{"subject string", RoleSubject, "subject"},
		{"observer string", RoleObserver, "observer"},
		{"empty string", Role(""), ""},
		{"unknown string", Role("custom"), "custom"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.role.String(); got != tt.wantStr {
				t.Errorf("Role(%q).String() = %q, want %q", tt.role, got, tt.wantStr)
			}
		})
	}
}

// TestRoleCapabilitiesConsistency tests that role capabilities are consistent.
// A role that doesn't support tools shouldn't require hidden states for tool use.
func TestRoleCapabilitiesConsistency(t *testing.T) {
	roles := []Role{RoleSenior, RoleJunior, RoleConversant, RoleSubject, RoleObserver}

	for _, role := range roles {
		t.Run(string(role)+"_capabilities_are_consistent", func(t *testing.T) {
			// All valid roles should be valid
			if !role.IsValid() {
				t.Errorf("Role(%q) should be valid", role)
			}

			// Observer role should not generate responses
			if role == RoleObserver && role.CanGenerateResponses() {
				t.Errorf("Role(%q).CanGenerateResponses() should be false", role)
			}

			// Only senior and junior should support tools
			if role.SupportsTools() && role != RoleSenior && role != RoleJunior {
				t.Errorf("Role(%q).SupportsTools() should be false", role)
			}

			// Description should never be empty for valid roles
			if role.Description() == "" {
				t.Errorf("Role(%q).Description() should not be empty", role)
			}

			// Description should not be "Unknown role" for valid roles
			if role.Description() == "Unknown role" {
				t.Errorf("Role(%q).Description() should not be 'Unknown role'", role)
			}

			// String representation should match the role value
			if role.String() != string(role) {
				t.Errorf("Role(%q).String() = %q, want %q", role, role.String(), string(role))
			}
		})
	}
}

// TestRoleConstants tests that role constants have expected values.
func TestRoleConstants(t *testing.T) {
	tests := []struct {
		name     string
		role     Role
		wantVal  string
	}{
		{"RoleSenior value", RoleSenior, "senior"},
		{"RoleJunior value", RoleJunior, "junior"},
		{"RoleConversant value", RoleConversant, "conversant"},
		{"RoleSubject value", RoleSubject, "subject"},
		{"RoleObserver value", RoleObserver, "observer"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.role) != tt.wantVal {
				t.Errorf("Role constant %s = %q, want %q", tt.name, tt.role, tt.wantVal)
			}
		})
	}
}

// TestRoleToolSupportAndHiddenStatesRelationship tests the relationship
// between tool support and hidden state requirements.
func TestRoleToolSupportAndHiddenStatesRelationship(t *testing.T) {
	// Senior: tools yes, hidden states no (uses opaque Claude Code)
	t.Run("senior_tools_yes_hidden_states_no", func(t *testing.T) {
		if !RoleSenior.SupportsTools() {
			t.Error("RoleSenior should support tools")
		}
		if RoleSenior.RequiresHiddenStates() {
			t.Error("RoleSenior should not require hidden states (uses opaque Claude Code)")
		}
	})

	// Junior: tools yes, hidden states yes (uses The Loom)
	t.Run("junior_tools_yes_hidden_states_yes", func(t *testing.T) {
		if !RoleJunior.SupportsTools() {
			t.Error("RoleJunior should support tools")
		}
		if !RoleJunior.RequiresHiddenStates() {
			t.Error("RoleJunior should require hidden states (uses The Loom)")
		}
	})

	// Conversant: tools no, hidden states yes (for conveyance measurement)
	t.Run("conversant_tools_no_hidden_states_yes", func(t *testing.T) {
		if RoleConversant.SupportsTools() {
			t.Error("RoleConversant should not support tools")
		}
		if !RoleConversant.RequiresHiddenStates() {
			t.Error("RoleConversant should require hidden states (for measurement)")
		}
	})

	// Subject: tools no, hidden states yes (for measurement)
	t.Run("subject_tools_no_hidden_states_yes", func(t *testing.T) {
		if RoleSubject.SupportsTools() {
			t.Error("RoleSubject should not support tools")
		}
		if !RoleSubject.RequiresHiddenStates() {
			t.Error("RoleSubject should require hidden states (for measurement)")
		}
	})

	// Observer: tools no, hidden states no (passive)
	t.Run("observer_tools_no_hidden_states_no", func(t *testing.T) {
		if RoleObserver.SupportsTools() {
			t.Error("RoleObserver should not support tools")
		}
		if RoleObserver.RequiresHiddenStates() {
			t.Error("RoleObserver should not require hidden states (passive)")
		}
	})
}

// TestInvalidRoleBehavior tests behavior of invalid roles across all methods.
func TestInvalidRoleBehavior(t *testing.T) {
	invalidRoles := []Role{
		Role(""),
		Role("invalid"),
		Role("SENIOR"),
		Role("Junior"),
		Role("admin"),
		Role("supervisor"),
	}

	for _, role := range invalidRoles {
		t.Run("invalid_role_"+string(role), func(t *testing.T) {
			// Should not be valid
			if role.IsValid() {
				t.Errorf("Role(%q).IsValid() should be false", role)
			}

			// Should not support tools
			if role.SupportsTools() {
				t.Errorf("Role(%q).SupportsTools() should be false", role)
			}

			// Should not require hidden states
			if role.RequiresHiddenStates() {
				t.Errorf("Role(%q).RequiresHiddenStates() should be false", role)
			}

			// Should return "Unknown role" description
			if role.Description() != "Unknown role" {
				t.Errorf("Role(%q).Description() = %q, want %q", role, role.Description(), "Unknown role")
			}

			// String should still return the raw value
			if role.String() != string(role) {
				t.Errorf("Role(%q).String() = %q, want %q", role, role.String(), string(role))
			}

			// Can generate responses (default behavior)
			// Only RoleObserver explicitly cannot generate responses
			if !role.CanGenerateResponses() {
				t.Errorf("Role(%q).CanGenerateResponses() should be true (default)", role)
			}
		})
	}
}
