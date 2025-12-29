package backend

import (
	"strings"
	"testing"

	werrors "github.com/r3d91ll/weaver/pkg/errors"
)

// TestClaudeCodeConfig_Validate tests the Validate() method for ClaudeCodeConfig
// using table-driven tests to cover all validation scenarios.
func TestClaudeCodeConfig_Validate(t *testing.T) {
	tests := []struct {
		name           string
		config         ClaudeCodeConfig
		wantErr        bool
		wantCode       string
		wantFieldInMsg string // Field name that should appear in error message/context
	}{
		{
			name:    "empty config passes",
			config:  ClaudeCodeConfig{},
			wantErr: false,
		},
		{
			name: "valid config with all fields",
			config: ClaudeCodeConfig{
				Name:         "my-claude-backend",
				SystemPrompt: "You are a helpful assistant.",
				ContextLimit: 100000,
				MaxTokens:    50000,
			},
			wantErr: false,
		},
		{
			name: "valid config with underscores in name",
			config: ClaudeCodeConfig{
				Name: "claude_code_1",
			},
			wantErr: false,
		},
		{
			name: "valid config with hyphens in name",
			config: ClaudeCodeConfig{
				Name: "claude-code-primary",
			},
			wantErr: false,
		},
		{
			name: "valid config with zero ContextLimit",
			config: ClaudeCodeConfig{
				ContextLimit: 0,
			},
			wantErr: false,
		},
		{
			name: "valid config with zero MaxTokens",
			config: ClaudeCodeConfig{
				MaxTokens: 0,
			},
			wantErr: false,
		},
		{
			name: "negative ContextLimit returns error",
			config: ClaudeCodeConfig{
				ContextLimit: -100,
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationOutOfRange,
			wantFieldInMsg: "ContextLimit",
		},
		{
			name: "negative MaxTokens returns error",
			config: ClaudeCodeConfig{
				MaxTokens: -500,
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationOutOfRange,
			wantFieldInMsg: "MaxTokens",
		},
		{
			name: "Name with spaces returns error",
			config: ClaudeCodeConfig{
				Name: "my claude backend",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with tab returns error",
			config: ClaudeCodeConfig{
				Name: "claude\tcode",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with newline returns error",
			config: ClaudeCodeConfig{
				Name: "claude\ncode",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with @ returns error",
			config: ClaudeCodeConfig{
				Name: "claude@code",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with # returns error",
			config: ClaudeCodeConfig{
				Name: "claude#code",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with $ returns error",
			config: ClaudeCodeConfig{
				Name: "claude$code",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with % returns error",
			config: ClaudeCodeConfig{
				Name: "claude%code",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with parentheses returns error",
			config: ClaudeCodeConfig{
				Name: "claude(code)",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with brackets returns error",
			config: ClaudeCodeConfig{
				Name: "claude[code]",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with braces returns error",
			config: ClaudeCodeConfig{
				Name: "claude{code}",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with pipe returns error",
			config: ClaudeCodeConfig{
				Name: "claude|code",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with backslash returns error",
			config: ClaudeCodeConfig{
				Name: "claude\\code",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
		{
			name: "Name with angle brackets returns error",
			config: ClaudeCodeConfig{
				Name: "claude<code>",
			},
			wantErr:        true,
			wantCode:       werrors.ErrValidationInvalidValue,
			wantFieldInMsg: "Name",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()

			if tt.wantErr {
				if err == nil {
					t.Fatal("expected error, got nil")
				}

				// Verify error code
				if err.Code != tt.wantCode {
					t.Errorf("error code = %q, want %q", err.Code, tt.wantCode)
				}

				// Verify error message contains the field name
				if tt.wantFieldInMsg != "" {
					if !strings.Contains(err.Message, tt.wantFieldInMsg) {
						t.Errorf("error message should contain field name %q, got: %s",
							tt.wantFieldInMsg, err.Message)
					}
				}

				// Verify error has context with the field name
				if tt.wantFieldInMsg != "" {
					fieldVal, hasField := err.Context["field"]
					if !hasField {
						t.Error("error context should contain 'field' key")
					} else if fieldVal != tt.wantFieldInMsg {
						t.Errorf("context['field'] = %q, want %q", fieldVal, tt.wantFieldInMsg)
					}
				}

				// Verify error has suggestions
				if len(err.Suggestions) == 0 {
					t.Error("error should have at least one suggestion")
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
			}
		})
	}
}

// TestClaudeCodeConfig_Validate_SuggestionsContent verifies that validation errors
// include helpful, actionable suggestions.
func TestClaudeCodeConfig_Validate_SuggestionsContent(t *testing.T) {
	t.Run("invalid name suggestions mention valid characters", func(t *testing.T) {
		cfg := ClaudeCodeConfig{Name: "invalid name"}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for invalid name")
		}

		foundCharSuggestion := false
		foundExampleSuggestion := false
		for _, s := range err.Suggestions {
			if strings.Contains(s, "alphanumeric") ||
				strings.Contains(s, "hyphens") ||
				strings.Contains(s, "underscores") {
				foundCharSuggestion = true
			}
			if strings.Contains(s, "Example") ||
				strings.Contains(s, "example") ||
				strings.Contains(s, "my-claude") ||
				strings.Contains(s, "claude_code") {
				foundExampleSuggestion = true
			}
		}

		if !foundCharSuggestion {
			t.Error("suggestions should mention valid character types (alphanumeric, hyphens, underscores)")
		}
		if !foundExampleSuggestion {
			t.Error("suggestions should include an example of a valid name")
		}
	})

	t.Run("negative ContextLimit suggestions mention valid range", func(t *testing.T) {
		cfg := ClaudeCodeConfig{ContextLimit: -100}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for negative ContextLimit")
		}

		foundRangeSuggestion := false
		foundDefaultSuggestion := false
		for _, s := range err.Suggestions {
			if strings.Contains(s, "0") && strings.Contains(s, "greater") {
				foundRangeSuggestion = true
			}
			if strings.Contains(s, "default") || strings.Contains(s, "200000") {
				foundDefaultSuggestion = true
			}
		}

		if !foundRangeSuggestion {
			t.Error("suggestions should mention valid range (0 or greater)")
		}
		if !foundDefaultSuggestion {
			t.Error("suggestions should mention default value")
		}
	})

	t.Run("negative MaxTokens suggestions mention valid range", func(t *testing.T) {
		cfg := ClaudeCodeConfig{MaxTokens: -500}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for negative MaxTokens")
		}

		foundRangeSuggestion := false
		foundDefaultSuggestion := false
		for _, s := range err.Suggestions {
			if strings.Contains(s, "0") && strings.Contains(s, "greater") {
				foundRangeSuggestion = true
			}
			if strings.Contains(s, "default") || strings.Contains(s, "25000") {
				foundDefaultSuggestion = true
			}
		}

		if !foundRangeSuggestion {
			t.Error("suggestions should mention valid range (0 or greater)")
		}
		if !foundDefaultSuggestion {
			t.Error("suggestions should mention default value")
		}
	})
}

// TestClaudeCodeConfig_Validate_ErrorContext verifies that validation errors
// include appropriate context information.
func TestClaudeCodeConfig_Validate_ErrorContext(t *testing.T) {
	t.Run("invalid name error includes value in context", func(t *testing.T) {
		invalidName := "bad name!"
		cfg := ClaudeCodeConfig{Name: invalidName}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for invalid name")
		}

		valueCtx, hasValue := err.Context["value"]
		if !hasValue {
			t.Error("error context should contain 'value' key")
		} else if valueCtx != invalidName {
			t.Errorf("context['value'] = %q, want %q", valueCtx, invalidName)
		}
	})

	t.Run("out of range error includes value and range in context", func(t *testing.T) {
		cfg := ClaudeCodeConfig{ContextLimit: -100}
		err := cfg.Validate()
		if err == nil {
			t.Fatal("expected error for negative ContextLimit")
		}

		// Should have value in context
		if _, hasValue := err.Context["value"]; !hasValue {
			t.Error("error context should contain 'value' key")
		}

		// Should have min in context
		if _, hasMin := err.Context["min"]; !hasMin {
			t.Error("error context should contain 'min' key")
		}

		// Should have max in context
		if _, hasMax := err.Context["max"]; !hasMax {
			t.Error("error context should contain 'max' key")
		}
	})
}

// TestClaudeCodeConfig_Validate_Category verifies that validation errors
// have the correct category.
func TestClaudeCodeConfig_Validate_Category(t *testing.T) {
	tests := []struct {
		name   string
		config ClaudeCodeConfig
	}{
		{
			name:   "invalid name",
			config: ClaudeCodeConfig{Name: "bad name"},
		},
		{
			name:   "negative ContextLimit",
			config: ClaudeCodeConfig{ContextLimit: -1},
		},
		{
			name:   "negative MaxTokens",
			config: ClaudeCodeConfig{MaxTokens: -1},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.config.Validate()
			if err == nil {
				t.Fatal("expected error")
			}

			if err.Category != werrors.CategoryValidation {
				t.Errorf("error category = %q, want %q", err.Category, werrors.CategoryValidation)
			}
		})
	}
}

// TestClaudeCodeConfig_Validate_FirstErrorReturned verifies that validation
// returns on the first error encountered (Name is checked before numeric fields).
func TestClaudeCodeConfig_Validate_FirstErrorReturned(t *testing.T) {
	cfg := ClaudeCodeConfig{
		Name:         "bad name",    // Invalid
		ContextLimit: -100,          // Also invalid
		MaxTokens:    -500,          // Also invalid
	}

	err := cfg.Validate()
	if err == nil {
		t.Fatal("expected error")
	}

	// Name is validated first, so error should be about Name
	if err.Code != werrors.ErrValidationInvalidValue {
		t.Errorf("expected ErrValidationInvalidValue for Name error, got %q", err.Code)
	}
	if fieldCtx, ok := err.Context["field"]; ok && fieldCtx != "Name" {
		t.Errorf("expected first error to be about Name field, got %q", fieldCtx)
	}
}
