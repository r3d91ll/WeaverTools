// Package yarn manages conversations, measurements, and data storage.
package yarn

// ValidationError represents a validation failure for Yarn types.
// This mirrors Wool's ValidationError but keeps packages independent.
type ValidationError struct {
	Field   string
	Message string
}

// Error implements the error interface.
func (e *ValidationError) Error() string {
	return e.Field + ": " + e.Message
}
